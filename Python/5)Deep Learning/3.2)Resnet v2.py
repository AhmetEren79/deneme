# drn_pedestrian_balanced_fix.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

torch.backends.cudnn.benchmark = True

# =======================
# Device & seed
# =======================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
print("Device:", device)
torch.manual_seed(42)
np.random.seed(42)

# =======================
# Dataset utils (STRICT)
# =======================
def read_images_strict(path):
    """
    Klasördeki TÜM görselleri okur.
    - Yüklenemeyen/yanlış boyutlu dosya -> HATA.
    - Boyut (W,H)=(32,64) beklenir (reshape(64,32) ile uyumlu).
    - Gri değilse 'L' e çevrilir.
    Dönüş: np.ndarray [N, 64*32] (uint8)
    """
    file_list = [f for f in os.listdir(path) if not f.startswith('.')]
    file_list.sort()
    if len(file_list) == 0:
        raise RuntimeError(f"Boş klasör: {path}")

    arr = np.zeros((len(file_list), 64 * 32), dtype=np.uint8)
    for i, fname in enumerate(file_list):
        img_path = os.path.join(path, fname)
        with Image.open(img_path) as img:
            if img.mode != 'L':
                img = img.convert('L')
            if img.size != (32, 64):  # PIL size = (W,H)
                raise ValueError(f"Beklenen boyut (W,H)=(32,64); bulundu={img.size} @ {img_path}")
            data_u8 = np.asarray(img, dtype=np.uint8)  # (64,32)
            arr[i, :] = data_u8.flatten()
    return arr

# =======================
# Yol ayarları
# =======================
train_negative_path = r"LSIFIR\Classification\Train\neg"
train_positive_path = r"LSIFIR\Classification\Train\pos"
test_negative_path  = r"LSIFIR\Classification\Test\neg"
test_positive_path  = r"LSIFIR\Classification\Test\pos"

# =======================
# Veriyi oku (STRICT)
# =======================
train_negative_array = read_images_strict(train_negative_path)
train_positive_array = read_images_strict(train_positive_path)
test_negative_array  = read_images_strict(test_negative_path)
test_positive_array  = read_images_strict(test_positive_path)

x_train_neg = torch.from_numpy(train_negative_array)   # uint8
y_train_neg = torch.zeros(train_negative_array.shape[0], dtype=torch.long)
x_train_pos = torch.from_numpy(train_positive_array)
y_train_pos = torch.ones(train_positive_array.shape[0], dtype=torch.long)

x_test_neg = torch.from_numpy(test_negative_array)
y_test_neg = torch.zeros(test_negative_array.shape[0], dtype=torch.long)
x_test_pos = torch.from_numpy(test_positive_array)
y_test_pos = torch.ones(test_positive_array.shape[0], dtype=torch.long)

x_train = torch.cat([x_train_neg, x_train_pos], dim=0)
y_train = torch.cat([y_train_neg, y_train_pos], dim=0)
x_test  = torch.cat([x_test_neg,  x_test_pos],  dim=0)
y_test  = torch.cat([y_test_neg,  y_test_pos],  dim=0)

print("x_train:", x_train.size(), "y_train:", y_train.size())
print("x_test :", x_test.size(),  "y_test :", y_test.size())

# =======================
# Hiperparametreler
# =======================
num_epochs   = 20
batch_size   = 128
learning_rate = 1e-3

# =======================
# DataLoader (Weighted Sampler ONLY)
# =======================
num_neg = (y_train == 0).sum().item()
num_pos = (y_train == 1).sum().item()
print(f"Train counts -> neg: {num_neg}, pos: {num_pos}")

# Sampler ağırlıkları: sınıf frekansının tersi
w_neg = 1.0 / (num_neg + 1e-9)
w_pos = 1.0 / (num_pos + 1e-9)
print(f"Sampler weights -> w_neg={w_neg:.6e}, w_pos={w_pos:.6e}")

sample_weights = torch.where(
    y_train == 1,
    torch.full_like(y_train, fill_value=w_pos, dtype=torch.float32),
    torch.full_like(y_train, fill_value=w_neg, dtype=torch.float32),
).double()  # WeightedRandomSampler double ister

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_ds = data.TensorDataset(x_train, y_train)
test_ds  = data.TensorDataset(x_test,  y_test)

trainloader = data.DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              shuffle=False, num_workers=0, pin_memory=use_gpu)
testloader  = data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=use_gpu)

# =======================
# DRN (ResNet-vari)
# =======================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, p_drop=0.0):  # dropout=0.0
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(p_drop)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = ResNet(BasicBlock, [2, 2, 2]).to(device)

# =======================
# Loss, Optimizer, Scheduler
# =======================
criterion = nn.CrossEntropyLoss()  # <- class weight YOK (sadece sampler kullanıyoruz)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# =======================
# Training
# =======================
loss_list = []
train_acc_list = []
test_acc_list  = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, leave=False)
    for images_u8, labels in pbar:
        images = images_u8.view(images_u8.size(0), 1, 64, 32).float() / 255.0  # sadece /255.0
        if use_gpu:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()

    # --- Train Accuracy + pos rate ---
    model.eval()
    correct = 0
    total = 0
    pos_count = 0
    with torch.no_grad():
        for images_u8, labels in trainloader:
            images = images_u8.view(images_u8.size(0), 1, 64, 32).float() / 255.0
            if use_gpu:
                images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            pos_count += (preds == 1).sum().item()
    train_acc = 100.0 * correct / total
    train_pos_rate = 100.0 * pos_count / total
    train_acc_list.append(train_acc)

    # --- Test Accuracy + rapor ---
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images_u8, labels in testloader:
            images = images_u8.view(images_u8.size(0), 1, 64, 32).float() / 255.0
            if use_gpu:
                images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    test_acc = 100.0 * correct / total
    test_acc_list.append(test_acc)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    test_pos_rate = (all_preds == 1).mean() * 100.0
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}% | Test Acc = {test_acc:.2f}%")
    print(f"  Train pos-rate: {train_pos_rate:.2f}% | Test pos-rate: {test_pos_rate:.2f}%")
    print("  Confusion matrix:\n", cm)

    loss_list.append(running_loss / max(1, len(trainloader)))

# =======================
# Görselleştirme
# =======================
fig, ax1 = plt.subplots()
ax1.plot(loss_list, label="Loss")
ax2 = ax1.twinx()
ax2.plot(np.array(test_acc_list)/100.0, label="Test Acc")
ax2.plot(np.array(train_acc_list)/100.0, label="Train Acc")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title("Loss vs Accuracy")
plt.tight_layout()
plt.show()

# Son rapor:
print("\n=== Final classification report (TEST) ===")
print(classification_report(all_labels, all_preds, digits=3))
