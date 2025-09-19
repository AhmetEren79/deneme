import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# === 1. VERİ YÜKLEME VE CLAHE ÖN İŞLEME ===
data_dir = "C:/Users/AHMET EREN/Downloads/archive/NINS_Dataset/NINS_Dataset"
img_size = 224
data = []
labels = []

def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size))

    # Gri yap ve CLAHE uygula
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Tekrar 3 kanala çevir (CNN için gerekli)
    final_img = cv2.merge([enhanced, enhanced, enhanced])

    # Normalizasyon
    final_img = final_img.astype(np.float32) / 255.0
    return final_img

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path) or folder.lower() == 'models':
        continue

    label = 0 if folder.lower() == 'normal' else 1

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(folder_path, file)
        try:
            img = preprocess_image(img_path, img_size)
            if img is None:
                print(f"BOŞ/BİÇİMSİZ: {img_path}")
                continue
            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"HATA: {img_path} → {e}")

X = np.array(data)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Veri hazır. Toplam: {len(X)} | Eğitim: {len(X_train)} | Test: {len(X_test)}")

# === 2. MODELİ OLUŞTUR VE EĞİT ===

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# === 3. EĞİTİM SONUÇLARINI GÖRSELLEŞTİR ===

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title("Doğruluk")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title("Kayıp")
plt.legend()

plt.tight_layout()
plt.show()

# === 4. RESİM ÜZERİNDEN TAHMİN FONKSİYONU ===

def predict_image(img_path):
    try:
        img = preprocess_image(img_path, img_size)
        if img is None:
            return "Resim yüklenemedi."
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0][0]
        durum = "Tümör VAR" if prediction > 0.5 else "Tümör YOK"
        return f"{img_path} → {durum} ({prediction:.2f})"
    except Exception as e:
        return f"HATA: {e}"

# === ÖRNEK KULLANIM ===
test_path = "C:/Users/AHMET EREN/Downloads/archive/NINS_Dataset/NINS_Dataset/pituitary tumor/47088606.jpg"
print(predict_image(test_path))
