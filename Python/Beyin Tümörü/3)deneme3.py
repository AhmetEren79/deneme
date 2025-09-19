import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# === 1. VERİ YÜKLEME ===
data_dir = "C:/Users/AHMET EREN/Downloads/archive/NINS_Dataset/NINS_Dataset"
img_size = 160
data = []
labels = []

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
            img = cv2.imread(img_path)
            if img is None:
                print(f"BOŞ/BİÇİMSİZ: {img_path}")
                continue
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"HATA: {img_path} → {e}")

X = np.array(data) / 255.0
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Veri hazır. Toplam: {len(X)} | Eğitim: {len(X_train)} | Test: {len(X_test)}")

# === 2. VERİ ARTIRMA ===
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# === 3. MODELİ OLUŞTUR VE EĞİT ===
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# === 4. EĞİTİM SONUÇLARINI GÖRSELLEŞTİR ===
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title("Doğruluk")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title("Kayıp")
plt.legend()

plt.tight_layout()
plt.show()

# === 5. RESİM ÜZERİNDEN TAHMİN FONKSİYONU ===
def predict_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return "Resim yüklenemedi."
        img = cv2.resize(img, (img_size, img_size)) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0][0]
        durum = "Tümör VAR" if prediction > 0.5 else "Tümör YOK"
        return f"{img_path} → {durum} ({prediction:.2f})"
    except Exception as e:
        return f"HATA: {e}"

# === ÖRNEK KULLANIM ===
test_path = "C:/Users/AHMET EREN/Downloads/archive/NINS_Dataset/NINS_Dataset/pituitary tumor/47088606.jpg"
print(predict_image(test_path))
