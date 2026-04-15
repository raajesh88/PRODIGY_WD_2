# ==============================
# TASK-04: Hand Gesture Recognition (LeapGestRecog)
# ==============================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==============================
# 1. DATASET PATH
# ==============================
dataset_path = r"D:\PRODIGY_WD_2\dataset\leapGestRecog"  # 🔥 change if needed

IMG_SIZE = 64

data = []
labels = []

print("Loading dataset...")

# ==============================
# 2. LOAD DATA
# ==============================
for subject in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        # Extract label (00_palm → 0)
        label = int(gesture.split('_')[0])

        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data.append(img)
            labels.append(label)

# ==============================
# 3. PREPROCESS DATA
# ==============================
data = np.array(data)
labels = np.array(labels)

# Normalize
data = data / 255.0

# Reshape for CNN
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# One-hot encoding
labels = to_categorical(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# ==============================
# 4. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ==============================
# 5. BUILD MODEL
# ==============================
num_classes = labels.shape[1]   # 🔥 AUTO FIX

model = Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')  # ✅ FIXED
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 6. TRAIN MODEL
# ==============================
print("\nTraining started...\n")

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==============================
# 7. EVALUATE MODEL
# ==============================
loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc)

# ==============================
# 8. PREDICTION FUNCTION
# ==============================
def predict_sample(index=0):
    sample = X_test[index]
    sample = np.reshape(sample, (1, IMG_SIZE, IMG_SIZE, 1))

    prediction = model.predict(sample)
    predicted_class = np.argmax(prediction)

    print("\nPredicted Class:", predicted_class)
    print("Actual Class:", np.argmax(y_test[index]))

predict_sample(0)

# ==============================
# 9. SAVE MODEL
# ==============================
model.save("gesture_model.h5")
print("\n✅ Model saved as gesture_model.h5")