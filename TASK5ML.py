# ==============================
# TASK-05: Food Recognition + Calories (ONLINE - TFDS)
# ==============================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
import os


IMG_SIZE = 128
BATCH_SIZE = 32

# ==============================
# 1. LOAD DATASET (ONLINE)
# ==============================
print("Downloading Food101 dataset... (first time only)")

(dataset_train, dataset_test), dataset_info = tfds.load(
    "food101",
    split=['train[:10%]', 'validation[:10%]'],  # 🔥 small subset (fast)
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = dataset_info.features['label'].num_classes
class_names = dataset_info.features['label'].names

# ==============================
# 2. PREPROCESS FUNCTION
# ==============================
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

dataset_train = dataset_train.map(preprocess).batch(BATCH_SIZE)
dataset_test = dataset_test.map(preprocess).batch(BATCH_SIZE)

# ==============================
# 3. BUILD MODEL
# ==============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 4. TRAIN MODEL
# ==============================
print("\nTraining...\n")

model.fit(
    dataset_train,
    epochs=3,
    validation_data=dataset_test
)

# ==============================
# 5. CALORIE MAP (SAMPLE)
# ==============================
calorie_dict = {
    "pizza": 266,
    "hamburger": 295,
    "ice_cream": 207,
    "salad": 152,
    "steak": 271
}

# ==============================
# 6. PREDICTION
# ==============================
def predict_food(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(img)
    class_index = np.argmax(pred)

    food_name = class_names[class_index]
    calories = calorie_dict.get(food_name, "Unknown")

    print("\n========== RESULT ==========")
    print("Food:", food_name)
    print("Calories:", calories, "kcal")
    print("============================")

# ==============================
# 7. TEST
# ==============================
if os.path.exists("test.jpg"):
    predict_food("test.jpg")
else:
    print("\n⚠️ Put a test image named test.jpg")

# ==============================
# 8. SAVE MODEL
# ==============================
model.save("food_model_tfds.h5")
print("\n✅ Model saved")