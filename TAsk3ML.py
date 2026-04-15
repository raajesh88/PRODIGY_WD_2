import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Paths
cats_path = "dogcat/cats"
dogs_path = "dogcat/dogs"

data = []
labels = []

# Load cat images
for img_name in os.listdir(cats_path):
    img_path = os.path.join(cats_path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print("Skipping:", img_path)
        continue

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    img = img.flatten()  # convert to 1D array

    data.append(img)
    labels.append(0)  # cat = 0

# Load dog images
for img_name in os.listdir(dogs_path):
    img_path = os.path.join(dogs_path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print("Skipping:", img_path)
        continue

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()

    data.append(img)
    labels.append(1)  # dog = 1

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)