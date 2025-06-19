# PRODIGY_SE_03 - SVM Image Classifier for Cats vs Dogs

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import random

# ========== Step 1: Load and Preprocess Dataset ==========
data = []
labels = []

# Directory containing images (should contain images named with 'cat' or 'dog' in filename)
directory = 'dataset/'
image_size = 64  # Resize all images to 64x64

print("[INFO] Loading images...")
for img_name in os.listdir(directory):
    img_path = os.path.join(directory, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (image_size, image_size))
        data.append(img.flatten())  # Flatten image to 1D vector
        # Label encoding: 0 for Cat, 1 for Dog
        if 'cat' in img_name.lower():
            labels.append(0)
        elif 'dog' in img_name.lower():
            labels.append(1)

data = np.array(data)
labels = np.array(labels)

print(f"[INFO] Total images loaded: {len(data)}")
print(f"[INFO] Image vector shape: {data[0].shape}")

# ========== Step 2: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ========== Step 3: Train SVM Model ==========
print("[INFO] Training SVM model...")
svm_model = SVC(kernel='linear')  # Linear kernel for classification
svm_model.fit(X_train, y_train)

# ========== Step 4: Evaluate Model ==========
y_pred = svm_model.predict(X_test)

print("\n[RESULT] Accuracy:", accuracy_score(y_test, y_pred))
print("\n[RESULT] Classification Report:\n", classification_report(y_test, y_pred))

# ========== Step 5: Visualize Some Predictions ==========
plt.figure(figsize=(12, 6))
for i in range(10):
    index = random.randint(0, len(X_test) - 1)
    image = X_test[index].reshape(image_size, image_size)
    true_label = "Cat" if y_test[index] == 0 else "Dog"
    predicted_label = "Cat" if y_pred[index] == 0 else "Dog"

    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis('off')


plt.tight_layout()

# ✅ Save the figure
os.makedirs("output", exist_ok=True)  
plt.savefig("output/svm_visualization.png")


plt.show()
