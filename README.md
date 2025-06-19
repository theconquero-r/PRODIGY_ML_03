# 🧠 Prodigy ML Internship – Task 03

## 📌 Task Title: Image Classification using Support Vector Machine (SVM)

---

## 📝 Task Description

As part of my Machine Learning Internship at **Prodigy InfoTech**, I implemented an **SVM-based Image Classifier** to distinguish between **Cats and Dogs** using grayscale image data.

---

## 🔍 Objective

To build a binary image classification model using **Support Vector Machine (SVM)** that accurately predicts whether a given image contains a cat or a dog.

---

## 🛠️ Technologies Used

- Python
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib

---

## 📂 Project Folder Structure

PRODIGY_SE_03/
│
├── PRODIGY_SE_03.py # Main Python script
├── README.md # Project documentation
├── dataset/ # Folder containing cat and dog images
│ ├── cat.1.jpg
│ ├── dog.1.jpg
│ └── ... (more images)
├── output/ # (Optional) Saved visualization outputs
└── requirements.txt # Python dependencies


---

## 🧠 Dataset

- Source: [Kaggle - Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- The dataset contains thousands of labeled images of cats and dogs.

---

## 🔄 Workflow Overview

1. Loaded grayscale images and resized them to **64x64** pixels.
2. Flattened images into feature vectors.
3. Labeled data: **0 for Cat**, **1 for Dog**.
4. Split data into training and test sets.
5. Trained an **SVM classifier** using a linear kernel.
6. Evaluated accuracy and performance using a **classification report**.
7. Visualized true vs. predicted labels on random test samples.

---

## 📸 Output Visualization

The following plot shows 10 randomly selected test images along with their **True** and **Predicted** labels.

![Predicted Samples](output/svm_visualization.png)

---

## 🚀 How to Run

1. Clone the repository or download the files.
2. Place your cat/dog images in the `dataset/` folder.
3. Install dependencies:

```bash
pip install -r requirements.txt

python svm_classifier.py

🙏 Acknowledgment
Grateful to Prodigy InfoTech for this hands-on and enriching internship experience, where I got to apply machine learning techniques to real-world image classification tasks.

👨‍💻 Author
Karan Shakya
Intern at Prodigy InfoTech
GitHub: @theconquero-r