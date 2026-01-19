# 😷 Face Mask Detection Using Deep Learning & CNN

🚀 A hands-on deep learning project implementing a **Convolutional Neural Network (CNN)** for detecting whether individuals are wearing face masks in images.

This project demonstrates practical applications of **Computer Vision and Deep Learning**, including **data preprocessing, CNN model building, training, evaluation, and visualization of predictions**.

---

![Project Thumbnail](face_mask_detection.jpg)

---

## 📌 Overview

* **Goal**: Build a CNN to classify face images as **masked** or **unmasked**.
* **Focus Areas**:

  * Image preprocessing and resizing
  * CNN model design and improvement
  * Training and validation analysis
  * Evaluation using precision, recall, F1-score
  * Visualization of predictions
* **Framework Used**: TensorFlow + Keras
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

---

## 🧠 Problem Statement

* Automatically classify face images into **masked** and **unmasked** categories.
* Useful for **public health monitoring**, **safety compliance**, and **automated surveillance**.

---

## 🛠️ Technologies Used

* Python 🐍
* NumPy & Pandas 📊
* Matplotlib & Seaborn 📈
* TensorFlow & Keras 🧠
* scikit-learn ⚙️

---

## ✨ Key Features

* 🖼️ **Image Preprocessing**: Resizing, normalization, and data augmentation
* 🔁 **CNN Architecture**: Default and improved models with multiple convolutional layers, pooling, dropout, and fully connected layers
* 📊 **Training Visualization**: Accuracy and loss plots for both training and validation sets
* 🧮 **Evaluation Metrics**: Precision, recall, F1-score, and classification report
* 🖼️ **Prediction Visualization**: Displays test images with true and predicted labels

---

## 🧩 Project Structure

### 🔹 Default CNN Model

* **Architecture**:

```
Input → Conv2D (32) → MaxPooling2D → Conv2D (64) → Conv2D (64) → MaxPooling2D → Flatten → Dense (128) → Output (2, Softmax)
```

* Serves as a **baseline model** to evaluate performance before improvements.

---

### 🔹 Improved CNN Model

* **Improvements Over Default Model**:

  * **Data Augmentation**: Rotation, zoom, shift, horizontal flips
  * **Additional Convolutional Layer** for richer feature extraction
  * **Dropout Layers** to prevent overfitting
  * **Increased Filters** to capture more complex patterns

* **Architecture**:

```
Input → Conv2D (32) → MaxPooling2D → Conv2D (64) → Conv2D (64) → MaxPooling2D → Conv2D (128) → MaxPooling2D → Flatten → Dense (128) + Dropout → Output (2, Softmax)
```

---

### 🔹 Training & Evaluation

* **Metrics Tracked**: Training/Validation Accuracy and Loss
* **Classification Metrics**: Precision, Recall, F1-score (weighted and per-class)
* **Improved Model Performance**: Visualized through side-by-side precision and recall comparison plots

---

### 🔹 Prediction Visualization

* Select **5 images predicted as masked** and **5 images predicted as unmasked** from the test set
* Display images along with **true and predicted labels**
* Provides an intuitive understanding of model performance on individual samples

---

## 📈 Training & Evaluation Highlights

| Model        | Accuracy | Precision | Recall   | F1-Score |
| ------------ | -------- | --------- | -------- | -------- |
| Default CNN  | Moderate | Moderate  | Moderate | Moderate |
| Improved CNN | Higher   | Improved  | Improved | Improved |

> *Exact values vary depending on training runs due to data shuffling and random initialization.*

---

## 📁 Files

* `Face_Mask_Detection_CNN.ipynb` – Full notebook with code, outputs, plots, and explanations

---

## 🧾 License

This project is intended for **learning and demonstration purposes**.
Ensure you have appropriate permissions for datasets if used for commercial purposes.

---

## 🤝 Connect

- [LinkedIn](https://www.linkedin.com/in/varsha-shekhar)
- [Gmail](varshaiyer96@gmail.com)
