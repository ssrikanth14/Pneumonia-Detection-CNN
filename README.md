# 🩺 Pneumonia Detection using Convolutional Neural Networks (CNN)

## 📌 Project Overview

This project uses a **Convolutional Neural Network (CNN)** to classify Chest X-ray images into two categories:

- ✅ Normal
- 🫁 Pneumonia

The model was built using **TensorFlow/Keras** and deployed using **Streamlit** to provide an easy-to-use web interface.

---

## 🚀 Features

- Chest X-ray Image Classification
- CNN built from scratch
- Data Augmentation
- EarlyStopping
- ModelCheckpoint
- Streamlit Web Application
- Real-time Prediction
- Confidence Score Display

---

## 📂 Dataset

Dataset Used:

**Chest X-Ray Images (Pneumonia)**

Dataset contains

- Training Images
- Validation Images
- Testing Images

Classes

- NORMAL
- PNEUMONIA

---

## 🏗️ Project Structure

```text
Pneumonia-Detection-CNN/
│
├── app.py
├── README.md
├── requirements.txt
├── notebooks/
├── models/
├── assets/
└── dataset/
```

---

## 🧠 CNN Architecture

Input Image

↓

Conv2D (32)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Flatten

↓

Dense (128)

↓

Dropout (0.5)

↓

Output Layer (Sigmoid)

---

## 📊 Model Performance

### Initial Model

| Metric | Value |
|--------|-------|
| Test Accuracy | 71.63% |

---

### Improved Model

Techniques Applied

- Data Augmentation
- EarlyStopping
- ModelCheckpoint

| Metric | Value |
|--------|-------|
| Test Accuracy | **87.34%** |

---

## 📈 Classification Report

| Class | Precision | Recall | F1-score |
|--------|----------:|-------:|---------:|
| NORMAL | 0.88 | 0.77 | 0.82 |
| PNEUMONIA | 0.87 | 0.94 | 0.90 |

---

## 🖼️ Application Screenshots

### Home Page

![Home](assets/home.png)

---

### Prediction

![Prediction](assets/prediction.png)

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ssrikanth14/Pneumonia-Detection-CNN.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Pillow

---

## 🔮 Future Improvements

- Transfer Learning (EfficientNet/MobileNetV2)
- Grad-CAM Visualization
- Better Hyperparameter Tuning
- Multi-class Chest Disease Classification
- Cloud Deployment

---

## 👨‍💻 Author

**Srikanth**

GitHub:
https://github.com/ssrikanth14