# Multiclass Classification using Artificial Neural Network (ANN)

This project demonstrates how to build, train, evaluate, and save a **multiclass classification model** using a simple **Artificial Neural Network (ANN)** in **PyTorch**.

Synthetic data is generated using `sklearn.make_blobs`, and the model learns to classify samples into **3 different classes**.

---

## 📌 Features

- Multiclass classification (3 classes)
- Fully connected ANN (MLP)
- Trained using **CrossEntropyLoss**
- Accuracy and loss tracking
- Model saving with `torch.save`
- Clean and beginner-friendly PyTorch code

---

## 🧠 Model Architecture

The ANN consists of:


### Details:
- **Input features:** 3
- **Hidden units:** 8
- **Output classes:** 3
- **Activation:** ReLU
- **Loss function:** CrossEntropyLoss
- **Optimizer:** SGD

> ⚠️ Note: No `Softmax` is used in the model because `CrossEntropyLoss` internally applies it.

---

## 📊 Dataset

The dataset is synthetically generated using:

```python
sklearn.datasets.make_blobs
