# Multiclass Perceptron Classification (PyTorch)

This project implements a **multiclass Perceptron classifier** using **PyTorch**.  
It demonstrates how a single linear layer can be trained to classify data into multiple classes using **CrossEntropyLoss**.

Synthetic data is generated using `sklearn.make_blobs`.

---

## 📌 Overview

- Multiclass classification (3 classes)
- Linear Perceptron model
- Trained using gradient descent
- Loss and accuracy tracking
- Model saving and loading

---

## 🧠 Model Architecture

The Perceptron consists of **one linear layer**:


### Model Details
- **Input features:** 3  
- **Output classes:** 3  
- **Hidden layers:** None  
- **Activation:** None  
- **Loss function:** CrossEntropyLoss  

> ⚠️ Softmax is **not applied** in the model.  
> `CrossEntropyLoss` internally handles it.

---

## 📊 Dataset

The dataset is synthetically generated using:

```python
sklearn.datasets.make_blobs
