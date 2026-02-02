# Artificial Neural Network (ANN) using PyTorch

This project demonstrates the implementation of an **Artificial Neural Network (ANN)** for **binary classification** using **PyTorch**.  
The model is trained on a synthetic dataset and uses a fully connected feed-forward neural network with ReLU activations.

The goal of this implementation is to understand the complete **training pipeline** in PyTorch, including data preparation, model definition, loss calculation, optimization, evaluation, and model saving.

---

## Model Architecture

The ANN consists of the following layers:

- Input Layer: 3 features
- Hidden Layer 1: 5 neurons + ReLU
- Hidden Layer 2: 5 neurons + ReLU
- Output Layer: 1 neuron (logit output)

```text
Input (3)
  ↓
Linear (3 → 5) + ReLU
  ↓
Linear (5 → 5) + ReLU
  ↓
Linear (5 → 1)
