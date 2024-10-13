# Regression-with-Variational-Quantum-Circuits-Higher-Dimention-
This repository implements a **multidimensional regression model** using **Variational Quantum Circuits (VQCs)**, combining quantum neural networks with classical optimization techniques. It demonstrates how quantum circuits can approximate complex functions, with performance evaluated using 3D visualizations and R² metrics.

# Multidimensional Regression with Variational Quantum Circuits (VQC)

## Introduction

This project implements a **multidimensional regression model** using **Variational Quantum Circuits (VQCs)**. The goal is to approximate a two-dimensional target function using a quantum neural network, leveraging **quantum machine learning** techniques to capture complex relationships in data. This approach combines quantum circuits with classical optimization methods, demonstrating a hybrid quantum-classical system for regression tasks.

## Quantum Circuits Overview

### Quantum Embedding
The input features are encoded into quantum states using angle embedding. The features are mapped to rotation angles of quantum gates, allowing classical data to be represented in a quantum state. This enables the use of quantum operations to process and manipulate the data.

### Quantum Circuit Layers
The quantum circuit consists of several layers of parameterized quantum gates, specifically **Strongly Entangling Layers**, which entangle the qubits and apply rotations. Entanglement is essential as it captures the correlations between input features. These gates are the building blocks of the variational quantum circuit and are trainable during optimization.

### Quantum Neural Network
The quantum neural network (QNode) is a parameterized quantum circuit that takes input features and returns the expectation value of a quantum measurement. This output represents the model's prediction. The circuit is trained to approximate the target function by optimizing its parameters to minimize prediction error.

## Target Function

The target function is a simple quadratic function of the input variables. The goal of the model is to approximate this function using the quantum neural network.

## Optimization Process

### Loss Function
The loss function used in this project is the mean squared error (MSE), which measures the difference between the predicted outputs of the quantum circuit and the true values of the target function. The MSE is minimized during the training process to improve the model's predictions.

### Optimizer (Adam)
The **Adam optimizer** is used to update the parameters of the quantum circuit. Adam is well-suited for this task because it handles noisy gradients effectively, which can arise from quantum measurements.

### JAX and Just-in-Time Compilation
The optimization process is accelerated using **JAX**'s Just-in-Time (JIT) compilation. JIT helps optimize the computational graph and speeds up parameter updates, which is crucial when simulating quantum circuits.

## Training Process
The variational quantum circuit is trained over several iterations, where the parameters are updated using the gradients of the loss function. Each step of the optimization aims to reduce the error between the model's predictions and the target values. After the training process, the optimized parameters are used to make predictions on the input data.

## Evaluation and Results

After training, the model's performance is evaluated using the **R² score**, a standard metric for regression tasks. The model's predictions are compared with the target function, and the accuracy is visualized using 3D surface plots. 

- **Target Function**: The ground truth function used for training.
- **Predicted Function**: The function predicted by the quantum neural network after training.
- **R² Score**: A performance metric indicating how well the model approximates the target function.

## Technologies Used

- **Pennylane**: For constructing and simulating quantum circuits.
- **JAX**: For automatic differentiation and just-in-time (JIT) compilation.
- **Optax**: For gradient-based optimization.
- **Matplotlib**: For 3D visualizations of the target and predicted functions.
- **Scikit-learn**: For computing the R² score to evaluate model performance.

## Installation and Setup

### Prerequisites
Make sure you have Python 3.8+ installed, along with the required libraries:

```bash
pip install pennylane jax optax matplotlib scikit-learn
