# Regression-with-Variational-Quantum-Circuits-Higher-Dimention-
This repository implements a **multidimensional regression model** using **Variational Quantum Circuits (VQCs)**, combining quantum neural networks with classical optimization techniques. It demonstrates how quantum circuits can approximate complex functions, with performance evaluated using 3D visualizations and R² metrics.


This project demonstrates the use of Variational Quantum Circuits (VQCs) to perform a multidimensional regression task. Specifically, we aim to approximate a two-dimensional target function using a quantum neural network. Quantum computing, and particularly variational circuits, provide a promising approach for tasks that benefit from high-dimensional feature spaces and entanglement in data processing.
Variational quantum circuits (VQCs) involve parameterized quantum gates, and their parameters are updated using classical optimization methods. These models, also known as quantum-classical hybrid models, combine the power of quantum computation with the versatility of classical machine learning techniques.
Quantum Circuits Overview
1. Quantum Embedding:
The input features are encoded into a quantum state using Angle Embedding. Specifically, each data point x is mapped to the angle of a rotation gate on qubits.
AngleEmbedding takes each element in the vector x and applies it as a rotation around the Z-axis of the Bloch sphere on the corresponding qubit. In this case, we use two qubits (labeled 0 and 1), meaning our input space is two-dimensional. Angle encoding helps us map classical data into the quantum Hilbert space.
2. Quantum Circuit:
The quantum circuit comprises several layers of parameterized gates, specifically Strongly Entangling Layers. These layers are used to construct the variational ansatz, which is the core of the quantum neural network.
Strongly Entangling Layers: These layers apply a series of single-qubit rotations (parameterized by params) followed by controlled two-qubit operations, entangling the qubits. Entanglement is essential in quantum computing, as it allows quantum systems to model correlations between features that are impossible for classical systems to represent efficiently.
The function W(params) applies these entangling layers using the parameters params. These parameters are trainable and will be updated during the optimization process.
3. Quantum Neural Network (QNode):
The quantum neural network is defined using Pennylane's QNode, which allows us to interface quantum circuits with classical machine learning libraries (in this case, JAX).
QNode: This decorator converts a quantum function into an executable quantum circuit, connected to classical interfaces (like JAX). It can return measurement outcomes or expectation values.
PauliZ Expectation: The circuit computes the expectation value of the Pauli-Z operator on each wire. This value represents the quantum "output" after the circuit has processed the input features and applied the variational parameters. In this case, we return the product of PauliZ on two qubits, which captures the correlation between the qubits' states.
4. Target Function:
The target function is a simple quadratic function: This function will be approximated by the quantum neural network, and its shape will guide the training process.
Optimization Process
1. Loss Function:
The goal is to minimize the mean squared error (MSE) between the quantum neural network's predictions and the target function values. The MSE quantifies how close the model's predictions are to the actual values of the target function.
mse: This function calculates the squared error between the quantum neural network's output and the target function.
loss_fn: This function computes the mean of the squared errors over all training samples.
2. Optimizer (Adam):
The project uses Adam, a variant of stochastic gradient descent, to update the parameters of the quantum circuit. Adam is well-suited for noisy gradients, which can occur in quantum circuits due to the nature of quantum measurement. Adam optimizes the circuit's parameters by using momentum-based updates, adjusting each parameter according to both the gradient of the loss function and its previous update history.
3. JAX and Just-in-Time (JIT) Compilation:
The optimization process is compiled using JAX's jit functionality. JIT compilation accelerates the training by optimizing the computational graph and eliminating unnecessary operations. This is crucial when performing parameter optimization in quantum circuits, as quantum simulations can be computationally expensive.
JIT Compilation: The jit decorator ensures that the update steps are compiled and executed efficiently. This results in faster training loops.
Parameter Updates: In each step, the gradients of the loss function are computed using jax.value_and_grad, and the Adam optimizer updates the parameters accordingly.
4. Training Loop:
The training loop iterates for max_steps, updating the parameters at each step by minimizing the loss function. The function returns the optimized parameters of the quantum neural network.
5. Evaluation:
After training, we evaluate the model by comparing the predictions from the quantum neural network to the actual target values. The performance is measured using the R² score, which is a standard metric for regression tasks:
A high R2R^2R2 score (close to 1) indicates that the model successfully captures the variability in the data and accurately approximates the target function.
Results and Visualization
The model's performance is visualized using 3D surface plots:
•	Target Function: A 3D plot of the actual target function.
•	Predicted Function: A 3D plot of the quantum neural network's output after training.
•	Accuracy: The R² score is displayed to indicate how well the quantum circuit approximated the target function.
Conclusion
This project demonstrates the application of Variational Quantum Circuits (VQCs) for regression tasks. Using quantum entanglement and parameterized circuits, we can model complex relationships between variables. The hybrid quantum-classical approach, optimized with Adam and JAX's JIT compilation, enables efficient training of the model on classical hardware.
While this project uses a simple quadratic function for demonstration purposes, the techniques applied here can be extended to more complex data and higher-dimensional feature spaces. Future work can explore the impact of noise on quantum devices, alternative circuit architectures, and larger datasets.
Future Work
•	Implementing the model on a real quantum computer to test its robustness against noise.
•	Exploring different types of quantum embeddings (e.g., amplitude embedding or QAOA).
•	Investigating the scalability of this approach to higher-dimensional regression tasks and other machine learning problems.
