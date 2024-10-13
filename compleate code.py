import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import jax
from jax import numpy as jnp
import optax

# Set seed for reproducibility, ensuring that the random values generated are the same every time.
pnp.random.seed(42)

# Initialize a quantum device with two qubits using the 'default.qubit' simulator backend provided by Pennylane.
# The device is where quantum computations are simulated.
try:
    dev = qml.device('default.qubit', wires=2)
except Exception as e:
    print(f"Error initializing the quantum device: {e}")

# Function to encode classical data into quantum states.
# The 'AngleEmbedding' maps input data to rotation angles on the qubits (Z-axis rotations), effectively encoding the data into the quantum circuit.
def S(x):
    try:
        qml.AngleEmbedding(x, wires=[0, 1], rotation='Z')
    except Exception as e:
        print(f"Error in AngleEmbedding: {e}")

# Function to create parameterized quantum layers.
# 'StronglyEntanglingLayers' applies parameterized single-qubit rotations followed by entangling gates between qubits, making the qubits interact.
# These layers are used to introduce non-linear transformations in the quantum circuit.
def W(params):
    try:
        qml.StronglyEntanglingLayers(params, wires=[0, 1])
    except Exception as e:
        print(f"Error in StronglyEntanglingLayers: {e}")

# Define a quantum neural network, also called a QNode, which interfaces quantum circuits with classical machine learning (JAX in this case).
# This function encodes the input data into quantum states, applies several layers of quantum gates, and returns the expectation value of a PauliZ measurement.
@qml.qnode(dev, interface="jax")
def quantum_neural_network(params, x):
    try:
        # Extract the number of layers, qubits, and parameters in each rotation gate.
        layers = len(params[:, 0, 0]) - 1  # Number of layers in the circuit
        n_wires = len(params[0, :, 0])  # Number of qubits
        n_params_rot = len(params[0, 0, :])  # Parameters for rotations
        
        # Apply quantum layers and the data embedding in alternating fashion.
        for i in range(layers):
            W(params[i, :, :].reshape(1, n_wires, n_params_rot))  # Apply the parameterized quantum layer
            S(x)  # Embed the classical data into the quantum system
        
        # Apply the final quantum layer without further embedding.
        W(params[-1, :, :].reshape(1, n_wires, n_params_rot))
        
        # Return the expectation value of PauliZ measurements on the qubits, capturing the output of the quantum circuit.
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))  
    except Exception as e:
        print(f"Error in quantum_neural_network: {e}")

# Define the target function to be approximated by the quantum neural network.
# This is a simple quadratic function that acts as the ground truth for the regression task.
def target_function(x):
    try:
        f = 1/2 * (x[0]**2 + x[1]**2)
        return f
    except Exception as e:
        print(f"Error in target_function: {e}")
        return None

# Data generation: Create a meshgrid of points from x1 and x2 ranges, and stack them into input data.
# The goal is to create training data for the quantum model.
x1_min, x1_max = -1, 1
x2_min, x2_max = -1, 1
num_samples = 30

try:
    x1_train = pnp.linspace(x1_min, x1_max, num_samples)  # Generate evenly spaced points in the x1 range
    x2_train = pnp.linspace(x2_min, x2_max, num_samples)  # Generate evenly spaced points in the x2 range
    x1_mesh, x2_mesh = pnp.meshgrid(x1_train, x2_train)   # Create a grid of x1, x2 pairs

    # Stack x1 and x2 grids to form a 2D input feature set and compute corresponding target values
    x_train = pnp.stack((x1_mesh.flatten(), x2_mesh.flatten()), axis=1)
    y_train = target_function([x1_mesh, x2_mesh]).reshape(-1, 1)  # Compute the target values for each input pair

    # Print the first few training samples and targets to check the data
    print("x_train:\n", x_train[:5])
    print("y_train:\n", y_train[:5])
except Exception as e:
    print(f"Error generating training data: {e}")

# Define the mean squared error (MSE) function to measure the difference between predicted and actual target values.
# The quantum circuit's output is compared to the target function to calculate the error.
@jax.jit
def mse(params, x, targets):
    try:
        return (quantum_neural_network(params, x) - jnp.array(targets))**2
    except Exception as e:
        print(f"Error in mse calculation: {e}")

# Define the loss function, which averages the MSE over all samples.
# This function is what the optimizer seeks to minimize by updating the quantum circuit parameters.
@jax.jit
def loss_fn(params, x, targets):
    try:
        # Compute the MSE for each input-target pair and return the average loss.
        mse_pred = jax.vmap(mse, in_axes=(None, 0, 0))(params, x, targets)
        loss = jnp.mean(mse_pred)
        return loss
    except Exception as e:
        print(f"Error in loss function: {e}")

# Initialize the Adam optimizer, which updates the parameters based on the gradients of the loss function.
opt = optax.adam(learning_rate=0.05)
max_steps = 300  # Set the number of optimization steps

# Define the update function for each training step.
# This function calculates the gradients and applies the Adam optimizer to update the quantum circuit parameters.
@jax.jit
def update_step_jit(i, args):
    try:
        params, opt_state, data, targets, print_training = args
        loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)  # Compute gradients of loss
        updates, opt_state = opt.update(grads, opt_state)  # Update parameters using Adam optimizer
        params = optax.apply_updates(params, updates)  # Apply parameter updates

        # Conditionally print the training progress every 50 steps
        def print_fn():
            jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
        
        # Only print training progress if 'print_training' is True and i is a multiple of 50
        jax.lax.cond((jnp.mod(i, 50) == 0) & print_training, print_fn, lambda: None)
        return (params, opt_state, data, targets, print_training)
    except Exception as e:
        print(f"Error in update_step_jit: {e}")

# Main optimization loop that runs for 'max_steps' iterations to train the quantum neural network.
# Parameters are updated to minimize the loss function.
@jax.jit
def optimization_jit(params, data, targets, print_training=False):
    try:
        opt_state = opt.init(params)  # Initialize the optimizer state
        args = (params, opt_state, jnp.asarray(data), targets, print_training)
        (params, opt_state, _, _, _) = jax.lax.fori_loop(0, max_steps + 1, update_step_jit, args)  # Optimization loop
        return params
    except Exception as e:
        print(f"Error in optimization process: {e}")

# Initialize the quantum circuit with random parameters.
# These are the variables that will be tuned during the optimization.
wires = 2
layers = 4
params_shape = qml.StronglyEntanglingLayers.shape(n_layers=layers + 1, n_wires=wires)

try:
    params = pnp.random.default_rng().random(size=params_shape)  # Generate random initial parameters
    best_params = optimization_jit(params, x_train, jnp.array(y_train), print_training=True)  # Train the model
except Exception as e:
    print(f"Error initializing or optimizing parameters: {e}")

# Evaluate the quantum neural network on the training data after the parameters have been optimized.
def evaluate(params, data):
    try:
        y_pred = jax.vmap(quantum_neural_network, in_axes=(None, 0))(params, data)  # Make predictions for each data point
        return y_pred
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

# After training, predict values using the trained quantum neural network and evaluate its performance.
try:
    y_predictions = evaluate(best_params, x_train)  # Predict using optimized parameters

    # Calculate the R² score, which measures how well the predictions fit the true target values.
    from sklearn.metrics import r2_score
    r2 = round(float(r2_score(y_train, y_predictions)), 3)
    print("R² Score:", r2)
except Exception as e:
    print(f"Error calculating R² score or predictions: {e}")

# Visualization: Plotting the target function and predictions side-by-side
try:
    fig = plt.figure()

    # Target function
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(x1_mesh, x2_mesh, y_train.reshape(x1_mesh.shape), cmap='viridis')
    ax1.set_zlim(0, 1)
    ax1.set_xlabel('$x$', fontsize=10)
    ax1.set_ylabel('$y$', fontsize=10)
    ax1.set_zlabel('$f(x,y)$', fontsize=10)
    ax1.set_title('Target')

    # Predictions
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(x1_mesh, x2_mesh, y_predictions.reshape(x1_mesh.shape), cmap='viridis')
    ax2.set_zlim(0, 1)
    ax2.set_xlabel('$x$', fontsize=10)
    ax2.set_ylabel('$y$', fontsize=10)
    ax2.set_zlabel('$f(x,y)$', fontsize=10)
    ax2.set_title(f'Predicted \\nAccuracy: {round(r2*100, 3)}%')

    plt.tight_layout(pad=3.7)
    plt.show()
except Exception as e:
    print(f"Error during visualization: {e}")
