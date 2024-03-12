# fpu/examples/QML_Integration_Example.py

import torch
from fpu.src.ai.Quantum_Machine_Learning import QuantumNeuralNetwork
from torch.optim import Adam
from torch.nn import MSELoss

def train_quantum_neural_network(X_train, y_train, epochs=100, learning_rate=0.01):
    """
    Trains a Quantum Neural Network on the provided training data.
    
    :param X_train: Input features for training.
    :param y_train: Target outputs for training.
    :param epochs: Number of epochs to train for.
    :param learning_rate: Learning rate for the optimizer.
    """
    # Initialize the Quantum Neural Network
    n_features = X_train.shape[1]
    n_qubits = n_features  # Assuming a qubit for each feature for simplicity
    q_depth = 3  # Depth of the quantum circuit
    model = QuantumNeuralNetwork(n_features, n_qubits, q_depth)
    
    # Set up the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = MSELoss()

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_func(output, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

    return model

def generate_synthetic_data(n_samples=100, n_features=4):
    """
    Generates synthetic data for demonstration purposes.
    
    :param n_samples: Number of samples to generate.
    :param n_features: Number of features per sample.
    :return: Tuple of input features and target outputs.
    """
    X = torch.randn(n_samples, n_features)
    y = torch.sum(X, dim=1, keepdim=True)  # Simple target for demonstration
    return X, y

if __name__ == "__main__":
    # Generate synthetic data
    X_train, y_train = generate_synthetic_data(n_samples=100, n_features=4)

    # Train the Quantum Neural Network
    model = train_quantum_neural_network(X_train, y_train, epochs=100, learning_rate=0.01)

    # Simple demonstration of model prediction
    test_input = torch.randn(1, 4)
    predicted_output = model(test_input)
    print(f"Test input: {test_input}")
    print(f"Predicted output: {predicted_output}")
