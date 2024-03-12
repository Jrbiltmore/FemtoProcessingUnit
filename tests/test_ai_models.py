# fpu/tests/test_ai_models.py

import unittest
import torch
from fpu.src.ai.AI_Models import BasicCNN, LSTMModel, QuantumNeuralNetwork

class TestBasicCNN(unittest.TestCase):
    def test_forward_pass(self):
        model = BasicCNN(num_classes=10)
        input_tensor = torch.randn(1, 1, 28, 28)  # Simulate a batch of one MNIST image
        output = model(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, 10]))

class TestLSTMModel(unittest.TestCase):
    def test_forward_pass(self):
        model = LSTMModel(input_dim=28, hidden_dim=128, layer_dim=2, output_dim=10)
        input_tensor = torch.randn(1, 28, 28)  # Simulate a batch of one sequence with 28 timesteps each having 28 features
        output = model(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, 10]))

class TestQuantumNeuralNetwork(unittest.TestCase):
    def test_forward_pass(self):
        n_features = 4
        n_qubits = 4
        q_depth = 3
        model = QuantumNeuralNetwork(n_features, n_qubits, q_depth)
        input_tensor = torch.randn(1, n_features)  # Simulate a batch of one input with n_features
        output = model(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, 1]))

if __name__ == '__main__':
    unittest.main()
