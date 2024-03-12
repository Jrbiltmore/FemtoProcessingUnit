# fpu/src/ai/Machine_Learning_Processors.py

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from .AI_Models import BasicCNN, LSTMModel

class MachineLearningProcessor:
    def __init__(self, model_type='cnn', num_classes=10, learning_rate=0.001, batch_size=64, epochs=10):
        self.model_type = model_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        if model_type == 'cnn':
            self.model = BasicCNN(num_classes=num_classes)
        elif model_type == 'lstm':
            self.model = LSTMModel(input_dim=28, hidden_dim=128, layer_dim=2, output_dim=num_classes)
        else:
            raise ValueError("Unsupported model type. Choose either 'cnn' or 'lstm'.")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, X_train, y_train):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for data, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(X_test, dtype=torch.float32))
            _, predicted = torch.max(outputs.data, 1)
            accuracy = accuracy_score(y_test, predicted.numpy())
        print(f'Accuracy: {accuracy * 100}%')
        return accuracy

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(X, dtype=torch.float32))
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def save_model(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# Example usage
if __name__ == "__main__":
    # Generate or load your dataset here
    X, y = np.random.rand(1000, 1, 28, 28), np.random.randint(0, 10, 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    processor = MachineLearningProcessor(model_type='cnn')
    processor.train(X_train, y_train)
    accuracy = processor.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100}%")
# fpu/src/ai/Machine_Learning_Processors.py (continued)

from sklearn.metrics import classification_report
import json

class HyperparameterTuner:
    """
    A simple hyperparameter tuning class that could be expanded to use
    grid search, random search, or more advanced methods like Bayesian optimization.
    """
    def __init__(self, parameter_space, model_type='cnn', num_classes=10):
        self.parameter_space = parameter_space
        self.model_type = model_type
        self.num_classes = num_classes

    def grid_search(self, X_train, y_train, X_val, y_val):
        best_accuracy = 0
        best_params = {}
        for params in self.parameter_space:
            processor = MachineLearningProcessor(model_type=self.model_type, num_classes=self.num_classes, **params)
            processor.train(X_train, y_train)
            accuracy = processor.evaluate(X_val, y_val)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
        return best_params, best_accuracy

# Advanced data preprocessing placeholder
def advanced_data_preprocessing(X):
    """
    Implement advanced data preprocessing techniques here.
    This function is a placeholder and should be filled with actual preprocessing logic.
    """
    processed_X = X # Placeholder for actual preprocessing steps
    return processed_X

# Saving and loading model configurations
def save_model_config(model_config, path='model_config.json'):
    with open(path, 'w') as json_file:
        json.dump(model_config, json_file)

def load_model_config(path='model_config.json'):
    with open(path, 'r') as json_file:
        model_config = json.load(json_file)
    return model_config

# Extended usage example including hyperparameter tuning and model config saving
if __name__ == "__main__":
    # Assuming X, y have been defined or loaded as before
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simple parameter space for demonstration
    parameter_space = [
        {'learning_rate': 0.001, 'epochs': 10},
        {'learning_rate': 0.0001, 'epochs': 20},
    ]

    tuner = HyperparameterTuner(parameter_space, model_type='cnn', num_classes=10)
    best_params, best_accuracy = tuner.grid_search(X_train, y_train, X_test, y_test)
    print(f"Best Params: {best_params}, Best Accuracy: {best_accuracy}")

    # Training the best model
    processor = MachineLearningProcessor(model_type='cnn', num_classes=10, **best_params)
    processor.train(X_train, y_train)
    processor.evaluate(X_test, y_test)

    # Saving model and config
    processor.save_model(path='best_model.pth')
    save_model_config(best_params, path='best_model_config.json')
