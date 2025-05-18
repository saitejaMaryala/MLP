# Multi-Layer Perceptron Implementation

A PyTorch implementation of a Multi-Layer Perceptron (MLP) with support for different optimization methods and activation functions.

## Features

- Multiple optimization methods:
  - Batch Gradient Descent
  - Mini-batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
- Activation functions:
  - ReLU
  - Sigmoid
  - Tanh
  - Linear
- Cross-entropy loss function
- GPU support for faster training
- K-fold cross-validation (10 folds)

## Architecture

The MLP implementation includes:
- Input layer: 1024 nodes (32x32 flattened grayscale images)
- Hidden layers: [1024, 512] nodes
- Output layer: 369 nodes (number of unique symbols)
- Learning rate: 0.05

## Usage

```python
# Initialize the MLP
mlp = MLP(
    input_size=1024,
    output_size=369,
    hidden_sizes=[1024, 512],
    learning_rate=0.05,
    activation_function='relu'
)

# Train using different optimizers
# Batch Gradient Descent
losses, accuracies = mlp.batch_fit(X_train, y_train, epochs=10)

# Mini-batch Gradient Descent
losses, accuracies = mlp.Mini_batch_fit(X_train, y_train, epochs=10, batch_size=50)

# Stochastic Gradient Descent
losses, accuracies = mlp.SGD_fit(X_train, y_train, epochs=3)
```

## Model Evaluation

The model performance is evaluated using:
- Cross-entropy loss
- Classification accuracy
- Mean and standard deviation across folds
- Training loss and accuracy curves

## Performance Analysis

The implementation includes analysis of:
- Model consistency across folds
- Impact of different activation functions
- Comparison of optimization methods
- Generalization capabilities through standard deviation analysis

## Dependencies

- PyTorch
- NumPy
- Pandas
- OpenCV (cv2)
- Matplotlib