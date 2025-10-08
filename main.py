import numpy as np

# ===============================
# Artificial Neural Network from Scratch
# ===============================

# Load and preprocess MNIST data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

x_train = np.reshape(x_train, (60000, 784)).T / 255
x_test = np.reshape(x_test, (10000, 784)).T / 255

# ===============================
# Hyperparameters
# ===============================
BATCH_SIZE = 75
LEARNING_RATE = 0.01
EPOCHS = 10


# ===============================
# Model Initialization
# ===============================
def initialize_parameters():
    """Initialize weights and biases for a 2-layer neural network."""
    W1 = np.random.rand(10, 784) * 0.01
    B1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10) * 0.01
    B2 = np.random.rand(10, 1)
    return W1, B1, W2, B2


# ===============================
# Activation Functions
# ===============================
def softmax(A):
    return np.exp(A) / np.sum(np.exp(A), axis=0, keepdims=True)


def relu_derivative(A):
    return A > 0


# ===============================
# Forward and Backward Propagation
# ===============================
def forward_propagation(X, W1, B1, W2, B2):
    G1 = np.dot(W1, X) + B1
    Y1 = np.maximum(G1, 0)
    G2 = np.dot(W2, Y1) + B2
    Y2 = softmax(G2)
    return G1, Y1, G2, Y2


def one_hot_encode(labels):
    """Convert labels to one-hot encoded matrix."""
    return np.eye(10)[labels].T


def backward_propagation(Y2, Y1, W2, G1, X, T):
    """Compute gradients for all parameters."""
    m = T.shape[1]
    Z2 = Y2 - T
    dW2 = (1 / m) * np.dot(Z2, Y1.T)
    dB2 = (1 / m) * np.sum(Z2, axis=1, keepdims=True)
    Z1 = np.dot(W2.T, Z2) * relu_derivative(G1)
    dW1 = (1 / m) * np.dot(Z1, X.T)
    dB1 = (1 / m) * np.sum(Z1, axis=1, keepdims=True)
    return dW1, dB1, dW2, dB2


# ===============================
# Evaluation
# ===============================
def accuracy(X, W1, B1, W2, B2, labels):
    G1 = np.dot(W1, X) + B1
    Y1 = np.maximum(G1, 0)
    G2 = np.dot(W2, Y1) + B2
    Y2 = softmax(G2)
    predictions = np.argmax(Y2, axis=0)
    acc = np.mean(predictions == labels) * 100
    return acc


# ===============================
# Training Loop
# ===============================
def train(x_train, y_train, x_test, y_test):
    W1, B1, W2, B2 = initialize_parameters()
    m = y_train.shape[0]

    for epoch in range(EPOCHS):
        indices = np.arange(m)
        np.random.shuffle(indices)
        x_train = x_train[:, indices]
        y_train = y_train[indices]

        for j in range(int(m / BATCH_SIZE)):
            start, end = j * BATCH_SIZE, (j + 1) * BATCH_SIZE
            X = x_train[:, start:end]
            T = one_hot_encode(y_train[start:end])

            G1, Y1, G2, Y2 = forward_propagation(X, W1, B1, W2, B2)
            dW1, dB1, dW2, dB2 = backward_propagation(Y2, Y1, W2, G1, X, T)

            W1 -= LEARNING_RATE * dW1
            B1 -= LEARNING_RATE * dB1
            W2 -= LEARNING_RATE * dW2
            B2 -= LEARNING_RATE * dB2

        acc = accuracy(x_test, W1, B1, W2, B2, y_test)
        print(f"After {epoch+1}/{EPOCHS} epochs → Accuracy: {acc:.2f}%")

    # Uncomment to save model if accuracy is high
    # np.save("W1.npy", W1)
    # np.save("B1.npy", B1)
    # np.save("W2.npy", W2)
    # np.save("B2.npy", B2)


train(x_train, y_train, x_test, y_test)
