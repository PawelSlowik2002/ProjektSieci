import numpy as np


class NeuralNetworkClassifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    # Nowa funkcja dla klasyfikacji: zamienia wyniki na prawdopodobieństwa (od 0 do 1)
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Używamy softmax na wyjściu
        self.output = self.softmax(self.z2)
        return self.output

    def backward(self, X, y, output):
        m = X.shape[0]
        # Dla softmax i cross-entropy gradient to po prostu: przewidywanie - prawda
        d_output = output - y

        dW2 = (1 / m) * np.dot(self.a1.T, d_output)
        db2 = (1 / m) * np.sum(d_output, axis=0, keepdims=True)

        d_a1 = np.dot(d_output, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)

        dW1 = (1 / m) * np.dot(X.T, d_z1)
        db1 = (1 / m) * np.sum(d_z1, axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    # Zwraca numer klasy (0, 1 lub 2) z najwyższym prawdopodobieństwem
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)