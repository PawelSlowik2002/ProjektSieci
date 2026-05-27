import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Inicjalizacja wag z losowymi wartościami i biasów (odchyleń) jako zera
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    # Funkcja aktywacji Sigmoid i jej pochodna
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # clip chroni przed błędem przepełnienia

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        # Przejście sygnału w przód (Forward Propagation)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Dla regresji na wyjściu nie dajemy sigmoida (chcemy konkretną liczbę, np. zawartość alkoholu)
        self.output = self.z2
        return self.output

    def backward(self, X, y, output):
        # Propagacja błędu wstecz (Backpropagation) - serce uczenia się sieci
        m = X.shape[0]  # liczba przykładów

        # Błąd na wyjściu (Różnica między tym co sieć zgadła, a prawdziwym wynikiem y)
        d_output = output - y.reshape(-1, 1)

        # Obliczanie gradientów dla drugiej warstwy
        dW2 = (1 / m) * np.dot(self.a1.T, d_output)
        db2 = (1 / m) * np.sum(d_output, axis=0, keepdims=True)

        # Przekazanie błędu do warstwy ukrytej
        d_a1 = np.dot(d_output, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)

        # Obliczanie gradientów dla pierwszej warstwy
        dW1 = (1 / m) * np.dot(X.T, d_z1)
        db1 = (1 / m) * np.sum(d_z1, axis=0, keepdims=True)

        # Aktualizacja wag (sieć się uczy)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000):
        # Pętla treningowa
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)