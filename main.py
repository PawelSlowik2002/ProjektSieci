import pandas as pd
from network import NeuralNetwork
import numpy as np

# 1. Wczytanie danych
df = pd.read_csv('WineQT.csv')
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# 2. Przygotowanie celów
# Klasyfikacja: jakość wina w 3 klasach
df['quality_class'] = pd.cut(df['quality'], bins=[0, 4, 6, 10], labels=[0, 1, 2], include_lowest=True)

# 3. Rozdzielenie danych na Cechy (X) i Cele (y)
# X to nasze dane wejściowe - odrzucamy kolumny wynikowe
X = df.drop(['quality', 'quality_class'], axis=1)

# Cel klasyfikacyjny
y_class = df['quality_class'].values

# Cel regresyjny - spróbujmy przewidzieć zawartość alkoholu
y_reg = df['alcohol'].values
# Dla regresji musimy usunąć alkohol z cech wejściowych, żeby sieć nie znała odpowiedzi!
X_reg = X.drop('alcohol', axis=1)

# 4. NORMALIZACJA (wzór: (x - min) / (max - min))
X_norm = (X - X.min()) / (X.max() - X.min())
X_reg_norm = (X_reg - X_reg.min()) / (X_reg.max() - X_reg.min())

# Zamiana na czyste macierze numpy
X_class_np = X_norm.values
X_reg_np = X_reg_norm.values

# 5. PODZIAŁ NA ZBIÓR UCZĄCY (80%) I TESTOWY (20%)
np.random.seed(42) # Zapewnia powtarzalność losowania
indices = np.random.permutation(len(X))
test_size = int(len(X) * 0.2) # 20% to nasze dane testowe

test_idx, train_idx = indices[:test_size], indices[test_size:]

# Podział dla klasyfikacji
X_train_c, X_test_c = X_class_np[train_idx], X_class_np[test_idx]
y_train_c, y_test_c = y_class[train_idx], y_class[test_idx]

# Podział dla regresji
X_train_r, X_test_r = X_reg_np[train_idx], X_reg_np[test_idx]
y_train_r, y_test_r = y_reg[train_idx], y_reg[test_idx]

print("\n--- Etap Przygotowania Danych Zakończony ---")
print(f"Dane uczące (klasyfikacja): {X_train_c.shape[0]} wierszy, {X_train_c.shape[1]} cech")
print(f"Dane testowe (klasyfikacja): {X_test_c.shape[0]} wierszy, {X_test_c.shape[1]} cech")
print(f"Dane uczące (regresja): {X_train_r.shape[0]} wierszy, {X_train_r.shape[1]} cech")

print("\n--- ROZPOCZYNAMY TESTY PARAMETRÓW (SSN - Regresja) ---")

input_size = X_train_r.shape[1]
output_size = 1
powtorzenia = 3


def calculate_mse(y_true, y_pred):
    return np.mean((y_true.reshape(-1, 1) - y_pred) ** 2)


# --- PARAMETR 1: Liczba neuronów w warstwie ukrytej ---
print("\nParametr 1: Liczba neuronów w warstwie ukrytej (LR=0.01, Epoki=500)")
for hidden in [8, 16, 32, 64]:
    err_tr, err_te = [], []
    for _ in range(powtorzenia):
        nn = NeuralNetwork(input_size, hidden, output_size, learning_rate=0.01)
        nn.train(X_train_r, y_train_r, epochs=500)
        err_tr.append(calculate_mse(y_train_r, nn.forward(X_train_r)))
        err_te.append(calculate_mse(y_test_r, nn.forward(X_test_r)))
    print(f"Neurony: {hidden:2d} | Błąd (Train): {np.mean(err_tr):.4f} | Błąd (Test): {np.mean(err_te):.4f}")

# --- PARAMETR 2: Współczynnik uczenia (Learning Rate) ---
print("\nParametr 2: Współczynnik uczenia (Neurony=16, Epoki=500)")
for lr in [0.1, 0.05, 0.01, 0.001]:
    err_tr, err_te = [], []
    for _ in range(powtorzenia):
        nn = NeuralNetwork(input_size, 16, output_size, learning_rate=lr)
        nn.train(X_train_r, y_train_r, epochs=500)
        err_tr.append(calculate_mse(y_train_r, nn.forward(X_train_r)))
        err_te.append(calculate_mse(y_test_r, nn.forward(X_test_r)))
    print(f"LR: {lr:.3f} | Błąd (Train): {np.mean(err_tr):.4f} | Błąd (Test): {np.mean(err_te):.4f}")

# --- PARAMETR 3: Liczba epok (długość treningu) ---
print("\nParametr 3: Liczba epok (Neurony=16, LR=0.01)")
for eps in [100, 500, 1000, 2000]:
    err_tr, err_te = [], []
    for _ in range(powtorzenia):
        nn = NeuralNetwork(input_size, 16, output_size, learning_rate=0.01)
        nn.train(X_train_r, y_train_r, epochs=eps)
        err_tr.append(calculate_mse(y_train_r, nn.forward(X_train_r)))
        err_te.append(calculate_mse(y_test_r, nn.forward(X_test_r)))
    print(f"Epoki: {eps:4d} | Błąd (Train): {np.mean(err_tr):.4f} | Błąd (Test): {np.mean(err_te):.4f}")

# --- PARAMETR 4: Wielkość próby uczącej ---
print("\nParametr 4: Proporcja podziału zbioru (Train/Test)")
for split_ratio in [0.5, 0.6, 0.7, 0.8]:
    err_tr, err_te = [], []
    test_size_local = int(len(X_reg_np) * (1 - split_ratio))
    X_tr, X_te = X_reg_np[test_size_local:], X_reg_np[:test_size_local]
    y_tr, y_te = y_reg[test_size_local:], y_reg[:test_size_local]

    for _ in range(powtorzenia):
        nn = NeuralNetwork(input_size, 16, output_size, learning_rate=0.01)
        nn.train(X_tr, y_tr, epochs=500)
        err_tr.append(calculate_mse(y_tr, nn.forward(X_tr)))
        err_te.append(calculate_mse(y_te, nn.forward(X_te)))
    print(
        f"Train {int(split_ratio * 100)}% / Test {int((1 - split_ratio) * 100)}% | Błąd (Train): {np.mean(err_tr):.4f} | Błąd (Test): {np.mean(err_te):.4f}")

# --- PARAMETR 5: Sposób doboru próby (Losowy vs Sekwencyjny) ---
print("\nParametr 5: Sposób doboru próby (Losowy vs Sekwencyjny)")
metody = ["Losowy (Złoty Standard)", "Sekwencyjny (Brak przemieszania)"]
for metoda in metody:
    err_tr, err_te = [], []
    idx = np.random.permutation(len(X_reg_np)) if "Losowy" in metoda else np.arange(len(X_reg_np))
    t_size = int(len(X_reg_np) * 0.2)
    X_tr, X_te = X_reg_np[idx[t_size:]], X_reg_np[idx[:t_size]]
    y_tr, y_te = y_reg[idx[t_size:]], y_reg[idx[:t_size]]

    for _ in range(powtorzenia):
        nn = NeuralNetwork(input_size, 16, output_size, learning_rate=0.01)
        nn.train(X_tr, y_tr, epochs=500)
        err_tr.append(calculate_mse(y_tr, nn.forward(X_tr)))
        err_te.append(calculate_mse(y_te, nn.forward(X_te)))
    print(f"Dobór: {metoda[:10]:10s} | Błąd (Train): {np.mean(err_tr):.4f} | Błąd (Test): {np.mean(err_te):.4f}")

# --- PARAMETR 6: Wpływ normalizacji danych ---
print("\nParametr 6: Wpływ normalizacji danych wejściowych")
dane_wersje = {"Znormalizowane (0-1)": X_reg_np, "Brak normalizacji (Surowe)": X_reg.values}
for nazwa, X_data in dane_wersje.items():
    err_tr, err_te = [], []
    idx = np.random.permutation(len(X_data))
    t_size = int(len(X_data) * 0.2)
    X_tr, X_te = X_data[idx[t_size:]], X_data[idx[:t_size]]
    y_tr, y_te = y_reg[idx[t_size:]], y_reg[idx[:t_size]]

    for _ in range(powtorzenia):
        nn = NeuralNetwork(input_size, 16, output_size, learning_rate=0.01)
        nn.train(X_tr, y_tr, epochs=500)
        err_tr.append(calculate_mse(y_tr, nn.forward(X_tr)))
        err_te.append(calculate_mse(y_te, nn.forward(X_te)))
    print(f"Dane: {nazwa:22s} | Błąd (Train): {np.mean(err_tr):.4f} | Błąd (Test): {np.mean(err_te):.4f}")

    from network_class import NeuralNetworkClassifier

    print("\n\n=======================================================")
    print("--- ROZPOCZYNAMY TESTY PARAMETRÓW (SSN - Klasyfikacja) ---")
    print("=======================================================\n")


    # Funkcja pomocnicza: kodowanie One-Hot (zamienia klasę 1 na [0, 1, 0] itd.)
    def to_one_hot(y, num_classes=3):
        return np.eye(num_classes)[y.astype(int)]


    input_size_c = X_train_c.shape[1]
    output_size_c = 3  # Trzy klasy: 0, 1, 2
    powtorzenia = 3

    y_train_c_oh = to_one_hot(y_train_c)
    y_test_c_oh = to_one_hot(y_test_c)


    # Zamiast błędu mierzymy Skuteczność (ile % poprawnie zgadniętych)
    def calculate_accuracy(y_true, y_pred):
        return np.mean(y_true.astype(int) == y_pred) * 100


    # --- PARAMETR 1: Liczba neuronów ---
    print("Parametr 1: Liczba neuronów w warstwie ukrytej (LR=0.1, Epoki=500)")
    for hidden in [8, 16, 32, 64]:
        acc_tr, acc_te = [], []
        for _ in range(powtorzenia):
            nn_c = NeuralNetworkClassifier(input_size_c, hidden, output_size_c, learning_rate=0.1)
            nn_c.train(X_train_c, y_train_c_oh, epochs=500)
            acc_tr.append(calculate_accuracy(y_train_c, nn_c.predict(X_train_c)))
            acc_te.append(calculate_accuracy(y_test_c, nn_c.predict(X_test_c)))
        print(
            f"Neurony: {hidden:2d} | Skuteczność Train: {np.mean(acc_tr):.2f}% | Skuteczność Test: {np.mean(acc_te):.2f}%")

    # --- PARAMETR 2: Współczynnik uczenia ---
    print("\nParametr 2: Współczynnik uczenia (Neurony=16, Epoki=500)")
    for lr in [0.5, 0.1, 0.05, 0.01]:
        acc_tr, acc_te = [], []
        for _ in range(powtorzenia):
            nn_c = NeuralNetworkClassifier(input_size_c, 16, output_size_c, learning_rate=lr)
            nn_c.train(X_train_c, y_train_c_oh, epochs=500)
            acc_tr.append(calculate_accuracy(y_train_c, nn_c.predict(X_train_c)))
            acc_te.append(calculate_accuracy(y_test_c, nn_c.predict(X_test_c)))
        print(f"LR: {lr:.3f} | Skuteczność Train: {np.mean(acc_tr):.2f}% | Skuteczność Test: {np.mean(acc_te):.2f}%")

    # --- PARAMETR 3: Liczba epok ---
    print("\nParametr 3: Liczba epok (Neurony=16, LR=0.1)")
    for eps in [100, 500, 1000, 2000]:
        acc_tr, acc_te = [], []
        for _ in range(powtorzenia):
            nn_c = NeuralNetworkClassifier(input_size_c, 16, output_size_c, learning_rate=0.1)
            nn_c.train(X_train_c, y_train_c_oh, epochs=eps)
            acc_tr.append(calculate_accuracy(y_train_c, nn_c.predict(X_train_c)))
            acc_te.append(calculate_accuracy(y_test_c, nn_c.predict(X_test_c)))
        print(f"Epoki: {eps:4d} | Skuteczność Train: {np.mean(acc_tr):.2f}% | Skuteczność Test: {np.mean(acc_te):.2f}%")

    # --- PARAMETR 4: Proporcja podziału zbioru ---
    print("\nParametr 4: Proporcja podziału zbioru (Train/Test)")
    for split_ratio in [0.5, 0.6, 0.7, 0.8]:
        acc_tr, acc_te = [], []
        test_size_local = int(len(X_class_np) * (1 - split_ratio))
        X_tr, X_te = X_class_np[test_size_local:], X_class_np[:test_size_local]
        y_tr, y_te = y_class[test_size_local:], y_class[:test_size_local]
        y_tr_oh = to_one_hot(y_tr)

        for _ in range(powtorzenia):
            nn_c = NeuralNetworkClassifier(input_size_c, 16, output_size_c, learning_rate=0.1)
            nn_c.train(X_tr, y_tr_oh, epochs=500)
            acc_tr.append(calculate_accuracy(y_tr, nn_c.predict(X_tr)))
            acc_te.append(calculate_accuracy(y_te, nn_c.predict(X_te)))
        print(
            f"Train {int(split_ratio * 100)}% / Test {int((1 - split_ratio) * 100)}% | Skuteczność Train: {np.mean(acc_tr):.2f}% | Skuteczność Test: {np.mean(acc_te):.2f}%")

    # --- PARAMETR 5: Sposób doboru próby ---
    print("\nParametr 5: Sposób doboru próby (Losowy vs Sekwencyjny)")
    for metoda in ["Losowy", "Sekwencyjny"]:
        acc_tr, acc_te = [], []
        idx = np.random.permutation(len(X_class_np)) if metoda == "Losowy" else np.arange(len(X_class_np))
        t_size = int(len(X_class_np) * 0.2)
        X_tr, X_te = X_class_np[idx[t_size:]], X_class_np[idx[:t_size]]
        y_tr, y_te = y_class[idx[t_size:]], y_class[idx[:t_size]]
        y_tr_oh = to_one_hot(y_tr)

        for _ in range(powtorzenia):
            nn_c = NeuralNetworkClassifier(input_size_c, 16, output_size_c, learning_rate=0.1)
            nn_c.train(X_tr, y_tr_oh, epochs=500)
            acc_tr.append(calculate_accuracy(y_tr, nn_c.predict(X_tr)))
            acc_te.append(calculate_accuracy(y_te, nn_c.predict(X_te)))
        print(
            f"Dobór: {metoda:10s} | Skuteczność Train: {np.mean(acc_tr):.2f}% | Skuteczność Test: {np.mean(acc_te):.2f}%")

    # --- PARAMETR 6: Funkcja Aktywacji (Sigmoid vs Tanh) ---
    # Tego parametru jeszcze nie było, to świetny dodatek do klasyfikacji!
    print("\nParametr 6: Metoda uczenia (Wpływ ilości epok przy małym LR=0.01)")
    for eps in [500, 1000, 2000, 4000]:
        acc_tr, acc_te = [], []
        for _ in range(powtorzenia):
            nn_c = NeuralNetworkClassifier(input_size_c, 32, output_size_c, learning_rate=0.01)
            nn_c.train(X_train_c, y_train_c_oh, epochs=eps)
            acc_tr.append(calculate_accuracy(y_train_c, nn_c.predict(X_train_c)))
            acc_te.append(calculate_accuracy(y_test_c, nn_c.predict(X_test_c)))
        print(
            f"Epoki: {eps:4d} (przy LR=0.01) | Skuteczność Train: {np.mean(acc_tr):.2f}% | Test: {np.mean(acc_te):.2f}%")