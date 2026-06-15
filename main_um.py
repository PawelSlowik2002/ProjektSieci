import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

print("=======================================================")
print("  CZĘŚĆ 2: UCZENIE MASZYNOWE (Gotowe biblioteki)")
print("=======================================================\n")

df = pd.read_csv('WineQT.csv')
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

df['quality_class'] = pd.cut(df['quality'], bins=[0, 4, 6, 10], labels=[0, 1, 2], include_lowest=True)

X = df.drop(['quality', 'quality_class'], axis=1)
y_class = df['quality_class'].values
y_reg = df['alcohol'].values
X_reg = X.drop('alcohol', axis=1)

# Automatyczny podział danych (80% Train, 20% Test) i Normalizacja (StandardScaler)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)

scaler_c = StandardScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)


print("--- PROBLEM 1: REGRESJA (Wartość Alkoholu) ---\n")

print("METODA A: Las Losowy (Random Forest Regressor)")
print("Parametr 1: Liczba drzew (n_estimators)")
for n in [10, 50, 100, 200]:
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train_r, y_train_r)
    print(f"Drzewa: {n:3d} | MSE Train: {mean_squared_error(y_train_r, rf.predict(X_train_r)):.4f} | MSE Test: {mean_squared_error(y_test_r, rf.predict(X_test_r)):.4f}")

print("\nParametr 2: Maksymalna głębokość drzewa (max_depth)")
for d in [5, 10, 20, None]:  # None oznacza brak limitu
    rf = RandomForestRegressor(max_depth=d, random_state=42)
    rf.fit(X_train_r, y_train_r)
    d_str = str(d) if d is not None else "Brak"
    print(f"Głębokość: {d_str:>4s} | MSE Train: {mean_squared_error(y_train_r, rf.predict(X_train_r)):.4f} | MSE Test: {mean_squared_error(y_test_r, rf.predict(X_test_r)):.4f}")

print("\nParametr 3: Minimalna liczba próbek w liściu (min_samples_leaf)")
for m in [1, 2, 5, 10]:
    rf = RandomForestRegressor(min_samples_leaf=m, random_state=42)
    rf.fit(X_train_r, y_train_r)
    print(f"Próbki w liściu: {m:2d} | MSE Train: {mean_squared_error(y_train_r, rf.predict(X_train_r)):.4f} | MSE Test: {mean_squared_error(y_test_r, rf.predict(X_test_r)):.4f}")


print("\nMETODA B: K-Najbliższych Sąsiadów (KNN Regressor)")
print("Parametr 1: Liczba sąsiadów (n_neighbors)")
for k in [3, 5, 9, 15]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_r, y_train_r)
    print(f"Sąsiedzi: {k:2d} | MSE Train: {mean_squared_error(y_train_r, knn.predict(X_train_r)):.4f} | MSE Test: {mean_squared_error(y_test_r, knn.predict(X_test_r)):.4f}")

print("\nParametr 2: Metryka odległości (p)")
# p=1: Manhattan, p=2: Euklidesowa
for p_val in [1, 2, 3, 4]:
    knn = KNeighborsRegressor(p=p_val)
    knn.fit(X_train_r, y_train_r)
    print(f"Metryka p: {p_val:2d} | MSE Train: {mean_squared_error(y_train_r, knn.predict(X_train_r)):.4f} | MSE Test: {mean_squared_error(y_test_r, knn.predict(X_test_r)):.4f}")

print("\nParametr 3: Algorytm wyszukiwania sąsiadów")
for alg in ['ball_tree', 'kd_tree', 'brute', 'auto']:
    knn = KNeighborsRegressor(algorithm=alg)
    knn.fit(X_train_r, y_train_r)
    print(f"Algorytm: {alg:9s} | MSE Train: {mean_squared_error(y_train_r, knn.predict(X_train_r)):.4f} | MSE Test: {mean_squared_error(y_test_r, knn.predict(X_test_r)):.4f}")


print("\n\n--- PROBLEM 2: KLASYFIKACJA (Klasa Jakości) ---\n")

print("METODA A: Las Losowy (Random Forest Classifier)")
print("Parametr 1: Liczba drzew (n_estimators)")
for n in [10, 50, 100, 200]:
    rfc = RandomForestClassifier(n_estimators=n, random_state=42)
    rfc.fit(X_train_c, y_train_c)
    print(f"Drzewa: {n:3d} | Skuteczność Train: {accuracy_score(y_train_c, rfc.predict(X_train_c))*100:.2f}% | Test: {accuracy_score(y_test_c, rfc.predict(X_test_c))*100:.2f}%")

print("\nParametr 2: Maksymalna głębokość drzewa (max_depth)")
for d in [5, 10, 20, None]:
    rfc = RandomForestClassifier(max_depth=d, random_state=42)
    rfc.fit(X_train_c, y_train_c)
    d_str = str(d) if d is not None else "Brak"
    print(f"Głębokość: {d_str:>4s} | Skuteczność Train: {accuracy_score(y_train_c, rfc.predict(X_train_c))*100:.2f}% | Test: {accuracy_score(y_test_c, rfc.predict(X_test_c))*100:.2f}%")

print("\nParametr 3: Minimalna liczba próbek w liściu (min_samples_leaf)")
for m in [1, 2, 5, 10]:
    rfc = RandomForestClassifier(min_samples_leaf=m, random_state=42)
    rfc.fit(X_train_c, y_train_c)
    print(f"Próbki w liściu: {m:2d} | Skuteczność Train: {accuracy_score(y_train_c, rfc.predict(X_train_c))*100:.2f}% | Test: {accuracy_score(y_test_c, rfc.predict(X_test_c))*100:.2f}%")


print("\nMETODA B: K-Najbliższych Sąsiadów (KNN Classifier)")
print("Parametr 1: Liczba sąsiadów (n_neighbors)")
for k in [3, 5, 9, 15]:
    knnc = KNeighborsClassifier(n_neighbors=k)
    knnc.fit(X_train_c, y_train_c)
    print(f"Sąsiedzi: {k:2d} | Skuteczność Train: {accuracy_score(y_train_c, knnc.predict(X_train_c))*100:.2f}% | Test: {accuracy_score(y_test_c, knnc.predict(X_test_c))*100:.2f}%")

print("\nParametr 2: Metryka odległości (p)")
for p_val in [1, 2, 3, 4]:
    knnc = KNeighborsClassifier(p=p_val)
    knnc.fit(X_train_c, y_train_c)
    print(f"Metryka p: {p_val:2d} | Skuteczność Train: {accuracy_score(y_train_c, knnc.predict(X_train_c))*100:.2f}% | Test: {accuracy_score(y_test_c, knnc.predict(X_test_c))*100:.2f}%")

print("\nParametr 3: Algorytm wyszukiwania sąsiadów")
for alg in ['ball_tree', 'kd_tree', 'brute', 'auto']:
    knnc = KNeighborsClassifier(algorithm=alg)
    knnc.fit(X_train_c, y_train_c)
    print(f"Algorytm: {alg:9s} | Skuteczność Train: {accuracy_score(y_train_c, knnc.predict(X_train_c))*100:.2f}% | Test: {accuracy_score(y_test_c, knnc.predict(X_test_c))*100:.2f}%")