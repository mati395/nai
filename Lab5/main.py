"""
PROBLEM: Rozwiązywanie problemów klasyfikacji przy użyciu Sieci Neuronowych (TensorFlow/Keras).
AUTOR: [Twoje Imię]
DATA: 2023-10-27
OPIS:
    Skrypt realizuje 4 zadania klasyfikacji:
    1. Pingwiny (Porównanie ANN z klasycznym ML).
    2. CIFAR-10 (Rozpoznawanie obrazów - zwierzęta/pojazdy).
    3. Fashion MNIST (Rozpoznawanie ubrań + Macierz Pomyłek).
    4. Spambase (Wykrywanie spamu w e-mailach - własny przypadek użycia).

INSTRUKCJA:
    1. Upewnij się, że plik 'spambase.data' jest w katalogu roboczym.
    2. Uruchom skrypt: `python main.py`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Konfiguracja
np.random.seed(42)
tf.random.set_seed(42)

def plot_history(history, title="Model History"):
    """Rysuje wykresy straty i dokładności."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'ro', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, classes, title="Confusion Matrix"):
    """Rysuje heatmapę macierzy pomyłek."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title(title)
    plt.show()

# ==========================================
# ZADANIE 1: PENGUINS
# ==========================================
def task_1_penguins():
    print(f"\n{'='*40}\nZADANIE 1: PENGUINS\n{'='*40}")
    try:
        df = sns.load_dataset('penguins').dropna()
    except:
        print("Brak dostępu do internetu/zbioru pingwinów.")
        return

    le = LabelEncoder()
    df['species_encoded'] = le.fit_transform(df['species'])
    X = pd.get_dummies(df.drop(['species', 'species_encoded'], axis=1), drop_first=True)
    y = df['species_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ANN
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=30, verbose=0, validation_split=0.2)

    print(f"Penguins ANN Accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
    plot_history(history, "Penguins ANN")

# ==========================================
# ZADANIE 2: CIFAR-10
# ==========================================
def task_2_cifar10():
    print(f"\n{'='*40}\nZADANIE 2: CIFAR-10\n{'='*40}")
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train, X_test = X_train[:5000] / 255.0, X_test[:1000] / 255.0 # Ograniczenie dla szybkości
    y_train, y_test = y_train[:5000], y_test[:1000]

    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, verbose=1, validation_split=0.1)
    print(f"CIFAR-10 Accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")

# ==========================================
# ZADANIE 3: FASHION MNIST
# ==========================================
def task_3_fashion():
    print(f"\n{'='*40}\nZADANIE 3: FASHION MNIST\n{'='*40}")
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    model = keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, verbose=1)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    plot_confusion_matrix_heatmap(y_test, y_pred, classes, "Fashion MNIST Confusion Matrix")

# ==========================================
# ZADANIE 4: SPAMBASE (ZASKOCZ MNIE)
# ==========================================
def task_4_spambase():
    print(f"\n{'='*40}\nZADANIE 4: SPAM DETECTION (UCI SPAMBASE)\n{'='*40}")

    # Ładowanie danych
    # Plik .data nie ma nagłówka. 57 kolumn cech + 1 kolumna targetu (ostatnia)
    try:
        df = pd.read_csv('spambase.data', header=None)
    except FileNotFoundError:
        print("BŁĄD: Nie znaleziono pliku 'spambase.data'. Upewnij się, że jest w katalogu.")
        return

    print(f"Wczytano dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn.")

    # Podział na cechy (X) i etykiety (y)
    # Ostatnia kolumna to klasa (1 = spam, 0 = not spam)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Skalowanie danych (Ważne dla sieci neuronowych!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Podział trening/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Budowa modelu
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),                 # Dropout zapobiega przeuczeniu
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Sigmoid dla klasyfikacji binarnej (0 lub 1)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Trening
    print("Rozpoczynam trening sieci wykrywającej spam...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

    # Ewaluacja
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[WYNIK] Spambase Test Accuracy: {acc:.4f}")

    # Wykresy
    plot_history(history, "Spam Detection Model")

    # Macierz pomyłek dla Spamu
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    plot_confusion_matrix_heatmap(y_test, y_pred, ['Not Spam', 'Spam'], "Spambase Confusion Matrix")

if __name__ == "__main__":
    # Uruchamiamy zadania po kolei
    # task_1_penguins() # Odkomentuj jeśli chcesz uruchomić
    # task_2_cifar10()
    # task_3_fashion()
    task_4_spambase()