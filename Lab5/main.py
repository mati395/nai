"""
Lab5 – Klasyfikacja z użyciem Sieci Neuronowych
================================================

Problem:
    Realizacja klasyfikacji danych przy użyciu sieci neuronowych (neural networks)
    na różnych zbiorach danych.

Autor:
    Mateusz Andrzejak, Szymon Anikej

Instrukcja użycia:
    1. Zainstaluj wymagane biblioteki:
       pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
    
    2. Upewnij się, że pliki datasetów znajdują się w katalogu Lab5/data/:
       - penguins.csv
       - spambase.data
    
    3. Uruchom skrypt:
       python main.py
    
    4. Program automatycznie pobierze zbiory CIFAR-10 i Fashion-MNIST przy pierwszym uruchomieniu

Referencje:
    - Artificial Neural Networks for Beginners: 
      https://www.researchgate.net/publication/1956697_Artificial_Neural_Networks_for_Beginners
    - Artificial Neural Networks tutorial:
      https://www.researchgate.net/publication/261392616_Artificial_Neural_Networks_tutorial
    - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
    - Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
    - Spambase Dataset: UCI Machine Learning Repository

Framework:
    TensorFlow/Keras - framework do budowy i trenowania sieci neuronowych
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Wyciszenie informacyjnych komunikatów TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, fashion_mnist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Ustawienia
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PENGUINS_CSV_PATH = os.path.join(DATA_DIR, "penguins.csv")
SPAMBASE_DATA_PATH = os.path.join(DATA_DIR, "spambase.data")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def plot_confusion_matrix(y_true, y_pred, class_names, title: str):
    """Rysuje macierz pomyłek."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("Prawdziwa klasa")
    plt.xlabel("Przewidziana klasa")
    plt.tight_layout()
    plt.show()


# === PUNKT 1: PENGUINS ========================================================

def point1_penguins():
    """Punkt 1: Klasyfikacja pingwinów - sieć neuronowa vs. poprzednie metody."""
    print("\n" + "="*60)
    print("PUNKT 1: Klasyfikacja Pingwinów")
    print("="*60)
    
    # Wczytanie danych
    df = pd.read_csv(PENGUINS_CSV_PATH)
    feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    df = df.dropna(subset=feature_cols + ["species"])
    
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df["species"].values)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Porównanie dwóch rozmiarów sieci
    print("\n[Porównanie dwóch rozmiarów sieci neuronowych]")
    
    # Sieć mniejsza
    model_small = models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(len(np.unique(y)), activation="softmax")
    ])
    model_small.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model_small.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
    acc_small = model_small.evaluate(X_test_scaled, y_test, verbose=0)[1]
    print(f"Mniejsza sieć (32-16):     {acc_small:.4f}")
    
    # Sieć większa
    model_large = models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(np.unique(y)), activation="softmax")
    ])
    model_large.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model_large.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
    acc_large = model_large.evaluate(X_test_scaled, y_test, verbose=0)[1]
    print(f"Większa sieć (128-64-32):  {acc_large:.4f}")
    
    # Porównanie z poprzednimi metodami
    print("\n[Porównanie z poprzednimi metodami]")
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    acc_tree = accuracy_score(y_test, tree.predict(X_test))
    print(f"Decision Tree:        {acc_tree:.4f}")
    
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE))
    svm.fit(X_train, y_train)
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    print(f"SVM (RBF):            {acc_svm:.4f}")
    print(f"Neural Network (mała): {acc_small:.4f}")
    print(f"Neural Network (duża): {acc_large:.4f}")
    
    # Confusion Matrix dla najlepszej sieci neuronowej
    y_pred_large = np.argmax(model_large.predict(X_test_scaled, verbose=0), axis=1)
    class_names = ["Adelie", "Chinstrap", "Gentoo"]
    plot_confusion_matrix(y_test, y_pred_large, class_names, "Penguins - Confusion Matrix (Neural Network)")


# === PUNKT 2: CIFAR-10 ========================================================

def point2_cifar10():
    """Punkt 2: Rozpoznawanie zwierząt z CIFAR-10."""
    print("\n" + "="*60)
    print("PUNKT 2: CIFAR-10 - Rozpoznawanie Zwierząt")
    print("="*60)
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Filtrowanie tylko zwierząt (bird, cat, deer, dog, frog, horse)
    animal_classes = [2, 3, 4, 5, 6, 7]
    train_mask = np.isin(y_train.flatten(), animal_classes)
    test_mask = np.isin(y_test.flatten(), animal_classes)
    
    X_train_animals = X_train[train_mask] / 255.0
    y_train_animals = y_train[train_mask].flatten()
    X_test_animals = X_test[test_mask] / 255.0
    y_test_animals = y_test[test_mask].flatten()
    
    # Mapowanie klas do 0-5
    class_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    y_train_animals = np.array([class_mapping[y] for y in y_train_animals])
    y_test_animals = np.array([class_mapping[y] for y in y_test_animals])
    
    print(f"Liczba próbek treningowych: {len(X_train_animals)}")
    print(f"Liczba próbek testowych: {len(X_test_animals)}")
    
    # Sieć konwolucyjna
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(6, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train_animals, y_train_animals, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    
    test_acc = model.evaluate(X_test_animals, y_test_animals, verbose=0)[1]
    print(f"\nAccuracy na zbiorze testowym: {test_acc:.4f}")


# === PUNKT 3: FASHION-MNIST ===================================================

def point3_fashion_mnist():
    """Punkt 3: Rozpoznawanie ubrań z Fashion-MNIST."""
    print("\n" + "="*60)
    print("PUNKT 3: Fashion-MNIST - Rozpoznawanie Ubrania")
    print("="*60)
    
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    print(f"Liczba próbek treningowych: {len(X_train)}")
    print(f"Liczba próbek testowych: {len(X_test)}")
    
    # Sieć konwolucyjna
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
    
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"\nAccuracy na zbiorze testowym: {test_acc:.4f}")


# === PUNKT 4: SPAMBASE ========================================================

def point4_spambase():
    """
    Punkt 4: Detekcja spamu w e-mailach (Spambase).
    
    Zbiór danych Spambase zawiera 57 cech opisujących e-maile:
    - Częstość występowania określonych słów (word_freq_*)
    - Częstość występowania określonych znaków (char_freq_*)
    - Statystyki dotyczące ciągów wielkich liter (capital_run_length_*)
    
    Sieć neuronowa uczy się klasyfikować e-maile na spam (1) lub nie-spam (0)
    na podstawie tych cech tekstowych.
    """
    print("\n" + "="*60)
    print("PUNKT 4: Spambase - Detekcja Spamu")
    print("="*60)
    
    # Wczytanie danych
    data = np.loadtxt(SPAMBASE_DATA_PATH, delimiter=",")
    X = data[:, :-1]  # 57 cech
    y = data[:, -1].astype(int)  # Etykieta: 0 = nie-spam, 1 = spam
    
    print(f"Liczba próbek: {len(X)}")
    print(f"Liczba cech: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sieć neuronowa
    model = models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)
    
    test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    print(f"\nAccuracy na zbiorze testowym: {test_acc:.4f}")
    
    # Analiza najważniejszych cech (feature importance)
    print("\n[Analiza najważniejszych cech]")
    
    # Nazwy cech (zgodnie z formatem spambase.names)
    feature_names = [
        "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
        "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
        "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
        "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
        "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
        "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
        "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
        "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
        "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
        "word_freq_edu", "word_freq_table", "word_freq_conference",
        "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#",
        "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"
    ]
    
    # Obliczenie różnicy średnich wartości cech między spamem a nie-spamem
    spam_mask = y_train == 1
    non_spam_mask = y_train == 0
    
    spam_means = np.mean(X_train[spam_mask], axis=0)
    non_spam_means = np.mean(X_train[non_spam_mask], axis=0)
    feature_diff = spam_means - non_spam_means
    
    # Sortowanie cech według różnicy (wartość bezwzględna)
    feature_importance = np.abs(feature_diff)
    top_indices = np.argsort(feature_importance)[::-1][:15]  # Top 15 cech
    
    print("\nTop 15 cech najbardziej związanych ze spamem:")
    print("-" * 70)
    print(f"{'Cecha':<35} {'Spam':>10} {'Nie-spam':>10} {'Różnica':>10}")
    print("-" * 70)
    for idx in top_indices:
        feat_name = feature_names[idx]
        spam_val = spam_means[idx]
        non_spam_val = non_spam_means[idx]
        diff = feature_diff[idx]
        print(f"{feat_name:<35} {spam_val:>10.4f} {non_spam_val:>10.4f} {diff:>10.4f}")
    
    # Wizualizacja top 15 cech
    top_features = [feature_names[i] for i in top_indices]
    top_diffs = [feature_diff[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if d > 0 else 'blue' for d in top_diffs]
    plt.barh(range(len(top_features)), top_diffs, color=colors)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Różnica średnich wartości (Spam - Nie-spam)')
    plt.title('Top 15 cech najbardziej związanych ze spamem\n(Czerwony = wyższe w spamie, Niebieski = wyższe w nie-spamie)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nInterpretacja:")
    print("- Czerwone słupki: cechy częściej występujące w spamie")
    print("- Niebieskie słupki: cechy częściej występujące w nie-spamie")
    print("- Im większa wartość bezwzględna, tym ważniejsza cecha dla klasyfikacji")


# === MAIN ======================================================================

def main():
    """Główna funkcja uruchamiana z linii komend."""
    print("="*60)
    print("Lab5 – Klasyfikacja z użyciem Sieci Neuronowych")
    print("Framework: TensorFlow/Keras")
    print("="*60)
    
    try:
        point1_penguins()
        point2_cifar10()
        point3_fashion_mnist()
        point4_spambase()
        
        print("\n" + "="*60)
        print("Zakończono wszystkie zadania")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nBłąd: Nie znaleziono pliku - {e}")
        print(f"Upewnij się, że pliki datasetów znajdują się w katalogu: {DATA_DIR}")
        return
    except Exception as e:
        print(f"\nBłąd podczas wykonywania: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
