"""
Lab4 – prosta klasyfikacja z użyciem Drzewa Decyzyjnego i SVM
=============================================================

Zadanie:
    1. Wybrać dwa zbiory danych do klasyfikacji.
    2. Nauczyć drzewo decyzyjne oraz SVM (koniecznie!) klasyfikować dane.
    3. Pokazać metryki jakości klasyfikacji (accuracy, raport klasyfikacji, macierz pomyłek).
    4. Przygotować przykładowe wizualizacje danych.
    5. Wywołać klasyfikatory dla przykładowych danych wejściowych.
    6. Zaprezentować różne kernel function w SVM i krótko omówić ich wpływ.

Wykorzystane zbiory danych
--------------------------

1) Banknote Authentication Dataset (binary classification)
   Źródło (Machine Learning Mastery – standard datasets):
   - opis: https://machinelearningmastery.com/standard-machine-learning-datasets/
   - bezpośredni plik CSV: https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv

2) Penguins Dataset (multi-class classification)
   Źródło (Kaggle, unikatowy zbiór w grupie):
   - opis + plik: https://www.kaggle.com/datasets/kainatjamil12/pengunis
   - w repozytorium zapisany jako lokalny plik: Lab4/penguins.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

PENGUINS_CSV_PATH = "penguins.csv"

# === FUNKCJE POMOCNICZE ========================================================

def print_section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


# === WCZYTANIE DANYCH – BANKNOTE ==============================================

def load_banknote_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Wczytuje Banknote Authentication Dataset z publicznego URL.

    Link do danych (wymagany w zadaniu):
        https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv

    Kolumny (wg opisu zbioru):
        1. variance    – wariancja falkowa obrazu banknotu
        2. skewness    – skośność
        3. curtosis    – kurtoza
        4. entropy     – entropia
        5. class       – etykieta (0 = fałszywy, 1 = prawdziwy)

    Zwraca:
        df  – pełny DataFrame z kolumną 'class',
        X   – cechy numeryczne,
        y   – wektor etykiet klas.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv"
    cols = ["variance", "skewness", "curtosis", "entropy", "class"]
    df = pd.read_csv(url, header=None, names=cols)

    X = df[["variance", "skewness", "curtosis", "entropy"]].copy()
    y = df["class"].astype(int)
    return df, X, y


def visualize_banknote(df: pd.DataFrame) -> None:
    """
    Prosta wizualizacja danych banknotów:
        * macierz korelacji cech,
        * wykres punktowy dwóch najważniejszych cech (variance vs skewness).
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 5))
    corr = df[["variance", "skewness", "curtosis", "entropy"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Banknote – macierz korelacji cech")
    plt.tight_layout()
    plt.show()

    # Scatter: variance vs skewness
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="variance",
        y="skewness",
        hue="class",
        palette="Set1",
        alpha=0.7,
    )
    plt.title("Banknote – variance vs skewness (kolor = klasa)")
    plt.tight_layout()
    plt.show()


# === WCZYTANIE DANYCH – PENGUINS ==============================================

def load_penguins_dataset(csv_path: str = PENGUINS_CSV_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Wczytuje Penguins Dataset z lokalnego pliku CSV (pobrany z Kaggle).

    Link do danych (wymagany w zadaniu):
        https://www.kaggle.com/datasets/kainatjamil12/pengunis

    W pliku `penguins.csv` znajdują się m.in. kolumny:
        * species – gatunek pingwina (zmienna celu, klasy wieloklasowe),
        * bill_length_mm, bill_depth_mm,
        * flipper_length_mm, body_mass_g,
        * sex, island, year.

    Zwraca:
        df  – DataFrame po wstępnym czyszczeniu,
        X   – cechy numeryczne,
        y   – etykiety gatunku (species).
    """
    df = pd.read_csv(csv_path)

    # usuwamy rekordy z brakami w kluczowych kolumnach
    feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    df = df.dropna(subset=feature_cols + ["species"])

    X = df[feature_cols].copy()
    y = df["species"].astype("category")
    return df, X, y


def visualize_penguins(df: pd.DataFrame) -> None:
    """
    Prosta wizualizacja zbioru pingwinów:
        * scatterplot flipper_length_mm vs body_mass_g, kolor = gatunek.
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="flipper_length_mm",
        y="body_mass_g",
        hue="species",
        style="sex",
        alpha=0.8,
        palette="Set2",
    )
    plt.title("Penguins – długość płetwy vs masa ciała")
    plt.tight_layout()
    plt.show()


# === TRENING MODELI – WSPÓLNA FUNKCJA =========================================

def train_and_evaluate_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
) -> tuple[DecisionTreeClassifier, make_pipeline]:
    """
    Dzieli dane na train/test, trenuje drzewo decyzyjne i SVM (RBF),
    a następnie wypisuje podstawowe metryki klasyfikacji.

    Argumenty:
        X           – macierz cech (DataFrame lub ndarray),
        y           – wektor etykiet,
        dataset_name – nazwa zbioru (do opisu w wydruku).

    Zwraca:
        tree_model  – drzewo decyzyjne wytrenowane na CAŁYM zbiorze,
        svm_model   – SVM (pipeline: StandardScaler + SVC) na CAŁYM zbiorze.
    """
    print_section(f"Klasyfikacja – {dataset_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)

    print("\n[Drzewo decyzyjne]")
    print("Accuracy:", round(accuracy_score(y_test, y_pred_tree), 4))
    print("Macierz pomyłek:\n", confusion_matrix(y_test, y_pred_tree))
    print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_tree))

    # SVM z kernelem RBF (domyślny wariant)
    svm_rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0))
    svm_rbf.fit(X_train, y_train)
    y_pred_svm = svm_rbf.predict(X_test)

    print("\n[SVM – kernel RBF]")
    print("Accuracy:", round(accuracy_score(y_test, y_pred_svm), 4))
    print("Macierz pomyłek:\n", confusion_matrix(y_test, y_pred_svm))
    print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_svm))

    # Modele na pełnym zbiorze – do demonstracji predykcji
    tree_full = DecisionTreeClassifier(random_state=0).fit(X, y)
    svm_full = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)).fit(X, y)
    return tree_full, svm_full


# === PORÓWNANIE RÓŻNYCH KERNELI SVM ===========================================

def compare_svm_kernels(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
) -> None:
    """
    Porównuje różne konfiguracje SVM (różne kernelle i parametry) na podanym zbiorze.

    Pokazywane są głównie dokładności (accuracy), aby zobaczyć,
    jak zmiana kernela i parametrów wpływa na wynik.
    """
    print_section(f"SVM – porównanie kernel function ({dataset_name})")

    configs = [
        ("linear_C1", dict(kernel="linear", C=1.0)),
        ("linear_C10", dict(kernel="linear", C=10.0)),
        ("rbf_default", dict(kernel="rbf", C=1.0, gamma="scale")),
        ("rbf_C10_gamma0.1", dict(kernel="rbf", C=10.0, gamma=0.1)),
        ("poly_deg3", dict(kernel="poly", C=1.0, degree=3, gamma="scale")),
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    results: list[tuple[str, str, float]] = []
    for name, params in configs:
        model = make_pipeline(StandardScaler(), SVC(random_state=0, **params))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, params["kernel"], acc))
        print(f"{name:16s} | kernel={params['kernel']:6s} | accuracy={acc:.4f}")

    best = max(results, key=lambda r: r[2])
    worst = min(results, key=lambda r: r[2])

    print("\nPodsumowanie (krótko):")
    print(f"* Najlepsza konfiguracja:  {best[0]} (kernel={best[1]}, accuracy={best[2]:.4f})")
    print(f"* Najsłabsza konfiguracja: {worst[0]} (kernel={worst[1]}, accuracy={worst[2]:.4f})")
    print("* Kernel RBF zwykle dobrze modeluje nieliniowe granice między klasami.")
    print("* Kernel liniowy jest prostszy – działa dobrze, gdy podział jest prawie liniowy.")
    print("* Kernel wielomianowy (poly) może uchwycić bardziej złożone zależności, ale łatwiej o przeuczenie.")


# === PRZYKŁADOWE PREDYKCJE ====================================================

def demo_predictions_banknote(tree_model: DecisionTreeClassifier, svm_model: make_pipeline) -> None:
    """
    Pokazuje przykładowe predykcje dla kilku sztucznych wektorów cech Banknote.

    Kolumny:
        variance, skewness, curtosis, entropy

    Wynik:
        0 – banknot fałszywy,
        1 – banknot prawdziwy.
    """
    samples = pd.DataFrame(
        {
            "variance": [3.0, -2.0, 1.0],
            "skewness": [7.0, 0.5, -3.0],
            "curtosis": [0.5, -2.0, 1.0],
            "entropy": [1.0, 0.0, -1.0],
        }
    )

    print_section("Przykładowe predykcje – Banknote")
    print("Wejściowe przykłady:\n", samples)

    tree_pred = tree_model.predict(samples)
    svm_pred = svm_model.predict(samples)

    print("Drzewo decyzyjne:", tree_pred.tolist())
    print("SVM (kernel=rbf):", svm_pred.tolist())


def demo_predictions_penguins(tree_model: DecisionTreeClassifier, svm_model: make_pipeline) -> None:
    """
    Pokazuje przykładowe predykcje dla kilku sztucznych pingwinów.

    Cechy:
        bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g

    Wynik:
        nazwa gatunku (np. Adelie, Gentoo, Chinstrap).
    """
    samples = pd.DataFrame(
        {
            "bill_length_mm": [40.0, 50.0, 46.0],
            "bill_depth_mm": [18.0, 15.0, 17.5],
            "flipper_length_mm": [190, 220, 200],
            "body_mass_g": [4000, 5000, 3800],
        }
    )

    print_section("Przykładowe predykcje – Penguins")
    print("Wejściowe przykłady:\n", samples)

    tree_pred = tree_model.predict(samples)
    svm_pred = svm_model.predict(samples)

    print("Drzewo decyzyjne:", tree_pred.tolist())
    print("SVM (kernel=rbf):", svm_pred.tolist())


# === PODSUMOWANIE WPLYWU KERNELI ======================

KERNEL_SUMMARY = """
Krótka interpretacja wpływu kernel function w SVM (z obserwacji na obu zbiorach):

1) Kernel liniowy (linear)
   * Tworzy liniową granicę decyzyjną w przestrzeni cech.
   * Działa dobrze, gdy dane są w dużym stopniu liniowo separowalne.
   * Na zbiorach Banknote i Penguins zwykle jest poprawny, ale nie zawsze najlepszy.

2) Kernel RBF (radial basis function)
   * Tworzy nieliniową, gładką granicę decyzyjną.
   * Bardzo elastyczny – dobrze radzi sobie z bardziej skomplikowanymi zależnościami.
   * Na naszych danych często osiąga najwyższą lub zbliżoną do najwyższej accuracy.

3) Kernel wielomianowy (poly)
   * Pozwala modelować złożone (wielomianowe) relacje między cechami.
   * Może poprawić wyniki, ale przy nieoptymalnych parametrach (degree, C, gamma)
     łatwo o przeuczenie i pogorszenie generalizacji.

W praktyce zbiory w tym zadaniu potwierdzają typowy scenariusz:
kernel RBF jest bezpiecznym i mocnym wyborem, kernel liniowy bywa dobry przy prostszych
zależnościach, a kernel wielomianowy wymaga ostrożnego strojenia.
"""


# === MAIN =====================================================================

def main():
    """
    Główna funkcja uruchamiana z linii komend.

    Kolejne kroki:
        1. Wczytanie i wizualizacja zbioru Banknote.
        2. Trening drzewa decyzyjnego i SVM (RBF) dla Banknote + metryki.
        3. Porównanie różnych kernel function (Banknote).
        4. Predykcje na przykładowych danych (Banknote).
        5. Wczytanie i wizualizacja zbioru Penguins.
        6. Trening drzewa decyzyjnego i SVM (RBF) dla Penguins + metryki.
        7. Porównanie różnych kernel function (Penguins).
        8. Predykcje na przykładowych danych (Penguins).
        9. Wydruk krótkiego tekstowego podsumowania wpływu kernel function.
    """
    # --- Zbiór 1: Banknote ----------------------------------------------------
    df_bank, X_bank, y_bank = load_banknote_dataset()
    print_section("Informacje o zbiorze – Banknote Authentication")
    print(df_bank.head())
    print("Liczba próbek:", len(df_bank))
    print("Rozkład klas:\n", y_bank.value_counts())

    visualize_banknote(df_bank)
    bank_tree, bank_svm = train_and_evaluate_classifiers(X_bank, y_bank, "Banknote Authentication")
    compare_svm_kernels(X_bank, y_bank, "Banknote Authentication")
    demo_predictions_banknote(bank_tree, bank_svm)

    # --- Zbiór 2: Penguins ----------------------------------------------------
    df_peng, X_peng, y_peng = load_penguins_dataset(PENGUINS_CSV_PATH)
    print_section("Informacje o zbiorze – Penguins")
    print(df_peng[["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].head())
    print("Liczba próbek:", len(df_peng))
    print("Rozkład klas:\n", y_peng.value_counts())

    visualize_penguins(df_peng)
    peng_tree, peng_svm = train_and_evaluate_classifiers(X_peng, y_peng, "Penguins")
    compare_svm_kernels(X_peng, y_peng, "Penguins")
    demo_predictions_penguins(peng_tree, peng_svm)

    # --- Podsumowanie wpływu kernel function ----------------------------------
    print_section("Podsumowanie wpływu kernel function (SVM)")
    print(KERNEL_SUMMARY)


if __name__ == "__main__":
    main()
