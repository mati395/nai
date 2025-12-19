# Lab5 – Klasyfikacja z użyciem Sieci Neuronowych

## Źródła danych

| Zbiór danych               | Typ zadania               | Link do opisu / danych                                                                 |
|----------------------------|---------------------------|----------------------------------------------------------------------------------------|
| Penguins                   | Klasyfikacja wieloklasowa | [Kaggle](https://www.kaggle.com/datasets/kainatjamil12/pengunis)                         |
| CIFAR-10                   | Klasyfikacja obrazów      | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                                |
| Fashion-MNIST             | Klasyfikacja obrazów      | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)                       |
| Spambase                   | Klasyfikacja binarna      | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/spambase)                   |

---

## Opis projektu
Projekt realizuje zadanie laboratoryjne z klasyfikacji danych przy użyciu **Sieci Neuronowych (Neural Networks)** z frameworkiem **TensorFlow/Keras**. Skrypt obejmuje:
1. Klasyfikację pingwinów z porównaniem skuteczności sieci neuronowej z poprzednimi metodami (Decision Tree, SVM).
2. Porównanie dwóch rozmiarów sieci neuronowych dla zbioru pingwinów.
3. Rozpoznawanie zwierząt z wykorzystaniem zbioru CIFAR-10.
4. Rozpoznawanie ubrań z wykorzystaniem zbioru Fashion-MNIST.
5. Detekcję spamu w e-mailach (Spambase) z analizą najważniejszych cech.
6. Wizualizację macierzy pomyłek (confusion matrix) dla klasyfikacji pingwinów.
7. Analizę feature importance dla detekcji spamu.

---

## Uruchomienie projektu
1. **Wymagania**:
   - Python 3.8+
   - Biblioteki: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
   - Zainstaluj zależności:
     ```bash
     pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
     ```
2. **Przygotowanie danych**:
   - Upewnij się, że pliki datasetów znajdują się w katalogu `Lab5/data/`:
     - `penguins.csv`
     - `spambase.data`
   - Zbiory CIFAR-10 i Fashion-MNIST zostaną automatycznie pobrane przy pierwszym uruchomieniu
3. **Uruchomienie**:
   ```bash
   python main.py
   ```

---

## Wizualizacje:

![alt text]()

![alt text]()

---

## Output programu:
  
 ![alt text]()

 
---

## Framework
**TensorFlow/Keras** - framework do budowy i trenowania sieci neuronowych

## Autorzy
Mateusz Andrzejak, Szymon Anikej

