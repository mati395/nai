# Gra: Connect Four (Cztery w rzędzie)
## Autorzy: Mateusz Andrzejak, Szymon Anikej
### Data: 2025-10-04

## Opis:
Gra dwuosobowa **"Cztery w rzędzie" (Connect Four)**, w której gracz mierzy się ze sztuczną inteligencją 
zaimplementowaną przy użyciu algorytmu Adversarial Search (**Negamax** z biblioteki **easyAI**).

## Instrukcja przygotowania środowiska:
1. Zainstaluj wymagane moduły:
   ```python 
   pip install easyAI numpy
   ```
2. Uruchom skrypt w terminalu poleceniem:
    ```python
   python main.py
   ```
3. Wybieraj kolumny (0–6), aby wykonywać ruchy.

## Zasady:
- Gra toczy się na planszy **6x7**.
```text
0 1 2 3 4 5 6
---------------
| . . . . . . |
| . . . . . . |
| . . . . . . |
| . . . . . . |
| . . . . . . |
| . . . . . . |
---------------
```
Gracz 1, twój ruch (wpisz numer kolumny 0-6):
- Gracze wykonują ruchy na przemian, zaczynając od Gracza 1 **(domyślnie: Ty)**.
- Celem jest ułożenie czterech swoich pionków w jednej linii: **poziomo**, **pionowo** lub **na ukos**.
- Gra kończy się wygraną jednego z graczy lub remisem, gdy plansza jest pełna.

## Dostosowywanie:
Poziom trudności AI jest kontrolowany przez głębokość przeszukiwania algorytmu Negamax. Domyślnie jest to 6:
```python
# Negamax(6) oznacza, że AI patrzy 6 ruchów w przód
ai_algo = Negamax(6)
```
Możesz zwiększyć tę liczbę, aby AI grało "mądrzej" (ale będzie też potrzebowało więcej czasu na obliczenia), lub zmniejszyć ją, aby grało szybciej i prościej.

**Kolejność ruchów**
Kolejność ruchów: Domyślnie grę rozpoczyna człowiek (Gracz 1). Aby AI rozpoczęła grę jako pierwsza, należy zamienić kolejność graczy podczas inicjalizacji obiektu game (zgodnie z komentarzem w kodzie):
```python
# Aby AI grało jako pierwsze:
game = ConnectFour([AI_Player(ai_algo), Human_Player()])
```
