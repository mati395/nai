# System rekomendacji filmów i seriali
## Autorzy: Mateusz Andrzejak, Szymon Anikej
### Data: 2025-11-20

## Opis
Silnik rekomendacyjny oparty na filtrowaniu kolaboratywnym. Na podstawie ocen użytkowników generuje:
- spersonalizowane rekomendacje,
- antyrekomendacje (czego unikać),
- listy naj- i najgorzej ocenianych tytułów.

Połączony z TMDB API w celu pobierania dodatkowych informacji o filmach i serialach (tytuł, opis, reżyser, rok, gatunki, ocena TMDB).

## Przygotowanie środowiska
1. Zainstaluj zależności:
   ```bash
   pip install pandas numpy scipy requests python-dotenv
   ```
2. W katalogu `Lab3` umieść pliki:
   - `main.py`
   - `data.csv` (źródło ocen w formacie `user,movie,rating,...`)
   - `.env` zawierający `TMDB_API_KEY=twój_klucz`

## Uruchomienie
```bash
python main.py
```
Skrypt:
1. Wczytuje i przetwarza dane (`load_and_process_data`).
2. Buduje macierz użytkownik–film (`pivot_table`).
3. Oblicza macierz podobieństw między użytkownikami (Pearson lub SVD).
4. Pozwala wybrać użytkownika z listy i prezentuje wyniki wraz z wyjaśnieniami.

## Funkcjonalności
- **Rekomendacje (`get_recommendations`)**  
  Dla filmów nieocenionych przez użytkownika wylicza przewidywane oceny, wskazuje, którzy użytkownicy wpłynęli na sugestię (ocena + podobieństwo).
- **Antyrekomendacje (`get_anti_recommendations`)**  
  Analogicznie, lecz sortuje od najniższych przewidywanych ocen.
- **Fallback**  
  Gdy brak danych dla konkretnego użytkownika, prezentuje listę top 5 lub najgorszych filmów wg średniej oceny.
- **Integracja z TMDB**  
  Funkcja `get_movie_info` pobiera opis, reżysera, gatunki i rok, najpierw szukając filmu, a potem serialu. W przypadku braku wyniku zwraca wartości domyślne.

## Dane wejściowe
Plik `data.csv` powinien mieć strukturę:
```
użytkownik,tytuł1,ocena1,tytuł2,ocena2,...
```
Każdy wiersz reprezentuje użytkownika oraz kolejne pary (film, ocena). Oceny są liczbami całkowitymi.

## Uwagi techniczne
- Domyślnie używany jest współczynnik korelacji Pearsona. Można przełączyć na wariant SVD, zmieniając parametr `method` w `compute_similarity`.
- Zapytania do TMDB są wykonywane z limitem czasu 5 s; w razie błędu API zwracane są dane zastępcze.
- Próg podobieństwa można regulować parametrem `similarity_threshold`.


