# FuzzyPlantWatering – Rozmyte sterowanie podlewaniem
## Autorzy: Mateusz Andrzejak, Szymon Anikej
### Data: 2025-10-30

## Opis
Autonomiczny system sterowania pompą podlewania z wykorzystaniem logiki rozmytej. Kontroler analizuje trzy sygnały wejściowe – wilgotność gleby, temperaturę powietrza oraz poziom światła – i zwraca rekomendowaną intensywność podlewania (0‑100%). W projekcie pokazujemy, jak na podstawie reguł lingwistycznych podejmować decyzje w czasie zbliżonym do rzeczywistego.

## Przygotowanie środowiska
1. Utwórz i aktywuj wirtualne środowisko:
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```
2. Zainstaluj zależności:
   ```bash
   pip install "numpy<2.0" "scipy<1.14" "scikit-fuzzy<1.14"
   pip install matplotlib
   pip install networkx
   ```

## Uruchomienie
```bash
python watering.py
```
Skrypt:
- buduje kompletny kontroler rozmyty (`skfuzzy.control`),
- wykonuje dwa scenariusze symulacyjne,
- opcjonalnie rysuje funkcje przynależności wszystkich zmiennych (do wykorzystania w sprawozdaniu).

## Wejścia/Wyjścia kontrolera
- `soil_moisture` (0–100 %): zbiory `dry`, `okay`, `wet`
- `air_temp` (0–40 °C): zbiory `low`, `medium`, `high`
- `light_level` (0–10): zbiory `dark`, `normal`, `bright`
- Wyjście `water_pump` (0–100 %): zbiory `off`, `medium`, `full`

## Logika sterowania
- Gleba sucha + wysokie parowanie (gorąco, jasno) ⇒ pełna moc pompy.
- Gleba wilgotna ⇒ pompa wyłączona.
- Sięganie po średni poziom podlewania w warunkach umiarkowanych lub przy ograniczonym parowaniu, aby uniknąć przelania.
- Reguły obejmują sytuacje chłodne i ciemne, gdzie podlewanie nie jest potrzebne.

## Scenariusze demonstracyjne
1. **Wysoki stres wodny**  
   Gleba 10 %, temperatura 32 °C, światło 9/10 ⇒ oczekiwane mocne podlewanie (wykresy włączone).
2. **Brak potrzeby podlewania**  
   Gleba 80 %, temperatura 15 °C, światło 1/10 ⇒ komenda bliska 0 % (bez wykresów).

W realnym zastosowaniu `demo_single_step()` może być uruchamiane w pętli sterującej co kilka sekund, analogicznie do systemów autopilota.


