"""
Projekt: FuzzyPlantWatering - rozmyty system sterowania podlewaniem roślin
Autorzy: Mateusz Andrzejak, Szymon Anikej
Data: 30.10.2025

Środowisko: Python 3.10+, biblioteki: numpy, scipy, scikit-fuzzy, matplotlib, networkx

Instrukcja przygotowania środowiska:
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
    pip install "numpy<2.0" "scipy<1.14" "scikit-fuzzy<1.14"
    pip install matplotlib
    pip install networkx

Opis problemu:
---------------
Chcemy zbudować autonomiczny system podlewania roślin z użyciem logiki rozmytej.
System ma podjąć decyzję, jak intensywnie włączyć pompę wody (0-100%), biorąc pod uwagę:

    1. Wilgotność gleby (soil_moisture, 0-100%)
    2. Temperaturę powietrza (air_temp, 0-40°C załóżmy typowe warunki szklarni)
    3. Poziom oświetlenia (light_level, 0-10 jednostek "nasłonecznienia")

Podejście:
    - Definiujemy zbiory rozmyte dla wejść i wyjścia.
    - Tworzymy reguły lingwistyczne typu:
        JEŚLI (gleba sucha) ORAZ (gorąco) ORAZ (jasno)
        TO (podlewaj dużo).
    - Uruchamiamy wnioskowanie i defuzyfikację → dostajemy sterowanie pompą.

Ten moduł:
    * Definiuje kontroler rozmyty sterujący podlewaniem.
    * Demonstruje działanie kontrolera na przykładowych danych
      (to jest nasza "symulacja w czasie rzeczywitym": tak jak autopilot
      dla lądownika / auta autonomicznego; mamy stan środowiska i decyzję).
    * Rysuje funkcje przynależności (kształty zbiorów rozmytych).

"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_controller():
    """
    Tworzy i zwraca kompletny rozmyty system sterowania podlewaniem.

    Zmienne wejściowe (Antecedents):
    --------------------------------
    soil_moisture : [0 .. 100] (%)
        'dry' / 'okay' / 'wet'

    air_temp : [0 .. 40] (°C)
        'low' / 'medium' / 'high'

    light_level : [0 .. 10] (umowne natężenie światła)
        'dark' / 'normal' / 'bright'

    Zmienna wyjściowa (Consequent):
    -------------------------------
    water_pump : [0 .. 100] (% mocy pompy / czasu otwarcia zaworu)
        'off' / 'medium' / 'full'

    Logika:
    -------
    - Jeśli gleba jest sucha i warunki powodują szybkie parowanie
      (gorąco, jasno), to pompa powinna podlewać mocno.
    - Jeśli gleba jest mokra, nie podlewamy.
    - Jeśli gleba jest średnio wilgotna, ale parowanie duże
      (duża temp + jasne światło), podlewamy średnio.
    - Jeśli chłodno i ciemno, podlewanie zwykle nie jest potrzebne.

    Zwraca:
    -------
    watering_simulator : ctrl.ControlSystemSimulation
        Obiekt gotowy do podawania danych wejściowych i wywołania .compute()
        aby otrzymać decyzję fuzzy nt. podlewania.
    watering_ctrl : ctrl.ControlSystem
        Surowy system kontrolny (lista reguł itd.).
    soil_moisture, air_temp, light_level, water_pump : Antecedent/Consequent
        Te obiekty są potrzebne do rysowania membershipów .view().
    """

    # --- Definicja zakresów zmiennych crisp ----------------------------
    soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil_moisture')
    air_temp = ctrl.Antecedent(np.arange(0, 41, 1), 'air_temp')
    light_level = ctrl.Antecedent(np.arange(0, 11, 1), 'light_level')

    water_pump = ctrl.Consequent(np.arange(0, 101, 1), 'water_pump')

    # --- Funkcje przynależności (membership functions) ----------------
    # Wilgotność gleby:
    # dry:    wysoka przynależność przy 0-30%
    # okay:   środek, ~20-60%
    # wet:    wysoka przy 50-100%
    soil_moisture['dry'] = fuzz.trimf(soil_moisture.universe, [0, 0, 30])
    soil_moisture['okay'] = fuzz.trimf(soil_moisture.universe, [20, 50, 60])
    soil_moisture['wet'] = fuzz.trimf(soil_moisture.universe, [50, 100, 100])

    # Temperatura powietrza:
    air_temp['low'] = fuzz.trimf(air_temp.universe, [0, 0, 18])
    air_temp['medium'] = fuzz.trimf(air_temp.universe, [15, 22, 28])
    air_temp['high'] = fuzz.trimf(air_temp.universe, [25, 40, 40])

    # Poziom światła:
    light_level['dark'] = fuzz.trimf(light_level.universe, [0, 0, 3])
    light_level['normal'] = fuzz.trimf(light_level.universe, [2, 5, 7])
    light_level['bright'] = fuzz.trimf(light_level.universe, [6, 10, 10])

    # Wyjście - pompa:
    water_pump['off'] = fuzz.trimf(water_pump.universe, [0, 0, 30])
    water_pump['medium'] = fuzz.trimf(water_pump.universe, [20, 50, 80])
    water_pump['full'] = fuzz.trimf(water_pump.universe, [70, 100, 100])

    # --- Reguły rozmyte ------------------------------------------------
    # 1. Jeśli gleba jest SUCHa i gorąco i jasno -> pompa FULL
    rule1 = ctrl.Rule(
        soil_moisture['dry'] & air_temp['high'] & light_level['bright'],
        water_pump['full']
    )

    # 2. Jeśli gleba jest SUCHa i (gorąco LUB normalne światło)
    #    -> też raczej podlewaj dużo
    rule2 = ctrl.Rule(
        soil_moisture['dry'] & (air_temp['high'] | light_level['normal']),
        water_pump['full']
    )

    # 3. Jeśli gleba jest OKAY i gorąco i jasno -> podlewaj MEDIUM
    rule3 = ctrl.Rule(
        soil_moisture['okay'] & air_temp['high'] & light_level['bright'],
        water_pump['medium']
    )

    # 4. Jeśli gleba jest OKAY i światło normalne -> podlej trochę (MEDIUM)
    rule4 = ctrl.Rule(
        soil_moisture['okay'] & light_level['normal'],
        water_pump['medium']
    )

    # 5. Jeśli gleba jest WILGOTNA -> pompa OFF
    rule5 = ctrl.Rule(
        soil_moisture['wet'],
        water_pump['off']
    )

    # 6. Jeśli zimno i ciemno -> OFF (nie ma potrzeby podlewać agresywnie)
    rule6 = ctrl.Rule(
        air_temp['low'] & light_level['dark'],
        water_pump['off']
    )

    # 7. Jeśli gleba jest SUCHa ale jest ciemno/chłodno -> MEDIUM
    #    (czyli nie lej pełną pompą, żeby nie przelać przy niskim parowaniu)
    rule7 = ctrl.Rule(
        soil_moisture['dry'] & (air_temp['low'] | light_level['dark']),
        water_pump['medium']
    )

    # Zbuduj system sterowania
    watering_ctrl = ctrl.ControlSystem(
        [rule1, rule2, rule3, rule4, rule5, rule6, rule7]
    )
    watering_simulator = ctrl.ControlSystemSimulation(watering_ctrl)

    return (
        watering_simulator,
        watering_ctrl,
        soil_moisture,
        air_temp,
        light_level,
        water_pump,
    )


def plot_memberships(soil_moisture, air_temp, light_level, water_pump):
    """
    Rysuje funkcje przynależności (zbiory rozmyte) dla wszystkich zmiennych.

    Ten wykres możesz wkleić do sprawozdania jako dokumentację projektu:
    - pokazuje jak rozumiemy "dry/okay/wet" itp.
    - nie wymaga żadnych metod .view(sim=...) ani dostępu do środka ControlSystem,
      więc jest kompatybilne ze starszymi wersjami scikit-fuzzy.

    Parametry
    ---------
    soil_moisture, air_temp, light_level : ctrl.Antecedent
    water_pump : ctrl.Consequent
    """
    soil_moisture.view()
    air_temp.view()
    light_level.view()
    water_pump.view()
    plt.show()


def demo_single_step(soil_val, temp_val, light_val, show_plots=True):
    """
    Wykonuje pojedynczy krok sterowania ("czas rzeczywisty"):
    Przyjmuje aktualne odczyty sensorów, przepuszcza je przez kontroler
    rozmyty i zwraca rekomendowaną intensywność podlewania.

    Parametry
    ---------
    soil_val : float
        Aktualna wilgotność gleby [% 0..100]
    temp_val : float
        Aktualna temperatura powietrza [°C 0..40]
    light_val : float
        Aktualne natężenie światła [0..10]
    show_plots : bool
        Czy wyświetlić funkcje przynależności (jako dokumentację)?

    Zwraca
    ------
    pump_command : float
        Procent mocy / otwarcia pompy (0..100). To jest sygnał,
        który w realnym układzie można przekazać np. do sterownika zaworu,
        przekaźnika pompy, PWM itd.

    Opis "real-time":
    -----------------
    To jest analogiczne do sterowania lądownikiem / autem autonomicznym:
    mamy stan środowiska w tej chwili (czujniki) i generujemy decyzję
    sterującą aktuator (pompę). W normalnym systemie robiłbyś to np.
    co 5 sekund w pętli sterującej.
    """

    (
        sim,
        _ctrl_system,
        soil_moisture,
        air_temp,
        light_level,
        water_pump,
    ) = build_fuzzy_controller()

    # Ustaw wejścia kontrolera
    sim.input['soil_moisture'] = soil_val
    sim.input['air_temp'] = temp_val
    sim.input['light_level'] = light_val

    # Oblicz wynik
    sim.compute()

    pump_command = sim.output['water_pump']
    print("=== FuzzyPlantWatering DECISION ===")
    print(f"Soil moisture: {soil_val:.1f}%")
    print(f"Air temp:      {temp_val:.1f}°C")
    print(f"Light level:   {light_val:.1f}/10")
    print(f"=> Pump command: {pump_command:.2f}% power")

    # W starych wersjach scikit-fuzzy:
    # - nie ma sim.view(...)
    # - nie ma ctrl_system.input / ctrl_system.output
    # Więc po prostu pokażemy membershipy (to nadaje się do raportu).
    if show_plots:
        plot_memberships(soil_moisture, air_temp, light_level, water_pump)

    return pump_command


if __name__ == "__main__":
    """
    Sekcja testowa / demonstracyjna.

    Scenariusz 1 (wysoki stres wodny):
    - Gleba bardzo sucha (10%)
    - Wysoka temperatura (32°C)
    - Bardzo jasno (9/10)
    Oczekujemy mocnego podlewania -> sterowanie pompy wysokie.

    Scenariusz 2 (brak potrzeby podlewania):
    - Gleba mokra (80%)
    - Chłodno (15°C)
    - Ciemno (1/10)
    Oczekujemy sterowania blisko zera.

    To jest nasz pokaz "sterowania w czasie rzeczywistym":
    w każdej chwili system mierzy stan i wydaje polecenie aktuatorowi.
    """

    # Scenariusz 1 - intensywne podlewanie (rysujemy wykresy)
    demo_single_step(
        soil_val=10.0,    # bardzo sucho
        temp_val=32.0,    # gorąco
        light_val=9.0,    # bardzo jasno
        show_plots=True
    )

    # Scenariusz 2 - podlewanie praktycznie zbędne (bez wykresów)
    demo_single_step(
        soil_val=80.0,    # mokro
        temp_val=15.0,    # chłodno
        light_val=1.0,    # prawie ciemno
        show_plots=False
    )
