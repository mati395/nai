"""
Projekt – System wykrywania dominującego koloru z kamery w czasie rzeczywistym
=============================================================================

Problem:
    Celem projektu jest stworzenie prostego systemu analizy obrazu
    działającego w czasie rzeczywistym, który na podstawie obrazu
    z kamery komputera rozpoznaje dominujący kolor w aktualnej klatce
    oraz wyznacza jego położenie.

Autorzy:
    Mateusz Andrzejak, Szymon Anikej

Instrukcja użycia:
    1. pip install opencv-python
       pip install numpy
    2. python main.py
    3. Pokaż kolory (czerwony, zielony, niebieski) przed kamerą
    4. Naciśnij 'q' aby zakończyć
"""

import cv2
import numpy as np


# Definicja zakresów kolorów w przestrzeni HSV
COLORS = {
    'czerwony': [
        (np.array([0, 50, 50]), np.array([10, 255, 255])),  # Dolny zakres czerwieni
        (np.array([170, 50, 50]), np.array([180, 255, 255]))  # Górny zakres czerwieni
    ],
    'zielony': [
        (np.array([40, 50, 50]), np.array([80, 255, 255]))
    ],
    'niebieski': [
        (np.array([100, 50, 50]), np.array([130, 255, 255]))
    ]
}

# Mapowanie kolorów do wartości BGR do rysowania
COLOR_BGR = {
    'czerwony': (0, 0, 255),
    'zielony': (0, 255, 0),
    'niebieski': (255, 0, 0)
}


def detect_color_mask(hsv_frame, color_name):
    """
    Tworzy maskę dla danego koloru w przestrzeni HSV.
    
    Args:
        hsv_frame: Obraz w przestrzeni HSV
        color_name: Nazwa koloru do wykrycia
        
    Returns:
        Maska binarna dla danego koloru
    """
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    
    if color_name in COLORS:
        for lower, upper in COLORS[color_name]:
            color_mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
    
    return mask


def find_color_contours(mask, min_area=100):
    """
    Znajduje kontury w masce i filtruje małe obszary.
    
    Args:
        mask: Maska binarna
        min_area: Minimalna powierzchnia konturu do uwzględnienia
        
    Returns:
        Lista konturów spełniających warunki
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return filtered_contours


def get_bounding_box_and_center(contours):
    """
    Wyznacza prostokąt ograniczający i punkt centralny dla konturów.
    
    Args:
        contours: Lista konturów
        
    Returns:
        Tuple (bounding_box, center) gdzie:
        - bounding_box: (x, y, width, height) lub None
        - center: (cx, cy) lub None
    """
    if not contours:
        return None, None
    
    # Łączenie wszystkich konturów
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Obliczenie centrum
    M = cv2.moments(all_points)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = x + w // 2
        cy = y + h // 2
    
    return (x, y, w, h), (cx, cy)


def detect_dominant_color(hsv_frame):
    """
    Wykrywa dominujący kolor w klatce.
    
    Args:
        hsv_frame: Obraz w przestrzeni HSV
        
    Returns:
        Tuple (color_name, mask, contours, bounding_box, center) lub None dla każdego elementu
    """
    color_pixel_counts = {}
    color_masks = {}
    color_contours = {}
    
    # Obliczenie liczby pikseli dla każdego koloru
    for color_name in COLORS.keys():
        mask = detect_color_mask(hsv_frame, color_name)
        contours = find_color_contours(mask)
        
        pixel_count = np.sum(mask > 0)
        
        color_pixel_counts[color_name] = pixel_count
        color_masks[color_name] = mask
        color_contours[color_name] = contours
    
    # Wykrycie koloru z największą liczbą pikseli
    if not color_pixel_counts or max(color_pixel_counts.values()) == 0:
        return None, None, None, None, None
    
    dominant_color = max(color_pixel_counts, key=color_pixel_counts.get)
    
    # Wyznaczenie bounding box i centrum dla dominującego koloru
    contours = color_contours[dominant_color]
    bounding_box, center = get_bounding_box_and_center(contours)
    
    return dominant_color, color_masks[dominant_color], contours, bounding_box, center


def draw_results(frame, color_name, bounding_box, center):
    """
    Rysuje wyniki na klatce obrazu.
    
    Args:
        frame: Obraz BGR do rysowania
        color_name: Nazwa wykrytego koloru
        bounding_box: (x, y, w, h) lub None
        center: (cx, cy) lub None
    """
    if color_name is None:
        return
    
    color_bgr = COLOR_BGR.get(color_name, (255, 255, 255))
    
    # Rysowanie bounding box
    if bounding_box:
        x, y, w, h = bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
    
    # Rysowanie centrum
    if center:
        cx, cy = center
        cv2.circle(frame, (cx, cy), 5, color_bgr, -1)
        cv2.circle(frame, (cx, cy), 10, color_bgr, 2)
    
    # Wyświetlenie informacji o kolorze
    text = f"Dominujacy kolor: {color_name.upper()}"
    if center:
        text += f" | Pozycja: ({center[0]}, {center[1]})"
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, color_bgr, 2)


def main():
    # Otwarcie kamery
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Błąd kamery")
        return
    
    print("System wykrywania dominującego koloru uruchomiony.")
    print("Naciśnij 'q' aby zakończyć program.")
    print("Wykrywane kolory: czerwony, zielony, niebieski")
    
    while True:
        # Pobranie klatki z kamery
        ret, frame = cap.read()
        
        if not ret:
            print("Błąd: Nie można odczytać klatki z kamery!")
            break
        
        # Konwersja BGR -> HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Wykrycie dominującego koloru
        color_name, mask, contours, bounding_box, center = detect_dominant_color(hsv_frame)
        
        # Rysowanie wyników na klatce
        draw_results(frame, color_name, bounding_box, center)
        
        # Wyświetlenie informacji diagnostycznej w konsoli
        if color_name:
            print(f"Wykryto: {color_name.upper()}", end='')
            if center:
                print(f" | Pozycja: ({center[0]}, {center[1]})", end='')
            if bounding_box:
                x, y, w, h = bounding_box
                print(f" | Bounding box: ({x}, {y}, {w}, {h})", end='')
            print()
        
        # Wyświetlenie obrazu
        cv2.imshow('Wykrywanie dominujacego koloru', frame)
        
        # Opcjonalne wyświetlenie maski (odkomentuj jeśli chcesz zobaczyć)
        # if mask is not None:
        #     cv2.imshow('Maska', mask)
        
        # Zakończenie programu po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Zwolnienie zasobów
    cap.release()
    cv2.destroyAllWindows()
    print("Program zakończony.")


if __name__ == "__main__":
    main()
