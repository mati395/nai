"""
Projekt – Bot do gry Galaxian
=============================

Bot grający w grę Galaxian (Atari) używając środowiska Gymnasium.
Bot używa prostej heurystyki opartej na analizie obrazu do wykrywania wrogów.

Instrukcja instalacji:
    pip install gymnasium[atari] opencv-python numpy
    pip install "gymnasium[accept-rom-license]"
    AutoROM --accept-license

Autorzy:
    Mateusz Andrzejak, Szymon Anikej
"""

import gymnasium as gym
import numpy as np
import cv2
import ale_py


gym.register_envs(ale_py)



def simple_heuristic_action(observation):
    """
    Prosta heurystyka wyboru akcji.
    Analizuje górną część ekranu (gdzie są wrogowie) i wybiera akcję.
    
    Args:
        observation: Obraz z gry (210, 160, 3)
        
    Returns:
        Akcja do wykonania (0-5)
    """
    # Analiza górnej części ekranu (gdzie są wrogowie)
    upper_half = observation[:100, :]
    gray = cv2.cvtColor(upper_half, cv2.COLOR_RGB2GRAY)
    
    # Proste wykrywanie aktywności w górnej części
    activity = np.mean(gray > 100)  # Procent jasnych pikseli
    
    if activity > 0.1:  # Są wrogowie
        # Oblicz średnią pozycję X jasnych pikseli
        bright_pixels = np.where(gray > 150)
        if len(bright_pixels[1]) > 0:
            enemy_x = np.mean(bright_pixels[1])
            screen_center = observation.shape[1] / 2
            
            # Poruszaj się w kierunku wrogów i strzelaj
            if enemy_x < screen_center - 20:
                return 5  # LEFTFIRE
            elif enemy_x > screen_center + 20:
                return 4  # RIGHTFIRE
            else:
                return 1  # FIRE
    
    # Brak wyraźnych wrogów - strzelaj i poruszaj się
    return np.random.choice([1, 4, 5])  # FIRE, RIGHTFIRE, LEFTFIRE


def random_action():
    """
    Losowa akcja - użyteczna jako fallback.
    
    Returns:
        Losowa akcja (0-5)
    """
    return np.random.randint(0, 6)


def play_episode(env, max_steps=10000, render=True):
    """
    Gra jeden epizod gry.
    
    Args:
        env: Środowisko Gymnasium
        max_steps: Maksymalna liczba kroków
        render: Czy wyświetlać grę
        
    Returns:
        Całkowity wynik (reward sum)
    """
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < max_steps:
        if render:
            env.render()
        
        # Wybór akcji - mieszanka heurystyki i losowości
        if np.random.random() < 0.7:  # 70% heurystyka, 30% losowo
            action = simple_heuristic_action(observation)
        else:
            action = random_action()
        
        # Wykonanie akcji
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    return total_reward


def main():
    """
    Główna funkcja - uruchamia bota do gry w Galaxian.
    """
    env_names_to_try = [
        "ALE/Galaxian-v5",
        "Galaxian-v5",
        "ALE/Galaxian-v0",
        "Galaxian-v0"
    ]
    
    env = None
    for env_name in env_names_to_try:
        try:
            env = gym.make(env_name, render_mode="human")
            break
        except Exception:
            continue
    
    print("Rozpoczynamy gre")
    print("Nacisnij Ctrl+C aby zakonczyc\n")
    
    try:
        # Graj kilka epizodów
        num_episodes = 3
        total_scores = []
        
        for episode in range(num_episodes):
            print(f"Epizod {episode + 1}/{num_episodes}")
            score = play_episode(env, max_steps=5000, render=True)
            total_scores.append(score)
            print(f"Wynik: {score}\n")
        
        print("Podsumowanie:")
        print(f"Sredni wynik: {np.mean(total_scores):.2f}")
        print(f"Najlepszy wynik: {max(total_scores)}")
        print(f"Najgorszy wynik: {min(total_scores)}")
        
    except KeyboardInterrupt:
        print("\nPrzerwano przez uzytkownika")
    finally:
        env.close()


if __name__ == "__main__":
    main()
