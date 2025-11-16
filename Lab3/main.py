"""
Silnik rekomendacji filmów/seriali oparty na filtrowaniu kolaboratywnym.

Opis problemu:
    Silnik rekomendacji generuje listę filmów/seriali, które mogą spodobować się użytkownikowi,
    oraz listę antyrekomendacji na podstawie ocen innych użytkowników.

Autor: Mateusz Andrzejak Szymon Anikiej

Instalacja zależności (w kolejności importów):
    pip install pandas
    pip install numpy
    pip install scipy
    pip install requests
    pip install python-dotenv

Instrukcja użycia:
    1. Umieść plik `data.csv` i `.env` w tym samym katalogu co skrypt.
    2. Utwórz plik `.env` z kluczem TMDB API: TMDB_API_KEY=twoj_klucz_tutaj
    3. Uruchom skrypt: `python main.py`.
    4. Wybierz użytkownika z listy.
    5. Otrzymasz rekomendacje i antyrekomendacje filmów wraz z wyjaśnieniem.

API:
    Skrypt używa TMDB API do pobierania informacji o filmach i serialach.
    Aby uzyskać klucz API: zarejestruj się na https://www.themoviedb.org/, 
    przejdź do Settings -> API i utwórz nowe konto API (typ: Developer).
"""

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
import requests
from dotenv import load_dotenv
import os
from urllib.parse import quote

load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

def load_and_process_data(file_path):
    """
    Wczytuje i przetwarza dane z pliku CSV.

    Args:
        file_path (str): Ścieżka do pliku CSV.

    Returns:
        pd.DataFrame: Przetworzone dane w formacie user, movie, rating.
    """
    processed_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            elements = line.strip().split(',')
            user = elements[0]
            for i in range(1, len(elements), 2):
                if i + 1 < len(elements):
                    movie = elements[i].strip()
                    try:
                        rating = int(elements[i + 1].strip())
                        processed_data.append({'user': user, 'movie': movie, 'rating': rating})
                    except ValueError:
                        continue
    return pd.DataFrame(processed_data)

def build_user_item_matrix(df):
    """
    Buduje macierz użytkownik-film.

    Args:
        df (pd.DataFrame): DataFrame z danymi.

    Returns:
        pd.DataFrame: Macierz użytkownik-film.
    """
    return df.pivot_table(index='user', columns='movie', values='rating').fillna(0)

def compute_similarity(user_item_matrix, method='pearson', k=50):
    """
    Oblicza macierz podobieństwa między użytkownikami.

    Args:
        user_item_matrix (pd.DataFrame): Macierz użytkownik-film.
        method (str): Metoda obliczania podobieństwa ('pearson' lub 'svd').
        k (int): Liczba składowych dla SVD.

    Returns:
        pd.DataFrame: Macierz podobieństwa.
    """
    if method == 'pearson':
        n_users = user_item_matrix.shape[0]
        similarity_matrix = pd.DataFrame(
            index=user_item_matrix.index,
            columns=user_item_matrix.index,
            data=0.0)

        for i in range(n_users):
            for j in range(n_users):
                user1 = user_item_matrix.iloc[i]
                user2 = user_item_matrix.iloc[j]
                common_movies = user1.index.intersection(user2.index)
                if len(common_movies) > 0:
                    corr, _ = pearsonr(user1[common_movies], user2[common_movies])
                    similarity_matrix.iloc[i, j] = corr if not pd.isna(corr) else 0.0

        return similarity_matrix

    elif method == 'svd':
        sparse_matrix = user_item_matrix.values
        user_ratings_mean = np.mean(sparse_matrix, axis=1)
        sparse_matrix_normalized = sparse_matrix - user_ratings_mean.reshape(-1, 1)

        U, sigma, Vt = svds(sparse_matrix_normalized, k=k)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

        return pd.DataFrame(
            predicted_ratings,
            index=user_item_matrix.index,
            columns=user_item_matrix.columns)

def get_recommendations(user, user_item_matrix, user_similarity_df, n=5, similarity_threshold=0.0):
    """
    Generuje rekomendacje filmów dla danego użytkownika wraz z wyjaśnieniem.

    Args:
        user (str): Nazwa użytkownika.
        user_item_matrix (pd.DataFrame): Macierz użytkownik-film.
        user_similarity_df (pd.DataFrame): Macierz podobieństwa między użytkownikami.
        n (int): Liczba rekomendacji.
        similarity_threshold (float): Próg podobieństwa.

    Returns:
        list: Lista krotek (film, przewidywana_ocena, wyjaśnienie).
    """
    if user not in user_item_matrix.index:
        return []

    user_similarities = user_similarity_df[user]
    relevant_users = user_similarities[user_similarities > similarity_threshold].index
    user_similarities = user_similarities[relevant_users]

    user_ratings = user_item_matrix.loc[user]
    unrated_movies = user_ratings[user_ratings == 0].index
    recommendations = []

    for movie in unrated_movies:
        movie_ratings = user_item_matrix.loc[relevant_users, movie]
        weights = user_similarities[relevant_users]
        rated_users = movie_ratings[movie_ratings != 0]

        if len(rated_users) > 0:
            relevant_weights = weights[rated_users.index]
            if relevant_weights.sum() > 0:
                predicted_rating = (rated_users * relevant_weights).sum() / relevant_weights.sum()
                explanation = [
                    (u, user_item_matrix.loc[u, movie], f"{weights[u]:.2f}")
                    for u in rated_users.index
                ]
                recommendations.append((movie, predicted_rating, explanation))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

def get_anti_recommendations(user, user_item_matrix, user_similarity_df, n=5, similarity_threshold=0.0):
    """
    Generuje antyrekomendacje filmów dla danego użytkownika wraz z wyjaśnieniem.

    Args:
        user (str): Nazwa użytkownika.
        user_item_matrix (pd.DataFrame): Macierz użytkownik-film.
        user_similarity_df (pd.DataFrame): Macierz podobieństwa między użytkownikami.
        n (int): Liczba antyrekomendacji.
        similarity_threshold (float): Próg podobieństwa.

    Returns:
        list: Lista krotek (film, przewidywana_ocena, wyjaśnienie).
    """
    if user not in user_item_matrix.index:
        return []

    user_similarities = user_similarity_df[user]
    relevant_users = user_similarities[user_similarities > similarity_threshold].index
    user_similarities = user_similarities[relevant_users]

    user_ratings = user_item_matrix.loc[user]
    unrated_movies = user_ratings[user_ratings == 0].index
    anti_recommendations = []

    for movie in unrated_movies:
        movie_ratings = user_item_matrix.loc[relevant_users, movie]
        weights = user_similarities[relevant_users]
        rated_users = movie_ratings[movie_ratings != 0]

        if len(rated_users) > 0:
            relevant_weights = weights[rated_users.index]
            if relevant_weights.sum() > 0:
                predicted_rating = (rated_users * relevant_weights).sum() / relevant_weights.sum()
                explanation = [
                    (u, user_item_matrix.loc[u, movie], f"{weights[u]:.2f}")
                    for u in rated_users.index
                ]
                anti_recommendations.append((movie, predicted_rating, explanation))

    anti_recommendations.sort(key=lambda x: x[1])
    return anti_recommendations[:n]

def get_top_movies(user_item_matrix, n=5):
    """
    Zwraca n najlepiej ocenionych filmów przez wszystkich użytkowników.

    Args:
        user_item_matrix (pd.DataFrame): Macierz użytkownik-film.
        n (int): Liczba filmów.

    Returns:
        list: Lista krotek (film, średnia_ocena).
    """
    avg_ratings = user_item_matrix.mean(axis=0)
    top_movies = avg_ratings.sort_values(ascending=False).head(n)
    return [(movie, avg_ratings[movie]) for movie in top_movies.index]

def get_worst_movies(user_item_matrix, n=5):
    """
    Zwraca n najgorzej ocenionych filmów przez wszystkich użytkowników.

    Args:
        user_item_matrix (pd.DataFrame): Macierz użytkownik-film.
        n (int): Liczba filmów.

    Returns:
        list: Lista krotek (film, średnia_ocena).
    """
    avg_ratings = user_item_matrix.mean(axis=0)
    worst_movies = avg_ratings.sort_values(ascending=True).head(n)
    return [(movie, avg_ratings[movie]) for movie in worst_movies.index]

def get_movie_info_tmdb(movie):
    """
    Pobiera informacje o filmie lub serialu z TMDB API.

    Args:
        movie (str): Tytuł filmu/serialu.

    Returns:
        dict: Informacje o filmie/serialu.
    """
    if not TMDB_API_KEY:
        return None

    try:
        # Najpierw szukaj filmu
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={quote(movie)}&language=pl"
        search_response = requests.get(search_url, timeout=5)
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            if search_data.get('results') and len(search_data['results']) > 0:
                # Weź pierwszy wynik
                movie_id = search_data['results'][0]['id']
                
                # Pobierz szczegóły
                details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=pl&append_to_response=credits"
                details_response = requests.get(details_url, timeout=5)
                
                if details_response.status_code == 200:
                    data = details_response.json()
                    directors = [crew['name'] for crew in data.get('credits', {}).get('crew', []) if crew.get('job') == 'Director']
                    director = directors[0] if directors else "Nieznany"
                    
                    genres = [g['name'] for g in data.get('genres', [])]
                    genre = ", ".join(genres) if genres else "Nieznany"
                    
                    return {
                        'title': data.get('title', movie),
                        'description': data.get('overview', "Brak opisu"),
                        'director': director,
                        'imdb_rating': f"{data.get('vote_average', 0):.1f}" if data.get('vote_average') else "Nieznany",
                        'genre': genre,
                        'year': data.get('release_date', '')[:4] if data.get('release_date') else "Nieznany"
                    }
        
        # Jeśli nie znaleziono filmu, szukaj serialu
        search_url_tv = f"https://api.themoviedb.org/3/search/tv?api_key={TMDB_API_KEY}&query={quote(movie)}&language=pl"
        search_response_tv = requests.get(search_url_tv, timeout=5)
        
        if search_response_tv.status_code == 200:
            search_data_tv = search_response_tv.json()
            if search_data_tv.get('results') and len(search_data_tv['results']) > 0:
                # Weź pierwszy wynik
                tv_id = search_data_tv['results'][0]['id']
                
                # Pobierz szczegóły
                details_url_tv = f"https://api.themoviedb.org/3/tv/{tv_id}?api_key={TMDB_API_KEY}&language=pl&append_to_response=credits"
                details_response_tv = requests.get(details_url_tv, timeout=5)
                
                if details_response_tv.status_code == 200:
                    data = details_response_tv.json()
                    creators = [creator['name'] for creator in data.get('created_by', [])]
                    director = creators[0] if creators else "Nieznany"
                    
                    genres = [g['name'] for g in data.get('genres', [])]
                    genre = ", ".join(genres) if genres else "Nieznany"
                    
                    return {
                        'title': data.get('name', movie),
                        'description': data.get('overview', "Brak opisu"),
                        'director': director,
                        'imdb_rating': f"{data.get('vote_average', 0):.1f}" if data.get('vote_average') else "Nieznany",
                        'genre': genre,
                        'year': data.get('first_air_date', '')[:4] if data.get('first_air_date') else "Nieznany"
                    }
    except Exception as e:
        pass
    
    return None

def get_movie_info(movie):
    """
    Pobiera informacje o filmie lub serialu z TMDB API.

    Args:
        movie (str): Tytuł filmu/serialu.

    Returns:
        dict: Informacje o filmie/serialu. Jeśli nie znaleziono, zwraca domyślne wartości.
    """
    info = get_movie_info_tmdb(movie)
    if info:
        return info
    
    # Jeśli API nie zwróciło informacji, zwróć domyślne wartości
    return {
        'title': movie,
        'description': "Nie znaleziono filmu",
        'director': "Nieznany",
        'imdb_rating': "Nieznany",
        'genre': "Nieznany",
        'year': "Nieznany"
    }

def main():
    file_path = 'data.csv'
    df = load_and_process_data(file_path)
    user_item_matrix = build_user_item_matrix(df)
    user_similarity_df = compute_similarity(user_item_matrix, method='pearson')

    users = df['user'].unique()
    print("Dostępni użytkownicy:")
    for i, user in enumerate(users, 1):
        print(f"{i}. {user}")

    choice = int(input("\nWybierz użytkownika (podaj numer): "))
    user = users[choice - 1]

    recommendations = get_recommendations(user, user_item_matrix, user_similarity_df, similarity_threshold=0.0)
    if not recommendations:
        print("\nBrak spersonalizowanych rekomendacji. Wyświetlam 5 najlepiej ocenionych filmów:")
        top_movies = get_top_movies(user_item_matrix)
        for movie, avg_rating in top_movies:
            info = get_movie_info(movie)
            print(f"- {info['title']} ({info['year']})")
            print(f"  Średnia ocena: {avg_rating:.2f}, IMDB: {info['imdb_rating']}")
            print(f"  Reżyser: {info['director']}")
            print(f"  Gatunek: {info['genre']}")
            print(f"  Opis: {info['description']}")
    else:
        print(f"\nRekomendacje dla użytkownika {user}:")
        for movie, rating, explanation in recommendations:
            info = get_movie_info(movie)
            print(f"- {info['title']} ({info['year']})")
            print(f"  Ocena użytkowników: {rating:.2f}, IMDB: {info['imdb_rating']}")
            print(f"  Reżyser: {info['director']}")
            print(f"  Gatunek: {info['genre']}")
            print(f"  Opis: {info['description']}")
            print("  Wyjaśnienie:")
            for u, r, sim in explanation:
                print(f"    {u} ocenił(a) na {r:.2f}, podobieństwo: {sim}")

    anti_recommendations = get_anti_recommendations(user, user_item_matrix, user_similarity_df, similarity_threshold=0.0)
    if not anti_recommendations:
        print("\nBrak spersonalizowanych antyrekomendacji. Wyświetlam 5 najgorzej ocenionych filmów:")
        worst_movies = get_worst_movies(user_item_matrix)
        for movie, avg_rating in worst_movies:
            info = get_movie_info(movie)
            print(f"- {info['title']} ({info['year']})")
            print(f"  Średnia ocena: {avg_rating:.2f}, IMDB: {info['imdb_rating']}")
            print(f"  Reżyser: {info['director']}")
            print(f"  Gatunek: {info['genre']}")
            print(f"  Opis: {info['description']}")
    else:
        print(f"\nAntyrekomendacje dla użytkownika {user}:")
        for movie, rating, explanation in anti_recommendations:
            info = get_movie_info(movie)
            print(f"- {info['title']} ({info['year']})")
            print(f"  Ocena użytkowników: {rating:.2f}, IMDB: {info['imdb_rating']}")
            print(f"  Reżyser: {info['director']}")
            print(f"  Gatunek: {info['genre']}")
            print(f"  Opis: {info['description']}")
            print("  Wyjaśnienie:")
            for u, r, sim in explanation:
                print(f"    {u} ocenił(a) na {r:.2f}, podobieństwo: {sim}")

if __name__ == "__main__":
    main()
