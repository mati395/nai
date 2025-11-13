"""
Silnik rekomendacji filmów/seriali oparty na filtrowaniu kolaboratywnym.

Opis problemu:
    Silnik rekomendacji generuje listę filmów/seriali, które mogą spodobować się użytkownikowi,
    oraz listę antyrekomendacji na podstawie ocen innych użytkowników.

Autor: Mateusz Andrzejak Szymon Anikiej

Instrukcja użycia:
    1. Umieść plik `data.csv` i `.env` w tym samym katalogu co skrypt.
    2. Uruchom skrypt: `python main.py`.
    3. Wybierz użytkownika z listy.
    4. Otrzymasz rekomendacje i antyrekomendacje filmów wraz z wyjaśnieniem.
"""

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
import requests
from dotenv import load_dotenv
import os

load_dotenv()
OMDB_API_KEY = os.getenv('OMDB_API_KEY')

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

def get_movie_info(movie):
    """
    Pobiera informacje o filmie z OMDb API.

    Args:
        movie (str): Tytuł filmu.

    Returns:
        dict: Informacje o filmie.
    """
    if not OMDB_API_KEY:
        return {
            'title': movie,
            'description': "Nie znaleziono filmu",
            'director': "Nie znaleziono filmu",
            'imdb_rating': "Nie znaleziono filmu",
            'genre': "Nie znaleziono filmu",
            'year': "Nie znaleziono filmu"
        }

    url = f"http://www.omdbapi.com/?t={movie}&apikey={OMDB_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data.get('Response') == 'True':
            return {
                'title': data.get('Title', movie),
                'description': data.get('Plot', "Brak opisu"),
                'director': data.get('Director', "Nieznany"),
                'imdb_rating': data.get('imdbRating', "Nieznany"),
                'genre': data.get('Genre', "Nieznany"),
                'year': data.get('Year', "Nieznany")
            }
        else:
            return {
                'title': movie,
                'description': "Nie znaleziono filmu",
                'director': "Nie znaleziono filmu",
                'imdb_rating': "Nie znaleziono filmu",
                'genre': "Nie znaleziono filmu",
                'year': "Nie znaleziono filmu"
            }
    else:
        return {
            'title': movie,
            'description': "Nie znaleziono filmu",
            'director': "Nie znaleziono filmu",
            'imdb_rating': "Nie znaleziono filmu",
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
