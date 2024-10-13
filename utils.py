import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
import altair as alt

# Load datasets with caching to prevent reloading multiple times
@st.cache_data
def load_data():
    """
    Load book, rating, and user datasets from CSV files.
    """
    try:
        books_df = pd.read_csv('Books.csv', engine='python')
        ratings_df = pd.read_csv('Ratings.csv', engine='python').sample(40)
        users_df = pd.read_csv('Users.csv', engine='python')
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.error(f"Error loading data: {e}")
        return None, None, None
    return books_df, ratings_df, users_df

def preprocess_data(books_df, ratings_df, users_df):
    """
    Merge datasets and create a user-book rating matrix.
    """
    try:
        user_ratings_df = ratings_df.merge(users_df, on='User-ID')
        book_user_ratings_df = books_df.merge(user_ratings_df, on='ISBN')
        book_user_ratings_df = book_user_ratings_df[['ISBN', 'Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']]
        book_user_ratings_df.reset_index(drop=True, inplace=True)

        # Map unique book IDs and create a user-book rating matrix
        unique_books_dict = {isbn: idx for idx, isbn in enumerate(book_user_ratings_df.ISBN.unique())}
        book_user_ratings_df['unique_book_id'] = book_user_ratings_df['ISBN'].map(unique_books_dict)
        user_book_matrix_df = book_user_ratings_df.pivot(index='User-ID', columns='unique_book_id', values='Book-Rating').fillna(0)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None

    return book_user_ratings_df, user_book_matrix_df.values

def compute_svd(user_book_matrix, num_factors=15):
    """
    Perform SVD on the user-book rating matrix.
    """
    try:
        U, sigma, Vt = svds(user_book_matrix, k=num_factors)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    except np.linalg.LinAlgError as e:
        st.error(f"SVD error: {e}")
        return None, None

    return predicted_ratings, Vt

def top_cosine_similarity(matrix, book_id, top_n=10):
    """
    Compute the top N cosine similarities for a book.
    """
    try:
        book_vector = matrix[book_id, :]
        magnitude = np.sqrt(np.einsum('ij, ij -> i', matrix, matrix))
        similarity_scores = np.dot(book_vector, matrix.T) / (magnitude[book_id] * magnitude)
        return np.argsort(-similarity_scores)[:top_n]
    except IndexError as e:
        st.error(f"Invalid book ID: {e}")
        return []

def similar_books(book_user_ratings_df, book_id, top_indexes):
    """
    Get titles of similar books based on cosine similarity.
    """
    recommendations = []
    try:
        original_book_title = book_user_ratings_df[book_user_ratings_df.unique_book_id == book_id]["Book-Title"].values[0]
        recommendations.append({'Book Title': original_book_title, 'Recommendation': 'Original Book'})
        for idx in top_indexes[1:]:
            recommended_title = book_user_ratings_df[book_user_ratings_df.unique_book_id == idx]['Book-Title'].values[0]
            recommendations.append({'Book Title': recommended_title, 'Recommendation': 'Similar Book'})
    except IndexError as e:
        st.error(f"Recommendation error: {e}")
    return recommendations

def visualize_user_book_matrix(matrix):
    """
    Plot heatmap of the user-book rating matrix (subset).
    """
    try:
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix[:10, :10], cmap='YlGnBu', annot=False)
        plt.title('User-Book Rating Matrix')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Visualization error: {e}")

def visualize_user_book_matrix_altair(matrix):
    """
    Visualize the user-book matrix using Altair.
    """
    try:
        df = pd.DataFrame(matrix[:10, :10], columns=[f'Book {i}' for i in range(10)])
        df['User'] = [f'User {i}' for i in range(10)]
        df = df.melt('User', var_name='Book', value_name='Rating')

        chart = alt.Chart(df).mark_rect().encode(
            x='Book:O', y='User:O', color='Rating:Q'
        ).properties(width=600, height=400, title='User-Book Matrix')

        st.altair_chart(chart)
    except Exception as e:
        st.error(f"Altair visualization error: {e}")