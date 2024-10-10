# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds

# Load the datasets with the 'python' engine
@st.cache_data
def load_data():
    book_df = pd.read_csv('Books.csv', engine='python')
    ratings_df = pd.read_csv('Ratings.csv', engine='python').sample(40)
    user_df = pd.read_csv('Users.csv', engine='python')
    return book_df, ratings_df, user_df

def preprocess_data(book_df, ratings_df, user_df):
    user_rating_df = ratings_df.merge(user_df, on='User-ID', how='inner')
    book_user_rating = book_df.merge(user_rating_df, on='ISBN', how='inner')
    book_user_rating = book_user_rating[['ISBN', 'Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']]
    book_user_rating.reset_index(drop=True, inplace=True)
    
    unique_books_dict = {isbn: i for i, isbn in enumerate(book_user_rating.ISBN.unique())}
    book_user_rating['unique_id_book'] = book_user_rating['ISBN'].map(unique_books_dict)
    
    user_book_matrix_df = book_user_rating.pivot(index='User-ID', columns='unique_id_book', values='Book-Rating').fillna(0)
    
    return book_user_rating, user_book_matrix_df.values

def compute_svd(user_book_matrix):
    NUMBER_OF_FACTORS_MF = 15
    U, sigma, Vt = svds(user_book_matrix, k=NUMBER_OF_FACTORS_MF)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    return all_user_predicted_ratings, Vt

def top_cosine_similarity(data, book_id, top_n=10):
    index = book_id 
    book_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(book_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

def similar_books(book_user_rating, book_id, top_indexes):
    recommendations = []
    recommendations.append(f'Recommendations for {book_user_rating[book_user_rating.unique_id_book == book_id]["Book-Title"].values[0]}: \n')
    for id in top_indexes + 1:
        recommendations.append(book_user_rating[book_user_rating.unique_id_book == id]['Book-Title'].values[0])
    return recommendations

def visualize_user_book_matrix(matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix[:10, :10], cmap='YlGnBu', annot=False)
    plt.title('Sample of User-Book Rating Matrix')
    st.pyplot(plt)

# Streamlit UI
st.title('Book Recommendation System')

# Load data
book_df, ratings_df, user_df = load_data()

# Preprocess data
book_user_rating, user_book_matrix = preprocess_data(book_df, ratings_df, user_df)

# Compute SVD
all_user_predicted_ratings, Vt = compute_svd(user_book_matrix)

# User input for book selection
book_id = st.number_input('Enter the unique ID of a book to get recommendations:', min_value=0, max_value=len(book_user_rating['unique_id_book'].unique()) - 1)
top_n = st.number_input('Number of recommendations to show:', min_value=1, max_value=10, value=3)

if st.button('Get Recommendations'):
    top_indexes = top_cosine_similarity(Vt.T[:, :50], book_id, top_n)
    recommendations = similar_books(book_user_rating, book_id, top_indexes)
    
    st.write(recommendations)

# Visualize user-book matrix
if st.button('Show User-Book Matrix Heatmap'):
    visualize_user_book_matrix(user_book_matrix)
