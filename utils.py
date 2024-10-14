# utils.py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

# Load the datasets with the 'python' engine
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
    book_title = book_user_rating[book_user_rating.unique_id_book == book_id]["Book-Title"].values[0]
    recommendations.append({'Book Title': book_title, 'Recommendation': 'Original Book'})
    for id in top_indexes + 1:
        recommended_title = book_user_rating[book_user_rating.unique_id_book == id]['Book-Title'].values[0]
        recommendations.append({'Book Title': recommended_title, 'Recommendation': 'Similar Book'})
    return recommendations

def visualize_user_book_matrix(matrix):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix[:10, :10], cmap='YlGnBu', annot=False)
    plt.title('Sample of User-Book Rating Matrix')
    return plt

def visualize_user_book_matrix_altair(matrix):
    import altair as alt
    df = pd.DataFrame(matrix[:10, :10], columns=[f'Book {i}' for i in range(10)])
    df['User'] = [f'User {i}' for i in range(10)]
    df = df.melt('User', var_name='Book', value_name='Rating')

    chart = alt.Chart(df).mark_rect().encode(
        x='Book:O',
        y='User:O',
        color='Rating:Q'
    ).properties(
        width=600,
        height=400,
        title='Sample of User-Book Rating Matrix'
    )

    return chart
