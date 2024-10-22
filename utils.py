import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets with the 'python' engine
def load_data():
    book_df = pd.read_csv('Books.csv', engine='python')
    ratings_df = pd.read_csv('Ratings.csv', engine='python').sample(4000) # 40000 
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
    for id in top_indexes:
        if id != book_id:  # Skip the selected book
            recommended_title = book_user_rating[book_user_rating['unique_id_book'] == id]['Book-Title'].values[0]
            recommendations.append({'Book Title': recommended_title, 'Recommendation': 'Similar Book'})
    return recommendations

def seaborn_plot(book_user_rating, recommendations):
    # Filter book_user_rating for only the recommended books
    recommended_titles = [rec['Book Title'] for rec in recommendations]
    recommended_books_ratings = book_user_rating[book_user_rating['Book-Title'].isin(recommended_titles)]

    # Create a new column to identify recommended books
    recommended_books_ratings['Recommended Book'] = recommended_books_ratings['Book-Title']

    # Create the Seaborn plot
    plt.figure(figsize=(12, 6))  # Adjust size for more space
    sns.countplot(data=recommended_books_ratings, x='Recommended Book', hue='Book-Rating', palette='viridis')

    # Add title and labels
    plt.title('Distribution of Ratings for Recommended Books')
    plt.xlabel('Recommended Book')
    plt.ylabel('Count of Ratings')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')  # Rotate and align right to avoid jumbled text
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')
    return plt

def save_feedback(feedback):
    with open('feedback.txt', 'a') as f:
        f.write(feedback + '\n')

def get_book_image_url(book_df, book_title):
    url = book_df.loc[book_df['Book-Title'] == book_title, 'Image-URL-L'].values[0]
    return url if url else None




