import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- 1. Data Loading --------------------
def load_data():
    """
    Load the book, ratings, and user datasets from user-uploaded files.
    """
    book_file = st.file_uploader("Upload Books CSV", type="csv")
    ratings_file = st.file_uploader("Upload Ratings CSV", type="csv")
    users_file = st.file_uploader("Upload Users CSV", type="csv")

    if book_file and ratings_file and users_file:
        book_df = pd.read_csv(book_file, engine='python')
        ratings_df = pd.read_csv(ratings_file, engine='python').sample(40000)
        user_df = pd.read_csv(users_file, engine='python')
        return book_df, ratings_df, user_df
    else:
        return None, None, None


# -------------------- 2. Data Preprocessing --------------------
def preprocess_data(book_df, ratings_df, user_df):
    """
    Merge datasets, create a pivot table, and add unique IDs for books.
    """
    user_rating_df = ratings_df.merge(user_df, left_on='User-ID', right_on='User-ID')
    book_user_rating = book_df.merge(user_rating_df, left_on='ISBN', right_on='ISBN')
    book_user_rating = book_user_rating[['ISBN', 'Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']]
    book_user_rating.reset_index(drop=True, inplace=True)

    unique_books_dict = {isbn: i for i, isbn in enumerate(book_user_rating.ISBN.unique())}
    book_user_rating['unique_id_book'] = book_user_rating['ISBN'].map(unique_books_dict)

    user_book_matrix_df = book_user_rating.pivot(index='User-ID', columns='unique_id_book', values='Book-Rating').fillna(0)

    return book_user_rating, user_book_matrix_df


# -------------------- 3. Matrix Factorization --------------------
def perform_matrix_factorization(user_book_matrix_df, num_factors=15):
    """
    Perform Singular Value Decomposition (SVD) on the user-item matrix.
    """
    user_book_matrix = user_book_matrix_df.values
    U, sigma, Vt = svds(user_book_matrix, k=num_factors)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    return U, sigma, Vt, predicted_ratings


# -------------------- 4. Recommendations --------------------
def top_cosine_similarity(data, book_id, top_n=10):
    """
    Find top N similar books based on cosine similarity.
    """
    index = book_id
    book_vector = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(book_vector, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]


def similar_books(book_user_rating, book_id, top_indexes):
    """
    Return book recommendations as a list.
    """
    recommendations = []
    book_title = book_user_rating[book_user_rating.unique_id_book == book_id]['Book-Title'].values[0]
    recommendations.append(f"Recommendations for {book_title}:")
    
    for idx in top_indexes + 1:
        recommended_book = book_user_rating[book_user_rating.unique_id_book == idx]['Book-Title'].values[0]
        recommendations.append(recommended_book)

    return recommendations


# -------------------- 5. Visualization --------------------
def visualize_user_book_matrix(matrix):
    """
    Plot a heatmap of the user-book matrix.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix[:10, :10], cmap='YlGnBu', annot=False)
    plt.title('Sample of User-Book Rating Matrix')
    st.pyplot(plt)


# -------------------- Main Function for Streamlit --------------------
def main():
    st.title("Book Recommendation System")

    # Load Data
    book_df, ratings_df, user_df = load_data()

    if book_df is not None and ratings_df is not None and user_df is not None:
        st.write("Datasets loaded successfully!")

        # Preprocess Data
        book_user_rating, user_book_matrix_df = preprocess_data(book_df, ratings_df, user_df)

        # Perform Matrix Factorization
        U, sigma, Vt, predicted_ratings = perform_matrix_factorization(user_book_matrix_df)

        # Select a book for recommendations
        book_titles = book_user_rating['Book-Title'].unique()
        selected_book_title = st.selectbox("Select a book to get recommendations:", book_titles)

        # Get book_id corresponding to selected book
        book_id = book_user_rating[book_user_rating['Book-Title'] == selected_book_title]['unique_id_book'].values[0]

        # Set parameters for recommendations
        k = 50  # Number of latent factors to slice
        top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

        # Get top N similar books
        sliced_Vt = Vt.T[:, :k]
        top_indexes = top_cosine_similarity(sliced_Vt, book_id, top_n)

        # Show recommendations
        recommendations = similar_books(book_user_rating, book_id, top_indexes)
        st.write("\n".join(recommendations))

        # Visualize the user-book matrix
        if st.checkbox("Show User-Book Matrix Heatmap"):
            visualize_user_book_matrix(user_book_matrix_df)
    else:
        st.write("Please upload all required CSV files.")


# Run the app
if __name__ == "__main__":
    main()
