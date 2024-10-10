import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
import altair as alt

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
    book_title = book_user_rating[book_user_rating.unique_id_book == book_id]["Book-Title"].values[0]
    recommendations.append({'Book Title': book_title, 'Recommendation': 'Original Book'})
    for id in top_indexes + 1:
        recommended_title = book_user_rating[book_user_rating.unique_id_book == id]['Book-Title'].values[0]
        recommendations.append({'Book Title': recommended_title, 'Recommendation': 'Similar Book'})
    return recommendations

def visualize_user_book_matrix(matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix[:10, :10], cmap='YlGnBu', annot=False)
    plt.title('Sample of User-Book Rating Matrix')
    st.pyplot(plt)

def visualize_user_book_matrix_altair(matrix):
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

    st.altair_chart(chart)


# Streamlit UI
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Recommendations', 'Matrix Heatmap'])

if page == 'Home':
    # Center the logo image at the top
    st.markdown("<div class='center-content'>", unsafe_allow_html=True)
    st.image('header_image.jpeg', width=150, caption="Book Recommendation System Logo")
    st.markdown("</div>", unsafe_allow_html=True)

    # Floating, multi-colored message
    st.markdown("""
        <div class="floating-text">
            <h3>Welcome to the Book Recommendation System! Use the sidebar to navigate.</h3>
        </div>
        """, unsafe_allow_html=True)

elif page == 'Recommendations':
    st.title('📖 Book Recommendations')
    # Load data
    book_df, ratings_df, user_df = load_data()

    # Preprocess data
    book_user_rating, user_book_matrix = preprocess_data(book_df, ratings_df, user_df)

    # Compute SVD
    all_user_predicted_ratings, Vt = compute_svd(user_book_matrix)

    # User input for book selection
    book_id = st.number_input('Enter the unique ID of a book to get recommendations:', min_value=0, max_value=len(book_user_rating['unique_id_book'].unique()) - 1)
    top_n = st.number_input('Number of recommendations to show:', min_value=1, max_value=10, value=3)

    if st.button('📚 Get Recommendations', key='recommend_button'):
        top_indexes = top_cosine_similarity(Vt.T[:, :50], book_id, top_n)
        recommendations = similar_books(book_user_rating, book_id, top_indexes)
        
        # Display recommendations in a table
        st.write(pd.DataFrame(recommendations))

elif page == 'Matrix Heatmap':
    st.title('🗺️ User-Book Matrix Heatmap')
    # Load data
    book_df, ratings_df, user_df = load_data()

    # Preprocess data
    book_user_rating, user_book_matrix = preprocess_data(book_df, ratings_df, user_df)

    if st.button('Show Heatmap', key='heatmap_button'):
        visualize_user_book_matrix_altair(user_book_matrix)

# Custom CSS for styling with colors and floating text
st.markdown(
    """
    <style>
    .main {
        background-color: #f9fbfd;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #eff2f7;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .floating-text {
        animation: float 3s ease-in-out infinite;
        background-image: linear-gradient(45deg, #ff4b4b, #ffb74d, #64b5f6);
        -webkit-background-clip: text;
        color: transparent;
        font-size: 24px;
        text-align: center;
        margin-top: 20px;
    }
    @keyframes float {
        0% {
            transform: translatey(0px);
        }
        50% {
            transform: translatey(-10px);
        }
        100% {
            transform: translatey(0px);
        }
    }
    h1 {
        color: #3a7bd5;
    }
    button[data-testid="recommend_button"] {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        cursor: pointer;
    }
    button[data-testid="heatmap_button"] {
        background-color: #008CBA;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Feedback section
st.sidebar.title('Feedback')
feedback = st.sidebar.text_area('Your feedback:')
if st.sidebar.button('Submit'):
    st.sidebar.write('Thank you for your feedback!')
