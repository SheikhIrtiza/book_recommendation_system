import streamlit as st
import pandas as pd
from utils import load_data, preprocess_data, compute_svd, top_cosine_similarity, similar_books, seaborn_plot, save_feedback, get_book_image_url

# Streamlit UI
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Recommendations'], key='navigation_radio')

# Add a header image
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://th.bing.com/th/id/OIP.EnOy3-5IcV9Zi9yJRYdATgAAAA?rs=1&pid=ImgDetMain" alt="Header Image" style="width:50%; border-radius: 10px;">
    </div>
    """,
    unsafe_allow_html=True
)

if page == 'Home':
    st.title('Book Recommendation System')
    st.markdown(
        """
        <div class="floating-text">
            <h3>Welcome to the Book Recommendation System! Use the sidebar to navigate.</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == 'Recommendations':
    st.title('Book Recommendations')

    # Load data
    book_df, ratings_df, user_df = load_data()
    # Preprocess data
    book_user_rating, user_book_matrix = preprocess_data(book_df, ratings_df, user_df)
    # Compute SVD
    all_user_predicted_ratings, Vt = compute_svd(user_book_matrix)

    # User input for book selection by title
    book_title = st.selectbox('Select a book title:', book_user_rating['Book-Title'].unique(), key='book_title_select')

    # Fetch book image for the selected book
    book_image_url = get_book_image_url(book_df, book_title)

    # Display selected book image
    if book_image_url:
        st.image(book_image_url, caption=book_title, width=150)

    # Automatically show 5 recommendations (skipping the input and button)
    top_n = 5  # Default value of 5 recommendations

    # Find book ID based on the selected title
    book_id = book_user_rating[book_user_rating['Book-Title'] == book_title]['unique_id_book'].values[0]

    # Get recommendations
    top_indexes = top_cosine_similarity(Vt.T[:, :50], book_id, top_n)
    recommendations = similar_books(book_user_rating, book_id, top_indexes)

    # Display recommendations with book images in rows
    st.markdown('### Recommended Books:')
    num_books = len(recommendations)
    # Adjust row layout based on number of recommendations
    num_cols = 2 if num_books > 1 else 1  # 2 books per row
    cols = st.columns(num_cols)
    for i, rec in enumerate(recommendations):
        col = cols[i % num_cols]  # Distribute books across columns
        recommended_book_title = rec['Book Title']
        recommended_book_image_url = get_book_image_url(book_df, recommended_book_title)
        with col:
            if recommended_book_image_url:
                st.image(recommended_book_image_url, caption=recommended_book_title, width=150)
            else:
                st.write(recommended_book_title)

    # Show Seaborn plot
    st.markdown('### Distribution of Book Ratings')
    seaborn_fig = seaborn_plot(book_user_rating, recommendations)
    st.pyplot(seaborn_fig)

# Custom CSS for styling with yellow and black combination
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
        background-image: linear-gradient(45deg, yellow, black);
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
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Feedback section
st.sidebar.title('Feedback')
feedback = st.sidebar.text_area('Your feedback:', key='feedback_textarea')
if st.sidebar.button('Submit', key='feedback_submit'):
    save_feedback(feedback)
    st.sidebar.write('Thank you for your feedback!')
