import streamlit as st
import pandas as pd
from test_utils import load_data, preprocess_data, compute_svd, top_cosine_similarity, similar_books, visualize_user_book_matrix, visualize_user_book_matrix_altair, seaborn_plot, save_feedback, plot_recommendations

# Streamlit UI
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Recommendations'])

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

    # User input for book selection
    book_id = st.number_input('Enter the unique ID of a book to get recommendations:', min_value=0, max_value=len(book_user_rating['unique_id_book'].unique()) - 1)
    top_n = st.number_input('Number of recommendations to show:', min_value=1, max_value=10, value=3)

    # Display multiselect options for filtering
    authors = st.multiselect("Filter by Author", book_user_rating['Book-Author'].unique())
    years = st.multiselect("Filter by Year of Publication", book_user_rating['Year-Of-Publication'].unique())
    publishers = st.multiselect("Filter by Publisher", book_user_rating['Publisher'].unique())

    if st.button('Get Recommendations', key='recommend_button'):
        top_indexes = top_cosine_similarity(Vt.T[:, :50], book_id, top_n)
        recommendations = similar_books(book_user_rating, book_id, top_indexes, authors, years, publishers)
        
        # Display recommendations in a table
        st.write(pd.DataFrame(recommendations))
        
        # Show heatmap of the user-book matrix
        st.markdown('### User-Book Matrix Heatmap')
        st.write('Here’s a sample of the user-book rating matrix heatmap based on the current dataset:')
        
        # Visualize the matrix as a heatmap
        st.altair_chart(visualize_user_book_matrix_altair(user_book_matrix))

        # Show Seaborn plot
        st.markdown('### Distribution of Book Ratings')
        seaborn_fig = seaborn_plot(book_user_rating)
        st.pyplot(seaborn_fig)

        # Show bar chart for recommendations
        st.markdown('### Original Book and Similar Books')
        recommendation_fig = plot_recommendations(recommendations)
        st.pyplot(recommendation_fig)

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
    button[data-testid="recommend_button"] {
        background-color: yellow;
        color: black;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        cursor: pointer;
    }
    button:hover {
        background-color: black;
        color: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Feedback section
st.sidebar.title('Feedback')
feedback = st.sidebar.text_area('Your feedback:')
if st.sidebar.button('Submit'):
    save_feedback(feedback)
    st.sidebar.write('Thank you for your feedback!')
