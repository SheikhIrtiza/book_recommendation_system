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

    # Create a form to prevent automatic refresh
    with st.form(key='book_selection_form'):
        # Store the initially selected book in session state to avoid overwriting
        if 'selected_book' not in st.session_state:
            st.session_state['selected_book'] = ''
        
        book_title = st.selectbox(
            'Select a book title:',
            book_user_rating['Book-Title'].unique(),
            key='book_title_select'
        )

        submit_button = st.form_submit_button(label='Get Recommendations')

        # Update the session state only when the form is submitted
        if submit_button:
            st.session_state['selected_book'] = book_title

    # Now, use st.session_state['selected_book'] to generate recommendations
    if st.session_state['selected_book']:
        book_title = st.session_state['selected_book']

        # Fetch book image for the selected book
        book_image_url = get_book_image_url(book_df, book_title)

        # Display selected book image
        if book_image_url:
            st.image(book_image_url, caption=book_title, width=150)

        # Find book ID based on the selected title
        book_id = book_user_rating[book_user_rating['Book-Title'] == book_title]['unique_id_book'].values[0]

        # Set number of recommendations (excluding the selected book)
        top_n = 6  # You want 5 recommendations + the selected book, so top_n = 6

        # Get recommendations using cosine similarity, excluding the selected book
        top_indexes = top_cosine_similarity(Vt.T[:, :50], book_id, top_n)

        # Remove the selected book from the similarity results if it's there
        top_indexes = [idx for idx in top_indexes if idx != book_id][:top_n - 1]

        # Get similar books based on the cosine similarity
        recommendations = similar_books(book_user_rating, book_id, top_indexes)

        # Ensure the selected book is added as the first recommendation
        selected_book = {'Book Title': book_title, 'Recommendation': 'Selected Book'}
        recommendations.insert(0, selected_book)  # Insert at the beginning of the list

        # Display recommendations with book images in rows
        st.markdown('### Recommended Books:')
        num_books = len(recommendations)
        num_cols = 3 if num_books > 1 else 1  # 2 books per row
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