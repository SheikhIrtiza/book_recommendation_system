**Book Recommendation System**

**Overview**

This project is a book recommendation system using Streamlit for the user interface and singular value decomposition (SVD) for generating recommendations based on book ratings.

**Features**

Navigation: Sidebar allows switching between the Home and Recommendations pages.

Book Recommendations: Get personalized book recommendations by entering a book ID and specifying the number of recommendations.

Visualizations: Displays a user-book matrix heatmap and rating distribution plots using Altair and Seaborn.

Feedback: Users can submit feedback via the sidebar.

**Setup Instructions**

*Clone the repository and install the required dependencies:*

1. git clone (repo url)

2. pip install -r requirements.txt

Place the datasets (Books.csv, Ratings.csv, Users.csv) in the root directory.

Run the Streamlit app:

3. **streamlit run main.py**

Files

main.py: Main Streamlit app that loads data, generates recommendations, and displays visualizations.

utils.py: Helper functions for loading data, computing SVD, and generating visualizations.

Data Sources

Books.csv: Contains book details.

Ratings.csv: Contains user-book ratings.

Users.csv: Contains user data.

Key Functions in utils.py

load_data(): Loads book, user, and rating data.

preprocess_data(): Prepares the user-book matrix.

compute_svd(): Applies SVD to the user-book matrix.

similar_books(): Finds books similar to the selected one.

visualize_user_book_matrix_altair(): Displays a heatmap using Altair.

seaborn_plot(): Plots the distribution of book ratings.

**Feedback**

User feedback is saved locally in feedback.txt