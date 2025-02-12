import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import difflib

# Load datasets
movies = pd.read_csv("movies.csv")  # Load movie dataset
ratings = pd.read_csv("ratings.csv")  # Load ratings dataset

# Function to clean movie titles by removing special characters
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title).lower()

# Apply the cleaning function to all movie titles
movies["clean_title"] = movies["title"].apply(clean_title)

# Vectorize movie titles using TF-IDF with unigrams and bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Function to search for movies based on title similarity
def search(title):
    title = clean_title(title)  # Clean the user input title
    query_vec = vectorizer.transform([title])  # Convert input title to vector
    similarity = cosine_similarity(query_vec, tfidf).flatten()  # Compute similarity scores
    indices = np.argsort(similarity)[-10:][::-1]  # Get top 10 similar movies
    results = movies.iloc[indices]

    movie_ids = results['movieId'].tolist()  # Extract movie IDs for further processing
    return results[["title"]], movie_ids  # Return matching movie titles and IDs

# Function to sort movie IDs by popularity (number of ratings)
def sort_movies_by_popularity(movie_ids):
    return sorted(movie_ids, key=lambda x: ratings[ratings["movieId"] == x]["userId"].nunique(), reverse=True)

# Function to find similar movies based on collaborative filtering
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    if len(similar_users) < 100:
        return None

    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.1]

    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# Streamlit App
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.sidebar.title("Movie Recommendation System")
user_input = st.sidebar.text_input("Enter a movie title:")
if st.sidebar.button("Recommend"):
    if user_input:
        search_results, movie_ids = search(user_input)
        flag = False

        if search_results.empty:
            possible_titles = difflib.get_close_matches(user_input, movies["title"].tolist(), n=5, cutoff=0.5)
            if possible_titles:
                st.sidebar.write("No exact match found. Did you mean one of these?")
                for title in possible_titles:
                    st.sidebar.write(f"- {title}")
            else:
                st.sidebar.write("No movie found. Try another title.")
        else:
            movie_ids = sort_movies_by_popularity(movie_ids)
            for movie_id in movie_ids:
                recommendations_df = find_similar_movies(movie_id)
                if recommendations_df is not None and not recommendations_df.empty:
                    flag = True
                    st.subheader("Recommended Movies:")
                    for _, row in recommendations_df.iterrows():
                        st.write(f"- {row['title']}")
                    break

        if not flag:
            st.write("No recommendations for this movie")
