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
    # Identify users who rated the given movie highly (rating > 4)
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()

    # If there aren't enough similar users, return no recommendations
    if len(similar_users) < 100:
        return None

    # Find other movies rated highly by these users
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]

    # Compute recommendation percentage for each movie
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    # Filter movies that at least 10% of similar users rated highly
    similar_user_recs = similar_user_recs[similar_user_recs > 0.1]

    # Compare with all users' preferences
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    # Merge the two dataframes to compute a recommendation score
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    # Sort movies by recommendation score in descending order
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    # Return top 10 recommended movies with their score, title, and genres
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


# Main execution block
if __name__ == "__main__":
    user_input = input("Enter a movie title: ")  # Get movie title from user
    search_results, movie_ids = search(user_input)  # Search for matching movies
    flag = False

    if search_results.empty:
        # Suggest closest matches if no direct match is found
        possible_titles = difflib.get_close_matches(user_input, movies["title"].tolist(), n=5, cutoff=0.5)
        if possible_titles:
            print("No exact match found. Did you mean one of these?")
            for title in possible_titles:
                print(title)
        else:
            print("No movie found. Try another title.")
    else:
        # Sort movies by popularity before finding recommendations
        movie_ids = sort_movies_by_popularity(movie_ids)

        for movie_id in movie_ids:
            recommendations_df = find_similar_movies(movie_id)  # Get similar movies

            if recommendations_df is not None and not recommendations_df.empty:
                flag = True
                print("\nRecommended Movies:")
                for title in recommendations_df["title"]:
                    print(title)
                break  # Stop after finding the first set of recommendations

    if flag == False:
        print("No recommendations for this movie")