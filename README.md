# Movie Recommendation System

## Overview
This repository contains a **Movie Recommendation System** that utilizes **TF-IDF Vectorization** and **Collaborative Filtering** to suggest movies based on user input. It is designed to find similar movies based on title similarity and user ratings.

## Features
- **Movie Title Search**: Uses TF-IDF vectorization to find the most similar movie titles.
- **Content-Based Filtering**: Recommends movies based on similarity of titles.
- **Collaborative Filtering**: Suggests movies based on user ratings and similar user preferences.
- **Popularity-Based Sorting**: Prioritizes popular movies to ensure better recommendations.

## Datasets Used
The system uses the following datasets:
1. **movies.csv** - Contains movie metadata (movieId, title, genres).
2. **ratings.csv** - Contains user ratings (userId, movieId, rating, timestamp).

### Downloading the Datasets
The datasets can be downloaded from the `dataset` directory available in this repository. Simply navigate to the `dataset` folder and extract the required CSV files. You can also manually place them in the project directory before running the script.

To download using Git:
```bash
git clone https://github.com/PadamArora/movie-rec-system.git
cd movie-rec-system/dataset
```
Ensure that the `movies.csv` and `ratings.csv` files are in the same directory as the script before execution.

## Installation
To run this project, follow these steps:

### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas scikit-learn numpy
```

### Clone the Repository
```bash
git clone https://github.com/PadamArora/movie-rec-system.git
cd movie-rec-system
```

### Running the System
1. **Run the Python script**
   ```bash
   python movie_recommendation.py
   ```
2. **Enter a movie title** when prompted.
3. **Get recommendations** based on content and user preferences.

## Code Explanation

### 1. Data Preprocessing
- The `movies.csv` dataset is loaded, and movie titles are cleaned to remove special characters.
- A new column `clean_title` is added to store cleaned titles.

### 2. Title Similarity Search
- Uses **TF-IDF Vectorization** to convert movie titles into numerical vectors.
- Computes **cosine similarity** to find the most relevant movie titles based on user input.

### 3. Sorting by Popularity
- Extracts **movieIds** from search results.
- Sorts these movies based on the number of user ratings to prioritize popular movies.

### 4. Collaborative Filtering
- Finds users who highly rated the selected movie (ratings > 4).
- Extracts movies rated highly by similar users.
- Filters movies that at least 10% of similar users have rated highly.
- Computes a **recommendation score** by comparing ratings from similar users with all users.
- Returns the **top 10 recommended movies** based on similarity and user preferences.

### 5. User Interaction
- The user enters a movie title.
- If an exact match isn't found, the program suggests the closest titles.
- Recommendations are displayed based on the **first valid movie with enough ratings**.

## Example Output
```bash
Enter a movie title: Avengers
Found movies:
Avengers, The (2012)
Avengers: Infinity War (2018)
Avengers: Age of Ultron (2015)
No recommendations available for some movies.

Recommended Movies:
1. Thor: The Dark World (2013)
2. Captain America: Civil War (2016)
3. Iron Man 3 (2013)
4. Ant-Man (2015)
...
```

## Enhancements & Future Work
- Improve search accuracy by incorporating **Levenshtein distance** for better title matching.
- Implement a **web-based UI** for easier interaction.
- Extend collaborative filtering using **matrix factorization** techniques.

## Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.

## License
This project is licensed under the **MIT License**.
