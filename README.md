# **Movie Recommendation System**  

## **Overview**  
This repository contains a **Movie Recommendation System** that utilizes **TF-IDF Vectorization** and **Collaborative Filtering** to suggest movies based on user input. It helps find similar movies based on title similarity and user ratings, ensuring personalized recommendations.  

## **Features**  
- **Movie Title Search**: Uses **TF-IDF vectorization** to find similar movie titles.  
- **Content-Based Filtering**: Recommends movies based on title similarity.  
- **Collaborative Filtering**: Suggests movies based on user ratings and similar user preferences.  
- **Popularity-Based Sorting**: Prioritizes frequently rated and highly rated movies for better recommendations.  

## **Datasets Used**  
The system requires the following datasets:  
- **`movies.csv`** - Contains movie metadata (`movieId`, `title`, `genres`).  
- **`ratings.csv`** - Contains user ratings (`userId`, `movieId`, `rating`, `timestamp`).  

### **Downloading the Datasets**  
The datasets are available in the `datasets` directory within this repository. Simply navigate to the `datasets` folder and extract the required CSV files. Alternatively, manually place them in the project directory before running the script.  

To download using Git:  
```bash
git clone https://github.com/PadamArora/MovieRecommendationSystem.git
cd movie-rec-system/datasets
```  
Ensure that **`movies.csv`** and **`ratings.csv`** are present in the same directory as the script before execution.  

## **Installation & Setup**  

### **Prerequisites**  
Follow these steps to set up and run the system:  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/PadamArora/MovieRecommendationSystem.git
   ```  

2. **Download the datasets** (if not already included in the repository).  

3. **Install dependencies**  
   ```bash
   pip install pandas scikit-learn numpy streamlit
   ```  

4. **Run the application**  
   ```bash
   streamlit run main.py
   ```  

5. **Enter a movie title** when prompted and receive recommendations.  

## **How It Works**  

### **1. Data Preprocessing**  
- Loads the **`movies.csv`** dataset and cleans movie titles by removing special characters.  
- Creates a **`clean_title`** column to store the processed titles.  

### **2. Title Similarity Search**  
- Uses **TF-IDF Vectorization** to convert movie titles into numerical vectors.  
- Computes **cosine similarity** to find the most relevant movie titles based on user input.  

### **3. Popularity-Based Sorting**  
- Extracts **`movieIds`** from search results.  
- Ranks movies based on the number of user ratings to prioritize well-rated films.  

### **4. Collaborative Filtering**  
- Identifies users who rated a selected movie highly (**ratings > 4**).  
- Extracts other movies rated highly by those users.  
- Filters movies that at least **10% of similar users** have rated highly.  
- Computes a **recommendation score** and returns the **top 10 recommendations** based on similarity and user preferences.  

### **5. User Interaction**  
- The user enters a **movie title**.  
- If an **exact match** isn’t found, the system suggests the closest titles.  
- Recommendations are displayed based on the **first valid movie with sufficient ratings**.  

## **Example Output**  
```bash
Enter a movie title: Avengers  

Recommended Movies:  
1. The Avengers (2012)  
2. Thor: The Dark World (2013)  
3. Avengers: Age of Ultron (2015)  
4. Iron Man 3 (2013)  
5. Captain America: The First Avenger (2011)  
6. Thor (2011)  
7. Captain America: The Winter Soldier (2014)  
8. Captain America: Civil War (2016)  
9. Ant-Man (2015)  
10. Iron Man 2 (2010)  
```  

## **Future Enhancements**  
- Improve search accuracy using **Levenshtein distance** for better title matching.  
- Develop a **web-based UI** for a more user-friendly experience.  
- Enhance collaborative filtering with **matrix factorization** techniques.  

## **Contributing**  
Contributions are welcome! Feel free to **fork the repository** and submit a **pull request**.  

## **License**  
This project is licensed under the **MIT License**.

