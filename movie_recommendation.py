# Movie Recommendation System

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create user-movie matrix
movie_matrix = data.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)

# Fill missing values
movie_matrix = movie_matrix.fillna(0)

# Calculate similarity
similarity = cosine_similarity(movie_matrix.T)

# Convert to dataframe
similarity_df = pd.DataFrame(
    similarity,
    index=movie_matrix.columns,
    columns=movie_matrix.columns
)

# Recommendation function
def recommend_movies(movie_name):

    similar_scores = similarity_df[movie_name].sort_values(ascending=False)

    print("\nRecommended Movies:\n")

    for movie in similar_scores.iloc[1:6].index:
        print(movie)

# Test recommendation
recommend_movies("Toy Story (1995)")