import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create matrix
movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)

# Compute similarity
similarity = cosine_similarity(movie_matrix.T)

similarity_df = pd.DataFrame(similarity,
                             index=movie_matrix.columns,
                             columns=movie_matrix.columns)

# Function to recommend movies
def recommend_movies(movie_name):
    similar_scores = similarity_df[movie_name].sort_values(ascending=False)
    return list(similar_scores.iloc[1:6].index)

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    if movie_name in similarity_df.columns:
        recommendations = recommend_movies(movie_name)
        st.write("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)
    else:
        st.write("Movie not found in dataset.")