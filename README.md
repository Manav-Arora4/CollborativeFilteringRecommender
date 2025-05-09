# 🎌 Anime Recommendation System using Collaborative Filtering (NumPy)
This project implements a basic collaborative filtering recommendation system using NumPy. It uses the MyAnimeList dataset from Kaggle to recommend anime titles to users based on their rating behavior and that of similar users.

📌 Project Overview
The goal of this project is to:

Demonstrate user-based collaborative filtering using matrix operations in NumPy.

Build a simple recommender system without using any external machine learning libraries.

Work with real-world anime rating data from MyAnimeList.

📂 Dataset
Source: Kaggle - MyAnimeList Dataset (https://www.kaggle.com/datasets/azathoth42/myanimelist)
Files used: 

anime.csv: Metadata about each anime (e.g. name, genre, type, rating).

rating.csv: User ratings for different anime titles.

These datasets provide the necessary user-item interaction matrix for building a recommender system.

🚀 Features
Pure NumPy implementation (no scikit-learn or surprise libraries).

Matrix-based similarity computation.

Predicts user ratings based on K-nearest neighbor users.

Recommendation logic based on predicted ratings.

🛠️ Installation
Clone the repository or download the notebook.

Install the dependencies:
```pip install numpy pandas```

📦 Files Needed
Make sure your project folder contains:

anime_cf_model.pkl – Trained model

anime.csv – Anime metadata from Kaggle

🧪 Usage
Here's how to load the model and generate recommendations for a specific or random user:
```
import pickle
import pandas as pd
import numpy as np
import random

# Load anime metadata
anime_df = pd.read_csv('anime.csv')

# Load the model components
with open("anime_cf_model.pkl", "rb") as f:
    model_data = pickle.load(f)

X = model_data['X']
W = model_data['W']
b = model_data['b']
user_to_index = model_data['user_to_index']
anime_to_index = model_data['anime_to_index']

# Reverse anime index to map back to IDs
inv_anime_index = {v: k for k, v in anime_to_index.items()}

# Function to recommend anime
def recommend_anime_for_user(username, X, W, b, user_to_index, anime_to_index, top_n=10):
    if username not in user_to_index:
        raise ValueError("User not found in training data.")

    user_idx = user_to_index[username]
    user_pred = X @ W[user_idx].T + b[0, user_idx]
    top_anime_indices = np.argsort(-user_pred)[:top_n]
    recommended_ids = [inv_anime_index[i] for i in top_anime_indices]

    recommended_titles = anime_df[anime_df['anime_id'].isin(recommended_ids)][['anime_id', 'title']]
    recommended_titles['rank'] = recommended_titles['anime_id'].apply(lambda x: recommended_ids.index(x))
    return recommended_titles.sort_values('rank')[['title']]

# Example: Generate recommendations for a random user
random_user = random.choice(list(user_to_index.keys()))
print(f"Generating recommendations for: {random_user}")
recommendations = recommend_anime_for_user(random_user, X, W, b, user_to_index, anime_to_index)
print(recommendations.to_string(index=False))
```

🚀 Features
User-based collaborative filtering using matrix factorization

Pure NumPy implementation (no external ML libraries)

Fast recommendation using a saved .pkl model

Works with real anime ratings data

🙏 Acknowledgments
Dataset by azathoth42

Built for assignment and learning collaborative filtering from scratch
