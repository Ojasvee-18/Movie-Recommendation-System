# Import libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class KNNRecommender:
    def __init__(self, ratings_path='ratings.csv', movies_path='movies.csv'):
        # Resolve paths relative to this file
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ratings_path = os.path.join(base_dir, ratings_path)
        self.movies_path = os.path.join(base_dir, movies_path)

        self._load_data()
        self._train_model()

    def _load_data(self):
        self.ratings = pd.read_csv(self.ratings_path, sep='\t', encoding='latin-1', 
                              usecols=['user_id', 'movie_id', 'rating'])
        self.movies = pd.read_csv(self.movies_path, sep='\t', encoding='latin-1', 
                             usecols=['movie_id', 'title', 'genres'])
        
    def _train_model(self):
        # Create Pivot Matrix (Item-based)
        # Rows = Movies, Cols = Users
        # Fill with 0
        movie_user_pivot = self.ratings.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
        self.movie_user_matrix = csr_matrix(movie_user_pivot.values)
        self.movie_ids = list(movie_user_pivot.index)
        self.movie_mapper = {movie_id: i for i, movie_id in enumerate(self.movie_ids)}
        self.movie_inv_mapper = {i: movie_id for i, movie_id in enumerate(self.movie_ids)}
        
        # Train KNN
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        self.knn.fit(self.movie_user_matrix)
        
    def recommend(self, user_id, top_k=20):
        # Simple Item-based recommendation for a user
        # 1. Find user's top rated movies
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        if user_ratings.empty:
             # Random if new
             return self.movies.sample(top_k).assign(prediction=0)
             
        # Get max rated movie by user
        top_rated_movie_id = user_ratings.sort_values('rating', ascending=False).iloc[0]['movie_id']
        
        # Check if movie in our matrix
        if top_rated_movie_id not in self.movie_mapper:
            return self.movies.sample(top_k).assign(prediction=0)
            
        # Find neighbors of this movie
        movie_idx = self.movie_mapper[top_rated_movie_id]
        distances, indices = self.knn.kneighbors(self.movie_user_matrix[movie_idx], n_neighbors=top_k+1)
        
        # Indices are indices in matrix, map back to movie_ids
        rec_indices = indices.flatten()[1:] # Skip self
        rec_distances = distances.flatten()[1:]
        
        rec_movie_ids = [self.movie_inv_mapper[idx] for idx in rec_indices]
        
        # Create result DF
        recs = self.movies[self.movies['movie_id'].isin(rec_movie_ids)].copy()
        
        # Add "score" (1 - distance)
        # We need to map back correctly. Simpler to just create a dict
        score_map = {mid: (1-dist) for mid, dist in zip(rec_movie_ids, rec_distances)}
        recs['prediction'] = recs['movie_id'].map(score_map)
        
        return recs.sort_values('prediction', ascending=False)

if __name__ == "__main__":
    recommender = KNNRecommender()
    print(recommender.recommend(1310))
