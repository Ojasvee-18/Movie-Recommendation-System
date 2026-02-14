# Import libraries
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

class SVDRecommender:
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
        # Create Pivot Table
        # We need to ensure we cover all users/movies or handle missing ones.
        self.R_df = self.ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        
        # Determine matrix
        self.R = self.R_df.to_numpy()
        self.user_ratings_mean = np.mean(self.R, axis=1)
        self.R_demeaned = self.R - self.user_ratings_mean.reshape(-1, 1)
        
        # Compute SVD
        # k=50 as per original script
        k = 50
        # Check if we have enough dimensions
        min_dim = min(self.R_demeaned.shape)
        if k >= min_dim:
            k = min_dim - 1
            
        self.U, self.sigma, self.Vt = svds(self.R_demeaned, k=k)
        self.sigma = np.diag(self.sigma)
        
        # Pre-compute all predictions? 
        # For 1M dataset (6k users * 4k movies), 24M doubles ~ 200MB. Fine to store in memory.
        self.all_user_predicted_ratings = np.dot(np.dot(self.U, self.sigma), self.Vt) + self.user_ratings_mean.reshape(-1, 1)
        self.preds_df = pd.DataFrame(self.all_user_predicted_ratings, columns=self.R_df.columns, index=self.R_df.index)

    def predict(self, user_id, movie_ids):
        # Check if user exists in training data
        if user_id in self.preds_df.index:
            user_preds = self.preds_df.loc[user_id]
            # Get predictions for requested movies
            # Handle movies not in columns
            valid_movies = [m for m in movie_ids if m in self.preds_df.columns]
            return user_preds[valid_movies].values
        else:
            # Cold start: return mean rating or 0
            return np.zeros(len(movie_ids))

    def recommend(self, user_id, top_k=20):
        # New simplified recommend logic using the pre-computed matrix
        
        if user_id not in self.preds_df.index:
            # Return popular movies if user unknown
            popular = self.ratings.groupby('movie_id')['rating'].count().sort_values(ascending=False).head(top_k).index
            recs = self.movies[self.movies['movie_id'].isin(popular)]
            recs['prediction'] = 5.0 # Fake score
            return recs

        # Get sorted predictions
        sorted_user_predictions = self.preds_df.loc[user_id].sort_values(ascending=False)
        
        # Get existing ratings to exclude
        user_data = self.ratings[self.ratings.user_id == user_id]
        
        # Filter out already rated
        recommendations = (self.movies[~self.movies['movie_id'].isin(user_data['movie_id'])].
             merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                   left_on='movie_id', right_on='movie_id').
             rename(columns={user_id: 'prediction'}).
             sort_values('prediction', ascending=False).
             head(top_k))

        return recommendations

if __name__ == "__main__":
    recommender = SVDRecommender()
    print(recommender.recommend(1310))