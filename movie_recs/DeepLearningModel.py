# Import libraries
import math
import numpy as np
import pandas as pd
import os

# Import TensorFlow/Keras libraries (Keras 3 is standalone)
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Import custom CF Model Architecture (Ensure CFModel.py is also updated to TF2)
from CFModel import CFModel

class DeepLearningRecommender:
    def __init__(self, ratings_path='ratings.csv', movies_path='movies.csv', users_path='users.csv', weights_path='weights.keras'):
        # Resolve paths relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ratings_path = os.path.join(base_dir, ratings_path)
        self.movies_path = os.path.join(base_dir, movies_path)
        self.users_path = os.path.join(base_dir, users_path)
        self.weights_path = os.path.join(base_dir, weights_path)
        
        self.K_FACTORS = 100 # The number of dimensional embeddings for movies and users
        self.RNG_SEED = 42

        self._load_data()
        self._build_model()

    def _load_data(self):
        # Reading ratings file
        self.ratings = pd.read_csv(self.ratings_path, sep='\t', encoding='latin-1', 
                              usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
        self.max_userid = self.ratings['user_id'].drop_duplicates().max()
        self.max_movieid = self.ratings['movie_id'].drop_duplicates().max()

        # Reading users file
        self.users = pd.read_csv(self.users_path, sep='\t', encoding='latin-1', 
                            usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

        # Reading movies file
        self.movies = pd.read_csv(self.movies_path, sep='\t', encoding='latin-1', 
                             usecols=['movie_id', 'title', 'genres'])

    def _build_model(self):
        # Define model
        self.model = CFModel(self.max_userid, self.max_movieid, self.K_FACTORS)
        
        # Compile the model
        self.model.compile(loss='mse', optimizer='adamax')

        # Check if weights file already exists
        if os.path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}...")
            # "Wake up" the model with dummy data so we can load weights
            dummy_input = [np.array([0]), np.array([0])]
            self.model(dummy_input)
            self.model.load_weights(self.weights_path)
            self.is_trained = True
        else:
            print("No weights found. Model needs training.")
            self.is_trained = False

    def train(self, epochs=30):
        # Create training set
        shuffled_ratings = self.ratings.sample(frac=1.0, random_state=self.RNG_SEED)
        
        Users = shuffled_ratings['user_emb_id'].values
        Movies = shuffled_ratings['movie_emb_id'].values
        Ratings = shuffled_ratings['rating'].values

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2), 
            ModelCheckpoint(filepath=self.weights_path, monitor='val_loss', save_best_only=True)
        ]

        history = self.model.fit(
            [Users, Movies], 
            Ratings, 
            epochs=epochs, 
            validation_split=0.1, 
            verbose=2, 
            callbacks=callbacks
        )
        self.is_trained = True
        return history

    def predict(self, user_id, movie_ids):
        if not self.is_trained:
            raise Exception("Model is not trained.")
        
        # user_id is 1-based, model expects 0-based
        # movie_ids is list of 1-based ids, model expects 0-based
        
        user_ids_vec = np.array([user_id - 1] * len(movie_ids))
        movie_ids_vec = np.array(movie_ids) - 1
        
        predictions = self.model.predict([user_ids_vec, movie_ids_vec]).flatten()
        return predictions

    def recommend(self, user_id, top_k=20):
        # Get the user's previously rated movies
        user_rated_movies = self.ratings[self.ratings['user_id'] == user_id]['movie_id'].values
        
        # Find movies not rated by user
        all_movie_ids = self.movies['movie_id'].values
        unseen_movies = np.setdiff1d(all_movie_ids, user_rated_movies)
        
        # Predict
        preds = self.predict(user_id, unseen_movies)
        
        # Create dataframe
        recs = pd.DataFrame({'movie_id': unseen_movies, 'prediction': preds})
        recs = recs.merge(self.movies, on='movie_id')
        recs = recs.sort_values('prediction', ascending=False).head(top_k)
        
        return recs

# Example usage (for testing)
if __name__ == "__main__":
    recommender = DeepLearningRecommender()
    if not recommender.is_trained:
         recommender.train(epochs=5)
    
    print(recommender.recommend(2000))