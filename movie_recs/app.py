from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from model_wrapper import ModelWrapper

app = Flask(__name__)

# Initialize model wrapper (loads/trains models on startup)
wrapper = ModelWrapper()

# Load movies for display
base_dir = os.path.dirname(os.path.abspath(__file__))
movies_df = pd.read_csv(os.path.join(base_dir, 'movies.csv'), sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])

@app.route('/')
def index():
    # Pick random movies for the user to rate
    random_movies = movies_df.sample(n=12).to_dict('records')
    return render_template('index.html', movies=random_movies)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ratings = data.get('ratings', {})
    model_name = data.get('model', 'svd')
    # Always start from the selected model's collaborative recommendations
    # for a fixed demo user so that changing the model clearly changes the
    # base ranking (DL vs SVD vs KNN).
    user_id = 2000
    base_recs = wrapper.get_recommendations(model_name, user_id, top_k=400).copy()

    # Parse user ratings (from the current page interaction)
    user_ratings = {}
    if ratings:
        try:
            user_ratings = {
                int(movie_id): float(score)
                for movie_id, score in ratings.items()
                if score is not None and str(score) != ''
            }
        except ValueError:
            user_ratings = {}

    if user_ratings:
        movies_meta = movies_df.set_index('movie_id')

        # Filter to only movies that exist in our metadata
        valid_ratings = {
            mid: r for mid, r in user_ratings.items()
            if mid in movies_meta.index
        }

        if valid_ratings:
            # Movies rated 3.5+ are treated as liked; if none, use all rated
            liked = {
                mid: r for mid, r in valid_ratings.items()
                if r >= 3.5
            }
            if not liked:
                liked = valid_ratings

            rated_ids = set(valid_ratings.keys())

            def parse_genres(genre_str):
                if pd.isna(genre_str):
                    return set()
                return set(str(genre_str).split('|'))

            liked_genres = {
                mid: parse_genres(movies_meta.loc[mid, 'genres'])
                for mid in liked.keys()
                if mid in movies_meta.index
            }

            def content_boost(row):
                movie_id = row['movie_id']
                # Do not recommend movies the user already rated
                if movie_id in rated_ids:
                    return -1e9

                cand_genres = parse_genres(row['genres'])
                if not cand_genres or not liked_genres:
                    return 0.0

                score = 0.0
                weight_sum = 0.0
                for mid, rating in liked.items():
                    g = liked_genres.get(mid, set())
                    if not g:
                        continue
                    inter = len(cand_genres & g)
                    union = len(c_genres | g) if (c_genres := cand_genres | g) else 0
                    if union == 0:
                        continue
                    jaccard = inter / union
                    # Higher ratings contribute more; shift so 3 is neutral
                    weight = max(rating - 2.5, 0.5)
                    score += jaccard * weight
                    weight_sum += weight

                if weight_sum == 0.0:
                    return 0.0
                return score / weight_sum

            # Apply a personalization boost on top of the model's prediction
            base_recs['boost'] = base_recs.apply(content_boost, axis=1)
            # Blend collaborative prediction and content-based boost
            lambda_weight = 2.0
            base_recs['prediction'] = base_recs['prediction'] + lambda_weight * base_recs['boost']

    # For the deep learning model, clamp scores into the 1–5 rating range
    if model_name == 'dl':
        base_recs['prediction'] = np.clip(base_recs['prediction'], 1.0, 5.0)

    # Sort by (possibly boosted and clamped) prediction and return top 20
    recommendations = base_recs.sort_values('prediction', ascending=False).head(20)
    recs_list = recommendations[['title', 'genres', 'prediction']].to_dict('records')
    return jsonify({'recommendations': recs_list})

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(wrapper.get_metrics())

@app.route('/more_movies', methods=['POST'])
def more_movies():
    """
    Return an extra batch of random movies for rating,
    excluding any movie_ids the client says are already shown.
    """
    data = request.json or {}
    existing_ids = set()
    for mid in data.get('existing_ids', []):
        try:
            existing_ids.add(int(mid))
        except (TypeError, ValueError):
            continue

    available = movies_df[~movies_df['movie_id'].isin(existing_ids)]
    if available.empty:
        # If we've somehow exhausted all movies, just return an empty list
        return jsonify({'movies': []})

    batch_size = min(12, len(available))
    new_movies = available.sample(n=batch_size).to_dict('records')
    return jsonify({'movies': new_movies})

if __name__ == '__main__':
    app.run(debug=True)
