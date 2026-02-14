# Movie Recommendation System

A comprehensive movie recommendation engine built with Python and Flask, showcasing three distinct collaborative filtering approaches: SVD, KNN, and Deep Learning (Neural Collaborative Filtering).

## Features

- **Interactive Interface:** Rate movies directly in the web UI.
- **Multi-Model Support:** Choose between three powerful recommendation algorithms:
    - **SVD (Singular Value Decomposition):** Matrix factorization technique.
    - **KNN (K-Nearest Neighbors):** User-based collaborative filtering.
    - **Deep Learning:** Neural Collaborative Filtering using Keras/TensorFlow.
- **Real-time Recommendations:** Get instant movie suggestions based on your ratings.
- **Performance Metrics:** View model performance statistics (RMSE, MAE, Precision, Recall).

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, TensorFlow/Keras, Pandas, NumPy
- **Frontend:** HTML/CSS/JavaScript

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Navigate to the source code directory:**
   ```bash
   cd movie_recs
   ```

3. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   Ensure you are in the `movie_recs` directory where `app.py` resides.
   ```bash
   python app.py
   ```
   *Note: The first run might take a moment to train the models if pre-trained weights are not found.*

2. **Open your browser:**
   Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

3. **Get Recommendations:**
   - Rate a few movies on the home page.
   - Select a model (SVD, KNN, or Deep Learning) from the dropdown.
   - Click "Get Recommendations".

## Project Structure

- `app.py`: Main Flask application file.
- `model_wrapper.py`: Wrapper class to manage and initialize the recommendation models.
- `SVDmodel.py`: Implementation of the SVD recommender.
- `KNNModel.py`: Implementation of the KNN recommender.
- `DeepLearningModel.py`: Implementation of the Deep Learning recommender.
- `templates/`: HTML templates for the web interface.
- `static/`: Static assets (CSS, JS).
- `requirements.txt`: Python dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.
