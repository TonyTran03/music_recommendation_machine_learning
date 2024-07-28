# Music Recommender System

This is a music recommender system built using `nltk` and `sklearn`. The system processes song metadata and lyrics to recommend similar songs based on user input.

## Features

- **Text Cleaning and Preprocessing**: Uses `nltk` for natural language processing tasks such as tokenization, stop word removal, and stemming.
- **TF-IDF Vectorization**: Utilizes `sklearn`'s `TfidfVectorizer` to transform text data into numerical features.
- **Cosine Similarity**: Employs `sklearn`'s `cosine_similarity` to measure the similarity between song vectors.
- **Recommendation System**: Recommends songs based on the similarity scores of the input song.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/music-recommender.git
    cd music-recommender
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset:
    - Ensure you have a CSV file with song metadata and lyrics. Example: `spotify_millsongs.csv`.
    - The CSV file should have columns such as `song_name`, `artist`, `lyrics`, etc.

2. Run the Jupyter notebook:

    ```bash
    jupyter notebook
    ```

3. Open `song_recommendation.ipynb` and follow the steps to preprocess the data, train the model, and get song recommendations.

## Example

Here's a quick example of how you can use the system in the notebook:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Load the dataset
df = pd.read_csv("spotify_millsongs.csv")
df.drop('link', axis=1, inplace=True)

# Sample the dataset if needed
df = df.sample(5000)

# Text Cleaning and Preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_lyrics'] = df['lyrics'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['cleaned_lyrics'])

# Function to get recommendations
def recommend_songs(song_name, num_recommendations=5):
    song_index = df[df['song_name'] == song_name].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[song_index], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]
    return df.iloc[similar_indices][['song_name', 'artist']]

# Get recommendations
print(recommend_songs("Song Name Here"))
