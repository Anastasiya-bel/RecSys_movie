import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(warm_users, movies_with_features):
    """
    This function calculates the cosine similarity between the movie
    descriptions, genres, actors, and directors of the last
    and second-to-last viewed movies for each user.
    """
    movies_with_features['combined_text'] = \
        movies_with_features['genres_name'].fillna('') + ' ' + \
        movies_with_features['actors'].fillna('') + ' ' + \
        movies_with_features['director'].fillna('') + ' ' + \
        movies_with_features['description'].fillna('')
    data = pd.merge(
        warm_users,
        movies_with_features,
        on='movie_id',
        how="left"
    )
    # Find the last and second-to-last viewed movies for each user
    last_and_second_last_viewed = \
        data.sort_values(by=['user_id', 'datetime']).groupby('user_id').tail(2)

    # Create TF-IDF vectorizer for the combined text column
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(
        movies_with_features['combined_text']
        )
    # Calculate cosine similarity for each user and their last viewed movie
    cosine_similarities_last = []
    for _, row in last_and_second_last_viewed.iterrows():
        movie_tfidf = tfidf.transform([row['combined_text']])
        similarities = cosine_similarity(movie_tfidf, tfidf_matrix)
        cosine_similarities_last.append(similarities[0][0])

    # Calculate cosine similarity for each user
    # and their second-to-last viewed movie
    cosine_similarities_second_last = []
    for _, row in last_and_second_last_viewed.iterrows():
        movie_tfidf = tfidf.transform([row['combined_text']])
        similarities = cosine_similarity(movie_tfidf, tfidf_matrix)
        cosine_similarities_second_last.append(similarities[0][1])

    # Add the cosine similarity columns to the original DataFrame
    last_and_second_last_viewed['cosine_similarity_last'] = \
        cosine_similarities_last
    last_and_second_last_viewed['cosine_similarity_second_last'] = \
        cosine_similarities_second_last
    last_and_second_last_viewed = last_and_second_last_viewed[[
        'user_id',
        'movie_id',
        'cosine_similarity_last',
        'cosine_similarity_second_last'
    ]]

    last_and_second_last_viewed.to_csv(
        'files/cosine_similarity.csv',
        index=False
    )

    return last_and_second_last_viewed
