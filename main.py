import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from train_model import train_lgbm
from popular_recs import PopularRecommender
from build_features import process_data
from generate_candidates import generate_and_save_candidates
from cosine_similarity import calculate_cosine_similarity


if __name__ == "__main__":
    logs = pd.read_csv('train/logs.csv')
    genres = pd.read_csv('train/genres.csv')
    movies = pd.read_csv('train/movies.csv')
    staff = pd.read_csv('train/staff.csv')
    countries = pd.read_csv('train/countries.csv')
    logs_with_features, movies_with_features, user_stats = process_data(
        logs, movies, staff, genres
    )
    logs_with_features = pd.read_csv('files/logs_processed.csv')
    movies_with_features = pd.read_csv('files/movies_processed.csv')
    user_stats = pd.read_csv("files/user_stats.csv")
    logs_with_features["datetime"] = pd.to_datetime(
        logs_with_features["datetime"],
        utc=True
    )
    candidates = generate_and_save_candidates(logs)
    candidates = pd.read_csv("files/reco_SASRec.csv")
    candidates = candidates[["user_id", "movie_id", "rank_SASRec"]]
    df_user_gr = logs_with_features.groupby('user_id')\
        .movie_id.nunique().reset_index()
    warm_users = logs_with_features[
        logs_with_features['user_id'].isin(
            df_user_gr[df_user_gr['movie_id'] >= 3]['user_id'].unique()
        )
    ].copy()
    cosine_similarity = calculate_cosine_similarity(
        warm_users,
        movies_with_features
    )
    cosine_similarity = pd.read_csv("files/cosine_similarity.csv")
    model = train_lgbm(
        candidates,
        logs_with_features,
        movies_with_features,
        cosine_similarity,
        user_stats
       )
    item_col = [
        'movie_id', 'watch_ts_quantile_95_diff', 'watch_ts_median_diff',
        'watch_ts_std', 'watched_in_all_time', 'name', 'year',
        'description', 'countries', 'release_novelty', 'actors',
        'director', 'genres_name', 'year_publication', 'month_publication',
        'release_day_of_week', 'genres_min', 'genres_max', 'genres_med',
        'countries_max', 'popularity_rank', "avg_popularity_rank_genre",
        'total_movies_genre'
    ]
    drop_col = ['user_id', 'movie_id', 'description']
    cat_col = ["year",
               "genres_name",
               "countries",
               "name",
               "actors",
               "director",
               "year_publication",
               "month_publication",
               "release_day_of_week"]
    user_stats_columns = [
        "user_id",
        "avg_watched_pct",
        "total_watched",
        "average_duration",
        "max_recency",
        "average_date_weight",
        "weekend_activity"
    ]
    candidates.dropna(subset=["movie_id"], axis=0, inplace=True)
    full_train = (
        candidates
        .merge(movies_with_features[item_col], on=["movie_id"], how="left")
        .merge(user_stats[user_stats_columns], on=["user_id"], how="left")
    )
    cosine_similarity_subset = cosine_similarity[[
        'user_id',
        'movie_id',
        'cosine_similarity_last',
        'cosine_similarity_second_last'
    ]]
    full_train = pd.merge(
        full_train,
        cosine_similarity_subset,
        on=['user_id', 'movie_id'],
        how='left'
    )
    label_encoder = LabelEncoder()
    for column in cat_col:
        full_train[column] = label_encoder.fit_transform(full_train[column])
    # fillna with the most frequent value
    full_train = full_train.fillna(full_train.mode().iloc[0])
    full_train.reset_index(drop=True, inplace=True)

    # Получение предсказаний для "теплых" пользователей
    y_pred_all = model.predict(full_train.drop(drop_col, axis=1))
    candidates.reset_index(drop=True, inplace=True)
    predictions_df = pd.DataFrame({"lgbm_pred": y_pred_all})
    predictions_df.reset_index(drop=True, inplace=True)

    # Concatenate candidates with predictions_df
    candidates = pd.concat([candidates, predictions_df], axis=1)
    candidates = candidates[["user_id", "movie_id", "lgbm_pred"]]
    candidates.drop_duplicates(
        subset=["user_id", "movie_id"],
        keep='first',
        inplace=True
    )
    candidates = candidates.sort_values(
        by=["user_id", "lgbm_pred"], ascending=[True, False]
    )
    candidates["rank"] = candidates.groupby("user_id").cumcount() + 1
    candidates = candidates[candidates["rank"] <= 20].drop("lgbm_pred", axis=1)
    candidates["movie_id"] = (
        candidates["movie_id"]
        .astype(float)
        .astype(pd.Int64Dtype())
    )
    boost_recs = candidates.groupby("user_id")["movie_id"].apply(list)
    boost_recs = pd.DataFrame(boost_recs)
    boost_recs.reset_index(inplace=True)

    # Making predictions for cold users with Popular Recommender
    idx_for_popular = list(
        set(logs["user_id"].unique()).difference(
            set(boost_recs["user_id"].unique())
        )
    )
    pop_model = PopularRecommender(
        days=30,
        dt_column="datetime",
        with_filter=True
    )
    pop_model.fit(logs)
    recs_popular = pop_model.recommend_with_filter(
        logs, idx_for_popular, top_K=20
    )
    all_recs = pd.concat([boost_recs, recs_popular], axis=0)
    output_directory = os.path.join("output")
    filename = "result.csv"
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, filename)
    all_recs.to_csv(output_path, index=False, header=False)
