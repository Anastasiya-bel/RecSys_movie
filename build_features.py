import pandas as pd
import numpy as np
from collections import Counter
import ast
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download("punkt")
nltk.download("stopwords")


def remove_punct(text):
    """
    Remove punctuation from the given text.

    Parameters:
    - text (str): The input text containing punctuation.

    Returns:
    str: Text with punctuation removed.
    """
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def preprocess_text(text):
    if isinstance(text, str):  # Проверяем, является ли 'text' строкой
        tokens = word_tokenize(text, language="russian")
        # Удаление пунктуации, приведение к нижн. регистру, удаление стоп-слов
        tokens = [remove_punct(token) for token in tokens]
        tokens = [
            token.lower()
            for token in tokens
            if token.lower() not in stopwords.words("russian")
        ]
        return " ".join(tokens)
    else:
        return ""  # Возвращаем пустую строку для нестроковых значений


# Преобразование строковых данных в список id в колонке 'staff'
def str_to_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


class AveragePopularityRank:
    def __init__(self, k, actions):
        self.k = k
        cnt = Counter()
        for action in actions["movie_id"]:
            cnt[action] += 1

        self.pop_rank = {}
        rank = 0
        for movie, cnt in cnt.most_common():
            rank += 1
            self.pop_rank[movie] = rank

    def __call__(self, movie_id):
        return self.pop_rank.get(movie_id, 0)


def add_item_watches_stats(logs, item_stats):
    """
    Computes item watches stats for particular interactions date split
    and adds them to item_stats dataframe
    """
    logs["datetime"] = pd.to_datetime(logs["datetime"])
    keep = item_stats.columns
    max_date = logs["datetime"].max()
    cols = list(range(7))
    for col in cols:
        watches = logs[
            logs["datetime"] == max_date - pd.Timedelta(days=6 - col)
        ]
        item_stats = item_stats.join(
            watches.groupby("movie_id")["user_id"].count(), lsuffix=col
        )
    item_stats.fillna(0, inplace=True)

    item_stats["watch_ts_quantile_95"] = 0
    item_stats["watch_ts_median"] = 0
    item_stats["watch_ts_std"] = 0
    for movie_id in item_stats.index:
        watches = logs[logs["movie_id"] == movie_id]
        day_of_year = watches["datetime"].apply(lambda x: x.dayofyear)\
            .astype(np.int64)
        item_stats.loc[movie_id, "watch_ts_quantile_95"] = (
            day_of_year.quantile(q=0.95, interpolation="nearest")
        )
        item_stats.loc[movie_id, "watch_ts_median"] = day_of_year.quantile(
            q=0.5, interpolation="nearest"
        )
        item_stats.loc[movie_id, "watch_ts_std"] = day_of_year.std()
    item_stats["watch_ts_quantile_95_diff"] = (
        max_date.dayofyear - item_stats["watch_ts_quantile_95"]
    )
    item_stats["watch_ts_median_diff"] = (
        max_date.dayofyear - item_stats["watch_ts_median"]
    )
    watched_all_time = logs.groupby("movie_id")["user_id"].count()
    watched_all_time.name = "watched_in_all_time"
    item_stats = item_stats.join(watched_all_time, on="movie_id", how="left")
    item_stats.fillna(0, inplace=True)
    added_cols = [
        "watch_ts_quantile_95_diff",
        "watch_ts_median_diff",
        "watch_ts_std",
        "watched_in_all_time",
    ]
    return item_stats[list(keep) + added_cols]


def process_data(logs, movies, staff, genres):
    """
    Process input data and generate processed data,
    saving results in the 'files' directory.

    Parameters:
    logs (DataFrame):
        -DataFrame containing user logs with columns 'datetime', 'duration'.
    movies (DataFrame):
        -DataFrame containing movie information with columns:
            -'id', 'name', 'year', 'description', 'staff',
            -'date_publication', 'genres', 'countries'.
    staff (DataFrame):
        -DataFrame containing information with columns 'id', 'name', 'role'.

    Returns:
    Tuple of DataFrames: (logs_processed, movies_processed)

    - logs_processed (DataFrame):
        -Processed user logs with additional columns:
             -'watched_pct', 'weekday', 'recency', 'date_weight'.
    - movies_processed (DataFrame):
        -Processed movie information with additional columns:
            -'release_novelty', 'actors', 'director', 'genres_min',
            -'genres_max', 'genres_med', 'countries_max', 'popularity_rank',
            -and merged statistics from user logs.
    """
    output = "files"
    os.makedirs(output, exist_ok=True)
    logs["datetime"] = pd.to_datetime(logs["datetime"])
    movies = movies.rename(columns={"id": "movie_id"})
    movies["year"] = pd.to_datetime(movies["year"]).dt.year

    # Release novelty
    movies["release_novelty"] = (
        pd.cut(
            movies["year"],
            bins=[-np.inf, 1980, 1990, 2000, 2010, 2020, np.inf],
            labels=False,
            right=False,
        )
        + 1
    )
    # Заполнение пустых значений в 'descriptions'
    movies["description"] = movies.apply(
        lambda row: f"{row['name']} {row['year']}"
        if pd.isnull(row["description"])
        else row["description"],
        axis=1,
    )

    # Применение предварительной обработки к столбцу "description"
    movies["description"] = movies["description"].apply(preprocess_text)
    movies["name"] = movies["name"].str.lower()
    movies["staff"] = movies["staff"].apply(str_to_list)

    # Функция для получения актеров из списка id
    def get_actors(id_list):
        actors = staff[
            (staff["id"].isin(id_list)) &
            (staff["role"] == "actor")
        ]["name"]
        return ", ".join(actors)

    # Функция для получения режиссера из списка id
    def get_director(id_list):
        director = staff[
            (staff["id"].isin(id_list)) &
            (staff["role"] == "director")
            ]["name"].values
        return ", ".join(director) if len(director) > 0 else None

    # Добавление колонок 'actors' и 'director' к датафрейму 'movies'
    movies["actors"] = movies["staff"].apply(get_actors)
    movies["director"] = movies["staff"].apply(get_director)

    # Заполнение пропущенных значений в 'director' и 'actors'
    movies["director"] = movies["director"].fillna("Unknown")
    movies["director"] = movies["director"].str.lower()
    movies["actors"] = movies["actors"].fillna("Unknown")
    movies["actors"] = movies["actors"].str.lower()
    movies["actors"] = movies["actors"].str.replace(r'["\'\[\],]', '', regex=True)
    movies = movies.drop("staff", axis=1)

    # Create a dictionary mapping genre IDs to genre names
    genre_id_to_name = dict(zip(genres["id"], genres["name"]))

    def replace_genre_ids(genre_ids):
        return [
            genre_id_to_name[int(id_)]
            for id_ in ast.literal_eval(genre_ids)
            if int(id_) in genre_id_to_name
        ]

    def clean_and_lower(genre_list):
        cleaned_genres = [
            genre.lower().translate(str.maketrans("", "", string.punctuation))
            for genre in genre_list
        ]
        return cleaned_genres

    # Replace genre IDs with genre names in the 'genres' column
    movies["genres_name"] = movies["genres"].apply(replace_genre_ids)
    movies["genres_name"] = movies["genres_name"].apply(clean_and_lower)
    movies["genres_name"] = movies["genres_name"].apply(lambda x: ", ".join(x))
    movies["genres_name"] = movies["genres_name"].str.replace(r'["\'\[\],]', '', regex=True)
    movies["genres_name"] = movies["genres_name"].fillna("Unknown")
    # Преобразование данных столбца 'date_publication' в таблице 'movies'
    movies["date_publication"] = pd.to_datetime(movies["date_publication"])
    movies["year_publication"] = (
        movies["date_publication"].dt.year.fillna(0).astype("category")
    )
    movies["month_publication"] = (
        movies["date_publication"].dt.month.fillna(0).astype("category")
    )
    movies["release_day_of_week"] = movies["date_publication"].dt.dayofweek
    movies["genres"] = movies["genres"].apply(lambda x: x.split(", "))
    num_genres = pd.Series(np.hstack(movies["genres"].values)).value_counts()
    movies["genres_min"] = movies["genres"].apply(
        lambda x: min([num_genres[el] for el in x])
    )
    movies["genres_max"] = movies["genres"].apply(
        lambda x: max([num_genres[el] for el in x])
    )
    movies["genres_med"] = movies["genres"].apply(
        lambda x: (np.median([num_genres[el] for el in x]))
    )
    movies["countries"].fillna("None", inplace=True)
    movies["countries"] = movies["countries"].str.lower()
    movies["countries_list"] = movies["countries"].apply(
        lambda x: x.split(", ") if ", " in x else [x]
    )
    num_countries = pd.Series(
        np.hstack(movies["countries_list"].values)
        ).value_counts()
    movies["countries_max"] = movies["countries_list"].apply(
        lambda x: max([num_countries[el] for el in x])
    )
    movies.drop(["countries_list"], axis=1, inplace=True)

    average_duration = logs["duration"].mean()

    # Вычисление % просмотра фильма и округление до 2 знаков после запятой
    logs["watched_pct"] = (
        (logs["duration"] / average_duration)
        .round(2)
        .apply(lambda x: min(x, 1))
    )
    logs["watched_pct"] = logs["watched_pct"].fillna(0)
    # Extract  day of the week
    logs["weekday"] = logs["datetime"].dt.dayofweek
    max_date = logs["datetime"].max()
    logs["recency"] = (max_date - logs["datetime"]).dt.days
    # Сделаем матрицу весов, зависящую от даты последнего просмотра
    dt_max = np.datetime64(logs["datetime"].max())
    logs["date_weight"] = 1 / (
        (dt_max - logs["datetime"].values).astype("timedelta64[D]")
        / np.timedelta64(1, "D")
        + 1
    )
    # Создание экземпляра AveragePopularityRank и применение к movies
    apr_calculator = AveragePopularityRank(k=200, actions=logs)
    movies["popularity_rank"] = movies["movie_id"].apply(apr_calculator)
    logs["is_weekend"] = logs["weekday"].isin([5, 6])
    weekend_activity = (
        logs.groupby("user_id")
        ["is_weekend"]
        .mean()
        .reset_index()
    )
    # Cредняя активность в выходные дни для каждого пользователя
    weekend_activity.rename(
        columns={"is_weekend": "weekend_activity"}, inplace=True
        )
    user_stats = (
        logs.groupby("user_id")["watched_pct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    user_stats.rename(
        columns={"mean": "avg_watched_pct", "count": "total_watched"},
        inplace=True
    )

    # Adding the additional columns to the user_stats DataFrame
    user_stats["average_duration"] = (
        logs.groupby("user_id")["duration"].mean().reset_index()["duration"]
    )
    user_stats["max_recency"] = (
        logs.groupby("user_id")["recency"].max().reset_index()["recency"]
    )
    user_stats["average_date_weight"] = (
        logs.groupby("user_id")["date_weight"]
        .mean()
        .reset_index()["date_weight"]
    )

    # Adding the "is_weekend" information to the user_stats DataFrame
    user_stats = pd.merge(
        user_stats,
        weekend_activity,
        on="user_id",
        how="left"
        )
    # Save user_stats_df to a CSV file
    user_stats.to_csv("files/user_stats.csv", index=False)
    item_stats = movies[["movie_id"]].set_index("movie_id")
    item_stats = add_item_watches_stats(logs, item_stats)
    item_stats.fillna(0, inplace=True)

    movies_with_features = pd.merge(
        item_stats, movies, how="left", left_on="movie_id", right_on="movie_id"
    )

    genre_stats = (
        movies.groupby("genres_min")["popularity_rank"]
        .agg(["mean", "count"])
        .reset_index()
    )
    genre_stats.rename(columns={
        "mean": "avg_popularity_rank_genre",
        "count": "total_movies_genre"
    }, inplace=True)
    movies_with_features = pd.merge(
        movies_with_features,
        genre_stats,
        how="left",
        left_on="genres_min",
        right_on="genres_min",
    )
    movies_with_features.drop(
        ["date_publication", "genres"],
        axis=1,
        inplace=True
        )
    movies_with_features.to_csv(
        os.path.join(output, "movies_processed.csv"), index=False
    )
    logs.to_csv(os.path.join(output, "logs_processed.csv"), index=False)

    return logs, movies_with_features, user_stats
