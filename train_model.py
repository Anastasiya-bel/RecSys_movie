import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import pickle
import os


def train_lgbm(
        candidates,
        logs_with_features,
        movies_with_features,
        cosine_similarity,
        user_stats):
    logs_with_features["datetime"] = pd.to_datetime(
        logs_with_features["datetime"],
        utc=True
    )
    df_user_gr = logs_with_features.groupby('user_id')\
        .movie_id.nunique().reset_index()
    warm_users = logs_with_features[
        logs_with_features['user_id'].isin(
            df_user_gr[df_user_gr['movie_id'] >= 3]['user_id'].unique()
        )
    ].copy()
    # taking candidates from lightfm model and generating positive samples
    pos = candidates.merge(
        warm_users,
        on=['user_id', 'movie_id'],
        how='inner'
    )
    pos["target"] = 1

    # Generating negative samples
    # target = 0 все что пользователь НЕ посмотрел из кандидатов lightfm
    # добавим сэмплирование, чтобы соблюсти баланс классов
    neg = candidates.set_index(['user_id', 'movie_id'])\
        .join(warm_users.set_index(['user_id', 'movie_id']))
    neg = neg[neg["watched_pct"].isnull()].reset_index()
    neg = neg.sample(frac=(pos.shape[0] / neg.shape[0]) * 1.0)
    neg["target"] = 0

    # Делим по пользователям, а не по дате.
    # Мотивация:для негативных взаимодействий нет даты
    lgbm_train_users, lgbm_eval_users = train_test_split(
        warm_users["user_id"].unique(), random_state=1, test_size=0.2
    )
    select_col = [
        "user_id",
        "movie_id",
        "rank_SASRec",
        "target",
    ]
    lgbm_train = shuffle(
        pd.concat(
            [
                pos[pos["user_id"].isin(lgbm_train_users)],
                neg[neg["user_id"].isin(lgbm_train_users)],
            ]
        )[select_col]
    )

    lgbm_eval = shuffle(
        pd.concat(
            [
                pos[pos["user_id"].isin(lgbm_eval_users)],
                neg[neg["user_id"].isin(lgbm_eval_users)],
            ]
        )[select_col]
    )
    lgbm_train.to_csv("files/train.csv")
    lgbm_eval.to_csv("files/test.csv")
    item_col = [
        "movie_id",
        "watch_ts_quantile_95_diff",
        "watch_ts_median_diff",
        "watch_ts_std",
        "watched_in_all_time",
        "name",
        "year",
        "description",
        "countries",
        "release_novelty",
        "actors",
        "director",
        "genres_name",
        "year_publication",
        "month_publication",
        "release_day_of_week",
        "genres_min",
        "genres_max",
        "genres_med",
        "countries_max",
        "popularity_rank",
        "avg_popularity_rank_genre",
        "total_movies_genre"
        ]
    cat_col = [
        "year",
        "genres_name",
        "countries",
        "name",
        "actors",
        "director",
        "year_publication",
        "month_publication",
        "release_day_of_week"
    ]
    user_stats_columns = [
        "user_id",
        "avg_watched_pct",
        "total_watched",
        "average_duration",
        "max_recency",
        "average_date_weight",
        "weekend_activity"
    ]
    train_feat = (
        lgbm_train
        .merge(user_stats[user_stats_columns], on=["user_id"], how="left")
        .merge(movies_with_features[item_col], on=["movie_id"], how="left")
    )
    eval_feat = (
        lgbm_eval.merge(user_stats[user_stats_columns], on=["user_id"], how="left")
        .merge(movies_with_features[item_col], on=["movie_id"], how="left")
    )
    cosine_similarity_columns = [
        "user_id",
        "movie_id",
        "cosine_similarity_last",
        "cosine_similarity_second_last"
    ]
    train_feat = pd.merge(
        train_feat,
        cosine_similarity[cosine_similarity_columns],
        on=['user_id', 'movie_id'],
        how='left'
    )
    eval_feat = pd.merge(
        eval_feat,
        cosine_similarity[cosine_similarity_columns],
        on=['user_id', 'movie_id'],
        how='left'
        )
    drop_col = ["user_id", "movie_id", "description"]
    target_col = ["target"]

    label_encoder = LabelEncoder()
    combined_data = pd.concat([train_feat, eval_feat], axis=0)

    for column in cat_col:
        combined_data[column] = label_encoder.fit_transform(
            combined_data[column]
        )
    train_feat = combined_data[:len(train_feat)]
    eval_feat = combined_data[len(train_feat):]

    X_train, y_train = (
        train_feat.drop(drop_col + target_col, axis=1),
        train_feat[target_col],
    )
    X_val, y_val = (
        eval_feat.drop(drop_col + target_col, axis=1),
        eval_feat[target_col]
    )
    X_train = X_train.fillna(X_train.mode().iloc[0])
    X_val = X_val.fillna(X_train.mode().iloc[0])
    # Конвертация данных в формат LightGBM
    train_data_lgb = lgb.Dataset(
        X_train, y_train, categorical_feature=cat_col, free_raw_data=False
    )
    # Training LightGBM with parameters previously chosen on cross validation
    params = {
        "application": "binary",
        "objective": "binary",
        "metric": "binary_logloss",
        "is_unbalance": "true",
        "boosting": "gbdt",
        "num_leaves": 30,
        "bagging_freq": 20,
        "verbose": 1,
        "importance_type": "gain",
        "class_weight": "balanced",
        "drop_rate": 0.9,
        "min_data_in_leaf": 30,
        "max_bin": 555,
        "n_estimators": 300,
        "min_sum_hessian_in_leaf": 1,
        "learning_rate": 0.2,
        "bagging_fraction": 0.85,
        "colsample_bytree": 1.0,
        "feature_fraction": 0.1,
        "lambda_l1": 5.0,
        "lambda_l2": 3.0,
        "max_depth": 9,
        "min_child_samples": 20,
        "min_child_weight": 3.0,
        "min_split_gain": 0.0,
        "subsample": 0.7,
    }
    fixed_num_boost_rounds = 2500
    lgbm = lgb.train(
        params,
        train_data_lgb,
        num_boost_round=fixed_num_boost_rounds
    )
    importance = lgbm.feature_importance(importance_type="gain")
    feature_importance_df = pd.DataFrame(
        {"Feature": lgbm.feature_name(), "Importance": importance}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )
    print(feature_importance_df)
    model_directory = os.path.join("models")
    model_filename = "lgbm_model.dill"

    os.makedirs(model_directory, exist_ok=True)

    model_path = os.path.join(model_directory, model_filename)

    with open(model_path, "wb") as f:
        pickle.dump(lgbm, f)
    return lgbm
