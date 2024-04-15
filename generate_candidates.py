import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import dill
import os
from tqdm import tqdm


def train_lightfm_model(logs):
    """
    Обучает модель LightFM

    Parameters:
    logs : DataFrame
        - Должен содержать столбцы 'user_id', 'movie_id', 'duration'.

    Returns: tuple
        - Кортеж из обученной модели LightFM
        и словаря с отображениями пользователей и объектов.

    Обучает модель LightFM
    на основе журналов взаимодействия пользователей с объектами.
    Создает матрицы взаимодействий и весов на основе этих журналов.
    Проводит обучение модели на полученных данных
    в течение заданного количества эпох.
    Сохраняет обученную модель в дир. 'models' в формате 'lfm_model.dill'.
    Возвращает обученную модель LightFM и
    словарь отображений пользователей и объектов.
    """
    dataset = Dataset()
    dataset.fit(logs["user_id"].unique(), logs["movie_id"].unique())
    interactions_matrix, weights_matrix = dataset.build_interactions(
        zip(*logs[["user_id", "movie_id", "duration"]].values.T)
    )
    weights_matrix_csr = weights_matrix.tocsr()
    # user / item mappings
    lightfm_mapping = dataset.mapping()
    lightfm_mapping = {
        "users_mapping": lightfm_mapping[0],
        "movies_mapping": lightfm_mapping[2],
    }
    lightfm_mapping["users_inv_mapping"] = {
        v: k for k, v in lightfm_mapping["users_mapping"].items()
    }
    lightfm_mapping["movies_inv_mapping"] = {
        v: k for k, v in lightfm_mapping["movies_mapping"].items()
    }
    # Training LightFM model
    lfm_model = LightFM(
        no_components=64,
        learning_rate=0.01,
        loss="warp",
        max_sampled=5,
        random_state=42
    )
    num_epochs = 100

    for _ in tqdm(range(num_epochs)):
        lfm_model.fit_partial(weights_matrix_csr)

    # Saving the model
    model_directory = os.path.join("models")
    model_filename = "lfm_model.dill"

    os.makedirs(model_directory, exist_ok=True)

    model_path = os.path.join(model_directory, model_filename)

    with open(model_path, "wb") as f:
        dill.dump(lfm_model, f)

    return lfm_model, lightfm_mapping


def generate_lightfm_recs_mapper(
    model,
    item_ids,
    known_items,
    user_features,
    item_features,
    N,
    user_mapping,
    item_inv_mapping,
    num_threads=1,
):
    """
    Генерирует рекомендации на основе модели LightFM.

    Parameters:
    model : LightFM
        Обученная модель LightFM.
    item_ids : list
        Список идентификаторов объектов(фильмов).
    known_items : dict
        Словарь с известными фильмами для каждого пользователя.
        Ключи словаря - идентификаторы пользователей,
        значения - списки известных фильмов.
    user_features : array-like or None
        Признаки пользователей (если используются).
    item_features : array-like or None
        Признаки объектов (если используются).
    N : int
        Количество рекомендаций для генерации.
    user_mapping : dict
        Соответствие идентификаторов пользователей в модели LightFM.
    item_inv_mapping : dict
        Обратное соответствие идентификаторов объектов в модели LightFM.
    num_threads : int, optional
        Количество потоков для выполнения предсказаний (по умолчанию 1).

    Returns:
    function
    """
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(
            user_id,
            item_ids,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )
        # Рассчитываем дополнительное количество рекомендаций для пользователя
        additional_N = len(known_items[user_id]) \
            if user_id in known_items else 0
        total_N = N + additional_N
        # Находим top-N рекомендаций
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]
        # Формируем список рекомендаций, исключая уже известные объекты
        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs
                          if item not in filter_items]
        return final_recs[:N]

    return _recs_mapper


def generate_and_save_candidates(logs):
    """
    Генерирует кандидаты для рекомендаций с помощью модели LightFM
     и сохраняет результаты в CSV.

    Parameters:
    logs (DataFrame): DataFrame с взаимодействием пользователей с объектами.
        Должен содержать столбцы 'user_id', 'movie_id', 'duration'.

    Returns:
    None

    Generates candidates for recommendations using the LightFM
    model based on user-item interactions logs
    and saves the results in a CSV file 'candidates_lightfm.csv'
    containing columns: 'user_id', 'movie_id', 'rank'.
    'user_id' represents the user, 'movie_id' is the recommended movie,
    and 'rank' indicates the rank of the recommendation.
    """
    logs["datetime"] = pd.to_datetime(logs["datetime"], utc=True)
    lfm_model, lightfm_mapping = train_lightfm_model(logs)
    # Определение "теплых" пользователей
    df_user_gr = logs.groupby('user_id').movie_id.nunique().reset_index()
    warm_users = logs[
        logs['user_id'].isin(
            df_user_gr[df_user_gr['movie_id'] >= 3]['user_id'].unique()
        )
    ].copy()
    top_N = 100
    all_cols = list(lightfm_mapping["movies_mapping"].values())

    candidates = pd.DataFrame({"user_id": warm_users["user_id"].unique()})
    overall_known_items = (
       warm_users.groupby("user_id")["movie_id"].apply(list).to_dict()
    )
    mapper = generate_lightfm_recs_mapper(
        lfm_model,
        item_ids=all_cols,
        known_items=overall_known_items,
        N=top_N,
        user_features=None,
        item_features=None,
        user_mapping=lightfm_mapping["users_mapping"],
        item_inv_mapping=lightfm_mapping["movies_inv_mapping"],
        num_threads=20,
    )

    candidates["movie_id"] = candidates["user_id"].map(mapper)
    candidates = candidates.explode("movie_id")
    candidates["rank"] = candidates.groupby("user_id").cumcount() + 1
    file_directory = os.path.join("files")
    candidates.to_csv(
        os.path.join(file_directory, "candidates_lightfm.csv"), index=False
    )
    return candidates
