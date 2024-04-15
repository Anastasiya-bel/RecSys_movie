import pandas as pd
from itertools import islice, cycle


class PopularRecommender:
    """
    Makes recommendations based on popular items
    """

    def __init__(
        self,
        max_K=20,
        days=30,
        item_column="movie_id",
        dt_column="datetime",
        with_filter=False,
    ):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []

    def fit(
        self,
        df,
    ):
        # Convert the datetime column to datetime type
        df[self.dt_column] = pd.to_datetime(df[self.dt_column], utc=True)

        min_date = (
            df[self.dt_column]
            .max()
            .normalize()
            - pd.DateOffset(days=self.days)
        )
        self.recommendations = (
            df.loc[df[self.dt_column] > min_date, self.item_column]
            .value_counts()
            .head(self.max_K)
            .index.values
        )

    def recommend(self, users=None, N=20):
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            return list(islice(cycle([recs]), len(users)))

    def recommend_with_filter(self, train, user_ids, top_K=20):
        user_ids = pd.Series(user_ids)
        watched_users = user_ids[user_ids.isin(train["user_id"])]
        new_users = user_ids[~user_ids.isin(watched_users)]
        full_recs = self.recommendations
        topk_recs = full_recs[:top_K]
        new_recs = pd.DataFrame({"user_id": new_users})
        new_recs["movie_id"] = list(islice(cycle([topk_recs]), len(new_users)))
        watched_recs = pd.DataFrame({"user_id": watched_users})
        watched_recs["movie_id"] = 0
        known_items = (
            train.groupby("user_id")["movie_id"]
            .apply(list)
            .to_dict()
        )
        watched_recs["additional_N"] = watched_recs["user_id"].apply(
            lambda user_id: len(known_items[user_id])
            if user_id in known_items
            else 0
        )
        watched_recs["total_N"] = watched_recs["additional_N"].apply(
            lambda add_N: add_N + top_K
            if add_N + top_K < len(full_recs)
            else len(full_recs)
        )
        watched_recs["total_recs"] = watched_recs["total_N"].apply(
            lambda total_N: full_recs[:total_N]
        )
        filter_func = lambda row: [
            item
            for item in row["total_recs"]
            if item not in known_items[row["user_id"]]
        ][:top_K]
        watched_recs["movie_id"] = watched_recs.loc[:, ["total_recs", "user_id"]].apply(
            filter_func, axis=1
        )
        watched_recs = watched_recs[["user_id", "movie_id"]]
        return pd.concat([new_recs, watched_recs], axis=0)
