import numpy as np
import pandas as pd


def create_deciles(df, pd_column="pd"):

    df["bin"] = pd.qcut(df[pd_column], 10, labels=False)

    summary = df.groupby("bin").agg(
        avg_pd=(pd_column, "mean"),
        default_rate=("target", "mean"),
        count=("target", "count")
    )

    return df, summary


def generate_score(df, pd_column="pd", PDO=20, base_score=600, base_odds=50):

    factor = PDO / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    df["score"] = offset - factor * np.log(
        df[pd_column] / (1 - df[pd_column])
    )

    return df