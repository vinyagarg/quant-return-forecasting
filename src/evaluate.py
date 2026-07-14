import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import joblib

from src.model import FEATURES, LABEL, train_test_split_by_date

def compute_ic(preds, actuals):
    """Information Coefficient: Pearson correlation between predicted and actual returns"""
    return pd.Series(preds).corr(pd.Series(np.asarray(actuals)))

def compute_rank_ic(preds, actuals):
    """Rank IC: Spearman correlation — more robust to outliers"""
    corr, _ = spearmanr(preds, actuals)
    return corr

def daily_ic_series(test_df, preds):
    test_df = test_df.copy()
    test_df["pred"] = preds
    daily_ic = test_df.groupby(test_df.index).apply(
        lambda x: x["pred"].corr(x["fwd_ret_5d"]) if len(x) > 1 else np.nan
    )
    return daily_ic.dropna()

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/dataset.parquet")
    train_df, test_df = train_test_split_by_date(df)

    model = joblib.load("model.pkl")
    preds = model.predict(test_df[FEATURES])
    actuals = test_df[LABEL]

    ic = compute_ic(preds, actuals)
    rank_ic = compute_rank_ic(preds, actuals)

    daily_ic = daily_ic_series(test_df, preds)
    ic_tstat = daily_ic.mean() / daily_ic.std() * np.sqrt(len(daily_ic))

    print(f"Mean IC (Pearson):     {ic:.4f}")
    print(f"Mean Rank IC (Spearman): {rank_ic:.4f}")
    print(f"Daily IC t-stat:       {ic_tstat:.4f}")
    print(f"Number of test days:   {len(daily_ic)}")