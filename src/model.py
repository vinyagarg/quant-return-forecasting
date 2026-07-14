import lightgbm as lgb
import pandas as pd
import joblib

FEATURES = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "volume_chg", "rsi_14", "bb_pct"]
LABEL = "fwd_ret_5d"

def train_test_split_by_date(df, split_date="2024-06-01"):
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    print(f"Train: {len(train)} rows ({train.index.min()} to {train.index.max()})")
    print(f"Test:  {len(test)} rows ({test.index.min()} to {test.index.max()})")
    return train, test

def train_model(train_df):
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train_df[FEATURES], train_df[LABEL])
    return model

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/dataset.parquet")
    train_df, test_df = train_test_split_by_date(df)
    model = train_model(train_df)
    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl")