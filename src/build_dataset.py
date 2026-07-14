import pandas as pd
import glob
import os
from src.features import build_features

def build_full_dataset(raw_dir="data/raw", out_path="data/processed/dataset.parquet"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_frames = []

    for filepath in glob.glob(f"{raw_dir}/*.csv"):
        ticker = os.path.basename(filepath).replace(".csv", "")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df = build_features(df)
        df["ticker"] = ticker
        df = df.dropna()  # drops both warm-up rows (rolling windows) and last 5 rows (no label yet)
        all_frames.append(df)

    full_df = pd.concat(all_frames).sort_index()
    full_df.to_parquet(out_path)
    print(f"Final dataset shape: {full_df.shape}")
    print(f"Date range: {full_df.index.min()} to {full_df.index.max()}")
    print(f"Tickers included: {full_df['ticker'].nunique()}")
    return full_df

if __name__ == "__main__":
    build_full_dataset()