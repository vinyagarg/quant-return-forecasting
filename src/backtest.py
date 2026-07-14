import pandas as pd
import numpy as np
import joblib

from src.model import FEATURES, LABEL, train_test_split_by_date

def simple_long_short_backtest(test_df, preds, top_pct=0.2, cost_bps=10):
    test_df = test_df.copy()
    test_df["pred"] = preds

    daily_pnl = []
    for date, group in test_df.groupby(test_df.index):
        n = len(group)
        if n < 5:
            continue
        group = group.sort_values("pred", ascending=False)
        n_top = max(1, int(n * top_pct))
        longs = group.iloc[:n_top]
        shorts = group.iloc[-n_top:]

        long_ret = longs["fwd_ret_5d"].mean()
        short_ret = shorts["fwd_ret_5d"].mean()
        gross_pnl = (long_ret - short_ret) / 2

        turnover_cost = (cost_bps / 10000) * 2  # round-trip cost estimate
        net_pnl = gross_pnl - turnover_cost
        daily_pnl.append({"date": date, "gross_pnl": gross_pnl, "net_pnl": net_pnl})

    return pd.DataFrame(daily_pnl).set_index("date")

def annualized_sharpe(daily_returns, periods_per_year=252):
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(periods_per_year)

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/dataset.parquet")
    train_df, test_df = train_test_split_by_date(df)

    model = joblib.load("model.pkl")
    preds = model.predict(test_df[FEATURES])

    pnl_df = simple_long_short_backtest(test_df, preds)
    pnl_df = pnl_df.iloc[::5]  # keep only every 5th day — removes overlapping-window bias

    gross_sharpe = annualized_sharpe(pnl_df["gross_pnl"], periods_per_year=252 / 5)
    net_sharpe = annualized_sharpe(pnl_df["net_pnl"], periods_per_year=252 / 5)

    print(f"Gross Sharpe (before costs): {gross_sharpe:.2f}")
    print(f"Net Sharpe (after costs):    {net_sharpe:.2f}")
    print(f"Mean daily gross PnL:        {pnl_df['gross_pnl'].mean():.5f}")
    print(f"Mean daily net PnL:          {pnl_df['net_pnl'].mean():.5f}")
    print(f"Total backtest days:         {len(pnl_df)}")