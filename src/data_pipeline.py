import yfinance as yf
import pandas as pd
import os

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS", "BAJFINANCE.NS"
]

def download_data(start="2018-01-01", end="2026-01-01", out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    for ticker in TICKERS:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            df.to_csv(f"{out_dir}/{ticker}.csv")
            print(f"  Saved {ticker}: {len(df)} rows")
        else:
            print(f"  WARNING: no data for {ticker}")

if __name__ == "__main__":
    download_data()