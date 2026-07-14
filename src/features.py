import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Features: only use info available up to time T ---
    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["volume_chg"] = df["Volume"].pct_change(5)
    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()

    bb = BollingerBands(df["Close"], window=20)
    df["bb_pct"] = (df["Close"] - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )

    # --- Label: forward return — this is the ONLY column allowed to look ahead ---
    df["fwd_ret_5d"] = df["Close"].shift(-5) / df["Close"] - 1

    return df