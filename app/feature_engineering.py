import pandas as pd


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all derived trading features to the dataframe.
    Expects standard OHLCV columns from yfinance.
    """

    if df is None or len(df) < 20:
        return None

    df = df.copy()

    # Ensure flat columns (yfinance sometimes returns multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standard column names safeguard
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # --- Daily Return ---
    df["daily_return"] = df["close"].pct_change()

    # --- 5 Day Return ---
    df["return_5d"] = df["close"].pct_change(5)

    # --- 3 Day Return ---
    df["return_3d"] = df["close"].pct_change(3)

    # --- 20 Day Avg Volume ---
    df["avg_volume_20"] = df["volume"].rolling(20).mean()

    # --- Relative Volume ---
    df["relative_volume"] = df["volume"] / df["avg_volume_20"]

    # --- 10 Day Range (Sideways Check) ---
    df["range_10d"] = (
        df["high"].rolling(10).max()
        - df["low"].rolling(10).min()
    ) / df["close"]

    return df
