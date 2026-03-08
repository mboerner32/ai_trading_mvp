"""
5-minute intraday LSTM for predicting same-session +20% hit probability.

Architecture: 2-layer LSTM → sigmoid binary classifier
Input:  12 five-minute bars (9:30–10:25 ET) × 7 features per bar
Output: probability that intraday HIGH touches +20% above session open
        before 3:50 PM ET on the same trading day

Features (7 per bar):
  1. bar_return      — (close − open) / open, clipped ±0.5
  2. cum_gain        — (close − day_open) / day_open, clipped 0–2.0
  3. vol_ratio       — bar_volume / (session_avg_vol / 78), clipped 0–50 then /50
  4. high_wick       — (high − close) / close, upside wick, clipped 0–0.5
  5. low_wick        — (close − low) / close, downside wick, clipped 0–0.5
  6. shares_log      — log10(shares_outstanding/1M)/3, clipped −1–1
  7. is_biotech      — 1.0 if Healthcare sector, else 0.0

Training data built by build_5m_sequences() using yfinance 60d/5m data.
Saved to lstm_sequences_5m.npz. Trained model saved to lstm_model_5m.pt.
"""

import os
from datetime import datetime, time as dtime

import numpy as np

SEQUENCE_LEN_5M = 12          # first hour of 5m bars (9:30–10:25 ET)
FEATURES_5M     = 7
FEATURES_VER_5M = 1
MODEL_PATH_5M   = "lstm_model_5m.pt"
SEQ_DATA_PATH_5M = "lstm_sequences_5m.npz"

# Qualifying criteria for a 5m training session
_MIN_OPEN_GAIN   = 0.05   # by bar 6 (10:00 AM), stock must be up ≥5% from open
_MIN_BAR6_CUMVOL = 100_000  # minimum cumulative volume by bar 6
_MAX_OPEN_PRICE  = 5.0    # session open price must be ≤ $5
_TARGET_GAIN     = 0.20   # label = 1 if intraday high hits +20% from open


def _extract_5m_features(bars: list, day_open: float,
                          session_total_vol: float,
                          shares_outstanding: float = None,
                          sector: str = None) -> list | None:
    """
    Extract feature vectors from a list of 5m bar dicts.
    Each bar: {"open", "high", "low", "close", "volume"}.
    Returns list of SEQUENCE_LEN_5M feature vectors, or None on failure.
    """
    try:
        import math

        if len(bars) < SEQUENCE_LEN_5M or day_open <= 0:
            return None

        avg_bar_vol = max(session_total_vol / 78, 1.0)

        if shares_outstanding and shares_outstanding > 0:
            log_shares = float(np.clip(math.log10(shares_outstanding / 1e6) / 3, -1.0, 1.0))
        else:
            log_shares = 0.0
        is_biotech = 1.0 if sector and "health" in sector.lower() else 0.0

        seq = []
        for bar in bars[:SEQUENCE_LEN_5M]:
            o = float(bar.get("open") or 0)
            h = float(bar.get("high") or 0)
            l = float(bar.get("low") or 0)
            c = float(bar.get("close") or 0)
            v = float(bar.get("volume") or 0)
            if c <= 0 or o <= 0:
                return None

            bar_ret   = float(np.clip((c - o) / o, -0.5, 0.5))
            cum_gain  = float(np.clip((c - day_open) / day_open, 0.0, 2.0))
            vol_ratio = float(np.clip(v / avg_bar_vol, 0.0, 50.0) / 50.0)
            high_wick = float(np.clip((h - c) / c, 0.0, 0.5))
            low_wick  = float(np.clip((c - l) / c, 0.0, 0.5))

            seq.append([bar_ret, cum_gain, vol_ratio, high_wick, low_wick,
                        log_shares, is_biotech])

        return seq if len(seq) == SEQUENCE_LEN_5M else None
    except Exception:
        return None


def build_5m_sequences(tickers: list = None) -> int:
    """
    Download 60 days of 5m OHLCV for each ticker, identify qualifying intraday
    sessions, extract feature sequences from first SEQUENCE_LEN_5M bars, and
    label each session (1 = hit +20% from open by 3:50 PM, 0 = did not).
    Saves to lstm_sequences_5m.npz. Returns number of sequences built.
    """
    import yfinance as yf

    if tickers is None:
        # Default: use a broad list of known momentum tickers
        from app.backfill import SEED_TICKERS
        tickers = SEED_TICKERS

    X_list, y_list = [], []

    for symbol in tickers:
        try:
            df = yf.download(symbol, period="60d", interval="5m",
                             progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < SEQUENCE_LEN_5M + 10:
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, __import__("pandas").MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            # Get sector/shares for static features
            shares_outstanding = None
            sector = None
            try:
                info = yf.Ticker(symbol).fast_info
                shares_outstanding = getattr(info, "shares", None)
            except Exception:
                pass
            try:
                sector = yf.Ticker(symbol).info.get("sector")
            except Exception:
                pass

            # Group bars by trading date
            import pandas as pd
            df.index = pd.to_datetime(df.index)
            df["date"] = df.index.date

            for date, day_df in df.groupby("date"):
                day_df = day_df.sort_index()
                if len(day_df) < SEQUENCE_LEN_5M + 5:
                    continue

                bars = day_df[["open", "high", "low", "close", "volume"]].to_dict("records")
                day_open = float(bars[0].get("open") or 0)
                if day_open <= 0 or day_open > _MAX_OPEN_PRICE:
                    continue

                # Check qualifying criteria by bar 6
                cum_vol_6 = sum(float(b.get("volume") or 0) for b in bars[:6])
                close_6   = float(bars[5].get("close") or 0)
                if cum_vol_6 < _MIN_BAR6_CUMVOL:
                    continue
                if close_6 <= 0 or (close_6 - day_open) / day_open < _MIN_OPEN_GAIN:
                    continue

                # Compute session total volume for normalization
                session_total_vol = float(day_df["volume"].sum())

                # Build feature sequence from first SEQUENCE_LEN_5M bars
                seq = _extract_5m_features(bars, day_open, session_total_vol,
                                           shares_outstanding, sector)
                if seq is None:
                    continue

                # Label: did intraday HIGH after bar SEQUENCE_LEN_5M hit +20%?
                target_price = day_open * (1 + _TARGET_GAIN)
                hit = any(
                    float(b.get("high") or 0) >= target_price
                    for b in bars[SEQUENCE_LEN_5M:]
                )

                X_list.append(seq)
                y_list.append(1 if hit else 0)

        except Exception as e:
            print(f"5m LSTM: error on {symbol} — {e}")
            continue

    if not X_list:
        print("5m LSTM: no sequences built — try more tickers or check data availability")
        return 0

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    np.savez(SEQ_DATA_PATH_5M, X=X, y=y)
    pos = int(y.sum())
    print(f"5m LSTM: saved {len(X_list)} sequences "
          f"({pos} positive / {len(X_list) - pos} negative) to {SEQ_DATA_PATH_5M}")
    return len(X_list)


def _build_5m_model():
    import torch.nn as nn
    class LSTM5mPredictor(nn.Module):
        def __init__(self, input_size=FEATURES_5M, hidden=32, layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers,
                                batch_first=True, dropout=dropout)
            self.fc   = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return __import__("torch").sigmoid(self.fc(out[:, -1, :]))

    return LSTM5mPredictor()


def train_lstm_5m(epochs: int = 30, lr: float = 0.001, batch_size: int = 32) -> dict:
    """
    Load lstm_sequences_5m.npz, train the 5m LSTM, save lstm_model_5m.pt.
    Returns {"accuracy": float, "samples": int, "epochs": int}.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    if not os.path.exists(SEQ_DATA_PATH_5M):
        raise FileNotFoundError(f"{SEQ_DATA_PATH_5M} not found — run build_5m_sequences() first")

    data = np.load(SEQ_DATA_PATH_5M)
    X, y = data["X"], data["y"]

    if len(X) < 30:
        raise ValueError(f"Only {len(X)} sequences — need at least 30 to train")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val).unsqueeze(1))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model     = _build_5m_model()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = (model(xb) >= 0.5).float()
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  5m LSTM epoch {epoch}/{epochs} — "
                  f"train_loss={train_loss/len(train_dl):.4f}  val_acc={val_acc:.3f}")

    torch.save({
        "model_state":    model.state_dict(),
        "val_accuracy":   val_acc,
        "samples":        len(X),
        "trained_at":     datetime.utcnow().isoformat(),
        "sequence_len":   SEQUENCE_LEN_5M,
        "features":       FEATURES_5M,
        "features_ver":   FEATURES_VER_5M,
    }, MODEL_PATH_5M)

    print(f"5m LSTM: saved to {MODEL_PATH_5M}  (val_accuracy={val_acc:.3f}, samples={len(X)})")
    return {"accuracy": round(val_acc, 4), "samples": len(X), "epochs": epochs}


def load_lstm_5m():
    """Load and return trained 5m model, or None if not available."""
    try:
        import torch
        ckpt = torch.load(MODEL_PATH_5M, map_location="cpu", weights_only=False)
        if ckpt.get("features_ver", 0) != FEATURES_VER_5M:
            print("5m LSTM: feature version mismatch — retrain required")
            return None
        model = _build_5m_model()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
    except Exception:
        return None


def predict_5m_hit_probability(symbol: str,
                                shares_outstanding: int = None,
                                sector: str = None) -> float | None:
    """
    Download latest 5m bars for symbol, extract the most recent qualifying
    intraday session, run the 5m LSTM, return 0–1 probability.
    Returns None on any error.
    """
    try:
        import torch
        import yfinance as yf
        import pandas as pd

        model = load_lstm_5m()
        if model is None:
            return None

        df = yf.download(symbol, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        df.index = pd.to_datetime(df.index)
        df["date"] = df.index.date

        # Use the most recent trading day that has enough bars
        for date in sorted(df["date"].unique(), reverse=True):
            day_df = df[df["date"] == date].sort_index()
            if len(day_df) < SEQUENCE_LEN_5M:
                continue

            bars = day_df[["open", "high", "low", "close", "volume"]].to_dict("records")
            day_open = float(bars[0].get("open") or 0)
            if day_open <= 0 or day_open > _MAX_OPEN_PRICE:
                continue

            session_total_vol = float(day_df["volume"].sum())
            seq = _extract_5m_features(bars[:SEQUENCE_LEN_5M], day_open,
                                        session_total_vol, shares_outstanding, sector)
            if seq is None:
                continue

            x = torch.tensor([seq], dtype=torch.float32)
            with torch.no_grad():
                prob = model(x).item()
            return round(prob, 4)

        return None

    except Exception:
        return None


def get_lstm_5m_status() -> dict:
    """Return metadata about the 5m model, or {"trained": False}."""
    try:
        import torch
        ckpt = torch.load(MODEL_PATH_5M, map_location="cpu", weights_only=False)
        return {
            "trained":    True,
            "accuracy":   round(ckpt.get("val_accuracy", 0) * 100, 1),
            "samples":    ckpt.get("samples", 0),
            "trained_at": (ckpt.get("trained_at", "") or "")[:10],
        }
    except Exception:
        return {"trained": False}


def get_5m_sequence_stats() -> dict:
    """Return stats about lstm_sequences_5m.npz."""
    try:
        if not os.path.exists(SEQ_DATA_PATH_5M):
            return {"available": False}
        data = np.load(SEQ_DATA_PATH_5M)
        y = data["y"]
        total    = int(len(y))
        positive = int(y.sum())
        pos_rate = round(positive / total * 100, 1) if total > 0 else 0.0
        import time as _time
        built_at = datetime.utcfromtimestamp(
            _time.time() if not os.path.exists(SEQ_DATA_PATH_5M)
            else os.path.getmtime(SEQ_DATA_PATH_5M)
        ).strftime("%Y-%m-%d")
        return {
            "available": True,
            "total":     total,
            "positive":  positive,
            "negative":  total - positive,
            "pos_rate":  pos_rate,
        }
    except Exception:
        return {"available": False}
