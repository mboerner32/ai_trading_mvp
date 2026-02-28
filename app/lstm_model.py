"""
Local PyTorch LSTM model for predicting 20%-hit probability on low-float momentum stocks.

Architecture: 2-layer LSTM → sigmoid binary classifier
Input:  20 trading days × 6 features per day
Output: probability that stock intraday HIGH touches +20% within 7 trading days

Training data is built by build_sequences_from_backfill() which is called automatically
after build_historical_dataset() completes. Sequences are saved to lstm_sequences.npz.
The trained model is saved to lstm_model.pt.
"""

import os
from datetime import datetime

import numpy as np

SEQUENCE_LEN  = 20          # 20 trading days of history per input window
FEATURES      = 6           # features per time step (see extract_features())
MODEL_PATH    = "lstm_model.pt"
SEQ_DATA_PATH = "lstm_sequences.npz"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(row) -> list | None:
    """
    Extract 6 normalised features from a prepared_dataframe row.
    Returns None if any required field is missing/invalid.
    """
    try:
        def _f(val, default=0.0):
            try:
                v = float(val)
                return default if v != v else v   # NaN → default
            except Exception:
                return default

        daily_return    = np.clip(_f(row.get("daily_return")),   -1.0,  1.0)
        relative_volume = np.clip(_f(row.get("relative_volume")),  0.0, 100.0) / 100.0
        close           = _f(row.get("close"))
        high            = _f(row.get("high"))
        low             = _f(row.get("low"))
        high_pct        = (high - close) / close if close > 0 else 0.0
        low_pct         = (close - low)  / close if close > 0 else 0.0
        return_3d       = np.clip(_f(row.get("return_3d")),  -1.0, 1.0)
        range_10d       = np.clip(_f(row.get("range_10d")),   0.0, 2.0)

        return [
            float(daily_return),
            float(relative_volume),
            float(np.clip(high_pct, 0.0, 1.0)),
            float(np.clip(low_pct,  0.0, 1.0)),
            float(return_3d),
            float(range_10d),
        ]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build sequences from completed backfill examples
# ---------------------------------------------------------------------------

def build_sequences_from_backfill(all_examples: list) -> int:
    """
    For each labeled example from the backfill, fetch the 20-day feature
    sequence ending on scan_date and save to lstm_sequences.npz.

    Returns number of sequences successfully built.
    """
    import yfinance as yf
    from app.scanner import prepare_dataframe

    # Group examples by symbol to minimise yfinance downloads
    by_symbol: dict[str, list] = {}
    for ex in all_examples:
        sym = ex.get("symbol")
        ts  = ex.get("timestamp", "")[:10]
        if sym and ts:
            by_symbol.setdefault(sym, []).append((ts, ex.get("days_to_20pct")))

    X_list, y_list = [], []

    for symbol, entries in by_symbol.items():
        try:
            df = yf.download(symbol, period="max", interval="1d",
                             progress=False, auto_adjust=False)
            if df.empty or len(df) < SEQUENCE_LEN + 2:
                continue
            df = prepare_dataframe(df)

            # Build a date → row-index lookup
            date_index = {
                str(idx)[:10]: i
                for i, idx in enumerate(df.index)
            }

            for scan_date_str, days_to_20pct in entries:
                idx = date_index.get(scan_date_str)
                if idx is None or idx < SEQUENCE_LEN:
                    continue

                seq = []
                valid = True
                for j in range(idx - SEQUENCE_LEN, idx):
                    feats = extract_features(df.iloc[j])
                    if feats is None:
                        valid = False
                        break
                    seq.append(feats)

                if not valid or len(seq) != SEQUENCE_LEN:
                    continue

                X_list.append(seq)
                y_list.append(1 if days_to_20pct is not None else 0)

        except Exception as e:
            print(f"LSTM sequences: error on {symbol} — {e}")
            continue

    if not X_list:
        print("LSTM sequences: no sequences built — check backfill data")
        return 0

    X = np.array(X_list, dtype=np.float32)   # (N, 20, 6)
    y = np.array(y_list, dtype=np.float32)   # (N,)

    np.savez(SEQ_DATA_PATH, X=X, y=y)
    print(f"LSTM sequences: saved {len(X_list)} sequences "
          f"({int(y.sum())} positive / {len(y_list) - int(y.sum())} negative) "
          f"to {SEQ_DATA_PATH}")
    return len(X_list)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def _build_model():
    import torch.nn as nn
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=FEATURES, hidden=64, layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers,
                                batch_first=True, dropout=dropout)
            self.fc   = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return __import__("torch").sigmoid(self.fc(out[:, -1, :]))

    return LSTMPredictor()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(epochs: int = 30, lr: float = 0.001, batch_size: int = 32) -> dict:
    """
    Load lstm_sequences.npz, train the LSTM classifier, save lstm_model.pt.
    Returns {"accuracy": float, "samples": int, "epochs": int}.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    if not os.path.exists(SEQ_DATA_PATH):
        raise FileNotFoundError(f"{SEQ_DATA_PATH} not found — run backfill first")

    data = np.load(SEQ_DATA_PATH)
    X, y = data["X"], data["y"]

    if len(X) < 50:
        raise ValueError(f"Only {len(X)} sequences — need at least 50 to train")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val).unsqueeze(1))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model     = _build_model()
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

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = (model(xb) >= 0.5).float()
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  LSTM epoch {epoch}/{epochs} — "
                  f"train_loss={train_loss/len(train_dl):.4f}  "
                  f"val_acc={val_acc:.3f}")

    torch.save({
        "model_state":    model.state_dict(),
        "val_accuracy":   val_acc,
        "samples":        len(X),
        "trained_at":     datetime.utcnow().isoformat(),
        "sequence_len":   SEQUENCE_LEN,
        "features":       FEATURES,
    }, MODEL_PATH)

    print(f"LSTM: saved model to {MODEL_PATH}  "
          f"(val_accuracy={val_acc:.3f}, samples={len(X)})")
    return {"accuracy": round(val_acc, 4), "samples": len(X), "epochs": epochs}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_lstm():
    """Load and return trained model, or None if not available."""
    try:
        import torch
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = _build_model()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
    except Exception:
        return None


def predict_hit_probability(symbol: str) -> float | None:
    """
    Fetch the last 3 months of OHLCV for symbol, extract the most recent
    SEQUENCE_LEN-day window, run the LSTM, and return a 0–1 probability.
    Returns None on any error (model not trained, bad data, etc.).
    """
    try:
        import torch
        import yfinance as yf
        from app.scanner import prepare_dataframe

        model = load_lstm()
        if model is None:
            return None

        df = yf.download(symbol, period="3mo", interval="1d",
                         progress=False, auto_adjust=False)
        if df.empty or len(df) < SEQUENCE_LEN:
            return None

        df = prepare_dataframe(df)
        if len(df) < SEQUENCE_LEN:
            return None

        seq = []
        for j in range(len(df) - SEQUENCE_LEN, len(df)):
            feats = extract_features(df.iloc[j])
            if feats is None:
                return None
            seq.append(feats)

        x = torch.tensor([seq], dtype=torch.float32)   # (1, 20, 6)
        with torch.no_grad():
            prob = model(x).item()

        return round(prob, 4)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def get_lstm_status() -> dict:
    """Return metadata about the saved model, or {"trained": False}."""
    try:
        import torch
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        return {
            "trained":    True,
            "accuracy":   round(ckpt.get("val_accuracy", 0) * 100, 1),
            "samples":    ckpt.get("samples", 0),
            "trained_at": (ckpt.get("trained_at", "") or "")[:10],
        }
    except Exception:
        return {"trained": False}
