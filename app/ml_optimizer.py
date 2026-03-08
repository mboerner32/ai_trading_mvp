"""
ML weight optimizer — gradient-boosted classifier trained on signals_json labels.

Replaces Claude's weight guessing once 500+ labeled live scans accumulate.
Features: 23 binary signal flags from signals_json.
Label:    days_to_20pct IS NOT NULL (stock hit +20% intraday within 7 days).

Auto-transition logic:
  get_labeled_count() >= XGB_THRESHOLD  →  train_xgb_weights() returns a weights dict
  get_labeled_count() <  XGB_THRESHOLD  →  train_xgb_weights() returns None (Claude used instead)

The returned weights dict is a drop-in replacement for DEFAULT_SQUEEZE_WEIGHTS, scaled
so the total point budget matches the defaults (~187 pts across all non-optional keys).
"""

import json
import os
import pickle
import sqlite3

from app.database import DB_NAME
from app.scoring_engine import DEFAULT_SQUEEZE_WEIGHTS

XGB_THRESHOLD = 500          # minimum labeled rows before activating
MODEL_PATH    = "ml_model.pkl"

# Signal keys used as features — must match _ALL_SIGNAL_KEYS in database.py
_SIGNAL_KEYS = [
    "rel_vol_50x", "rel_vol_25x", "rel_vol_10x", "rel_vol_5x",
    "daily_sweet_20_40", "daily_ok_10_20", "daily_ok_40_100",
    "sideways_chop", "yesterday_green",
    "shares_lt10m", "shares_lt30m", "shares_lt100m",
    "no_news_bonus", "high_cash_bonus",
    "institution_moderate", "institution_strong", "sector_biotech_bonus",
    "rsi_momentum_bonus", "macd_positive_bonus", "bb_upper_breakout",
    "consecutive_green_bonus", "low_float_ratio_bonus",
    "first_hour_vol_20m",
]

# Keys eligible for weight output (subset of _SIGNAL_KEYS that exist in DEFAULT_SQUEEZE_WEIGHTS)
_WEIGHT_KEYS = [k for k in _SIGNAL_KEYS if k in DEFAULT_SQUEEZE_WEIGHTS]
# Total point budget = sum of all default weights
_DEFAULT_BUDGET = sum(DEFAULT_SQUEEZE_WEIGHTS.values())


def get_labeled_count() -> int:
    """Count live scans with both signals_json and next_day_return (excluding historical mode)."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM scans
            WHERE signals_json IS NOT NULL
              AND next_day_return IS NOT NULL
              AND mode != 'historical'
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def _load_training_data():
    """
    Load (X, y) training matrices from live scans.
    X: (N, 23) binary feature matrix parsed from signals_json.
    y: (N,) binary labels — 1 if days_to_20pct IS NOT NULL, else 0.
    Returns (X, y, feature_names) or (None, None, None) on failure.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT signals_json, days_to_20pct FROM scans
            WHERE signals_json IS NOT NULL
              AND next_day_return IS NOT NULL
              AND mode != 'historical'
            ORDER BY id DESC LIMIT 2000
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None, None, None

        X, y = [], []
        for signals_json_str, days_to_20pct in rows:
            try:
                signals = json.loads(signals_json_str) if signals_json_str else {}
            except Exception:
                signals = {}
            features = [1 if signals.get(k) else 0 for k in _SIGNAL_KEYS]
            X.append(features)
            y.append(1 if days_to_20pct is not None else 0)

        return X, y, _SIGNAL_KEYS

    except Exception:
        return None, None, None


def train_xgb_weights(threshold: int = XGB_THRESHOLD) -> dict | None:
    """
    Train a gradient-boosted classifier on signals_json data.
    Returns a weights dict (drop-in for DEFAULT_SQUEEZE_WEIGHTS) if training succeeds
    and labeled count >= threshold. Returns None otherwise.

    The returned weights are scaled so their sum equals _DEFAULT_BUDGET,
    preserving the total point budget the scoring engine expects.
    Saves trained model to ml_model.pkl for status inspection.
    """
    count = get_labeled_count()
    if count < threshold:
        return None

    X, y, feature_names = _load_training_data()
    if X is None or len(X) < threshold:
        return None

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        # Save model + metadata
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":          model,
                "feature_names":  feature_names,
                "val_accuracy":   val_acc,
                "labeled_count":  count,
            }, f)

        print(f"ML OPTIMIZER: trained on {len(X)} samples — val_acc={val_acc:.3f}")

        # Map feature importances → weights
        importances = dict(zip(feature_names, model.feature_importances_))

        # Sum importances only for weight-eligible keys
        weight_importance_total = sum(importances.get(k, 0.0) for k in _WEIGHT_KEYS)
        if weight_importance_total == 0:
            return None

        # Scale to match default point budget
        weights = {}
        for key in DEFAULT_SQUEEZE_WEIGHTS:
            imp = importances.get(key, 0.0)
            raw = (imp / weight_importance_total) * _DEFAULT_BUDGET
            weights[key] = max(0, round(raw))

        return weights

    except Exception as e:
        print(f"ML OPTIMIZER: training failed — {e}")
        return None


def get_xgb_status() -> dict:
    """
    Return status dict for analytics display:
    {
        "labeled_count": int,
        "threshold":     int,
        "active":        bool,    # True if model is trained and file exists
        "val_accuracy":  float|None,
        "pct_ready":     float,   # 0–100 progress toward threshold
    }
    """
    count = get_labeled_count()
    pct   = min(round(count / XGB_THRESHOLD * 100, 1), 100.0)

    if not os.path.exists(MODEL_PATH):
        return {
            "labeled_count": count,
            "threshold":     XGB_THRESHOLD,
            "active":        False,
            "val_accuracy":  None,
            "pct_ready":     pct,
        }

    try:
        with open(MODEL_PATH, "rb") as f:
            meta = pickle.load(f)
        return {
            "labeled_count": count,
            "threshold":     XGB_THRESHOLD,
            "active":        True,
            "val_accuracy":  round(meta.get("val_accuracy", 0) * 100, 1),
            "pct_ready":     pct,
        }
    except Exception:
        return {
            "labeled_count": count,
            "threshold":     XGB_THRESHOLD,
            "active":        False,
            "val_accuracy":  None,
            "pct_ready":     pct,
        }
