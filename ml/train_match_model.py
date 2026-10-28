"""
Train and evaluate match-result models with walk-forward validation.

Folds (per cahier des charges §6.1):
  F1: train [2020-21, 2021-22]                         -> test 2022-23
  F2: train [2020-21..2022-23]                         -> test 2023-24
  F3: train [2020-21..2023-24]                         -> test 2024-25
  F4: train [2020-21..2024-25]                         -> test 2025-26
  F5: train [2020-21..2025-26]                         -> test 2026-27 (played only)

Baselines compared against: always-home, frequency, bookmaker.
Models compared: LogisticRegression, RandomForest, XGBoost.
Selection criterion: log loss (not accuracy).
Final model: retrained on 2020-21..2025-26, isotonic-calibrated, saved to disk.

Usage:
    python -m ml.train_match_model
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, log_loss, recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml import features

from datetime import datetime, timezone

# --- Config -----------------------------------------------------------------
SEED = 42
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FOLDS = [
    (["2020-21", "2021-22"], "2022-23"),
    (["2020-21", "2021-22", "2022-23"], "2023-24"),
    (["2020-21", "2021-22", "2022-23", "2023-24"], "2024-25"),
    (["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"], "2025-26"),
    (["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"], "2026-27"),
]

FINAL_TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

log = logging.getLogger("train_match")


# --- Metrics ----------------------------------------------------------------
def brier_multi(y_true, proba, n_classes=3):
    y_onehot = np.zeros_like(proba)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def rps(y_true, proba, n_classes=3):
    """Ranked Probability Score (ordinal multiclass H < D < A)."""
    y_onehot = np.zeros_like(proba)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    p_cum = np.cumsum(proba, axis=1)
    y_cum = np.cumsum(y_onehot, axis=1)
    diffs = (p_cum[:, :-1] - y_cum[:, :-1]) ** 2
    return float(np.mean(np.sum(diffs, axis=1) / (n_classes - 1)))


def compute_metrics(y_true, proba):
    """Full metric bundle for one (model, fold) pair."""
    preds = np.argmax(proba, axis=1)
    rec = recall_score(y_true, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "brier": brier_multi(y_true, proba),
        "rps": rps(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist(),
        "recall_home": float(rec[0]),
        "recall_draw": float(rec[1]),
        "recall_away": float(rec[2]),
        "n": int(len(y_true)),
    }


# --- Model factory ----------------------------------------------------------
def make_logistic():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=SEED,
            # multi_class is unnecessary: lbfgs auto-uses multinomial for 3 classes
        )),
    ])


def make_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED,
        )),
    ])


def make_xgb():
    return Pipeline([
        # XGBoost handles NaN natively, no imputer needed
        ("clf", XGBClassifier(
            n_estimators=200, objective="multi:softprob", num_class=3,
            random_state=SEED, eval_metric="mlogloss", verbosity=0,
            tree_method="hist",
        )),
    ])


GRIDS = {
    "logistic": {"clf__C": [0.1, 1.0, 10.0]},
    "rf": {"clf__max_depth": [10, None], "clf__min_samples_leaf": [5, 10]},
    "xgboost": {"clf__max_depth": [4, 6], "clf__learning_rate": [0.05, 0.1]},
}

MODEL_FACTORIES = {
    "logistic": make_logistic,
    "rf": make_rf,
    "xgboost": make_xgb,
}


# --- Baselines --------------------------------------------------------------
def baseline_always_home(y_true):
    """Always predict class 0 (H). Log loss/Brier/RPS not meaningful — skipped."""
    preds = np.zeros_like(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": None, "brier": None, "rps": None,
        "n": int(len(y_true)),
    }


def baseline_frequency(y_train, y_true):
    """Predict train frequency vector. Class = argmax (typically H)."""
    counts = np.bincount(y_train, minlength=3)
    freq = counts / counts.sum()
    proba = np.tile(freq, (len(y_true), 1))
    return compute_metrics(y_true, proba)


def baseline_bookmaker(y_true, odds_df):
    """Bookmaker implied probabilities (b365). Skips rows with NaN odds."""
    valid = odds_df.notna().all(axis=1).values
    if valid.sum() == 0:
        return {"n": 0, "accuracy": None, "log_loss": None, "brier": None, "rps": None}
    proba = odds_df.loc[valid].values.astype(float)
    return compute_metrics(y_true[valid], proba)


# --- Training ---------------------------------------------------------------
def tune_and_fit(model_name, X_train, y_train):
    """Tune on train only via TimeSeriesSplit, return best fitted pipeline."""
    tscv = TimeSeriesSplit(n_splits=3)
    base = MODEL_FACTORIES[model_name]()
    grid = GRIDS[model_name]
    gs = GridSearchCV(base, grid, scoring="neg_log_loss", cv=tscv, n_jobs=1, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


# --- Main -------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading matches + features ...")
    matches = features.load_matches()
    feats, terminal_state = features.build_features(matches, return_state=True)
    log.info("  %d matches, %d features", len(feats), feats.shape[1])

    # Save terminal state for fast predict.py startup
    state_path = ARTIFACTS / "terminal_state.joblib"
    joblib.dump(terminal_state, state_path)
    log.info("  terminal_state.joblib saved (%d teams)", len(terminal_state["elo"]))

    # Load feature lists
    meta = json.loads((ARTIFACTS / "match_features.json").read_text(encoding="utf-8"))
    main_features = meta["features_main"]      # 55 features, no odds
    log.info("  main features: %d", len(main_features))

    # Missing xG columns -> fill with feature-specific NaN, XGB handles it
    feats = feats.sort_values("date").reset_index(drop=True)

    # ---- Walk-forward ----
    fold_results: dict[str, list] = {m: [] for m in MODEL_FACTORIES}
    baseline_results: dict[str, list] = {
        "always_home": [], "frequency": [], "bookmaker": [],
    }

    for fold_idx, (train_seasons, test_season) in enumerate(FOLDS, start=1):
        train = feats[feats["season"].isin(train_seasons)]
        test = feats[feats["season"] == test_season]
        if len(train) == 0 or len(test) == 0:
            log.warning("Skipping fold %d (empty train or test)", fold_idx)
            continue

        X_tr = train[main_features]
        y_tr = train["result_encoded"].values.astype(int)
        X_te = test[main_features]
        y_te = test["result_encoded"].values.astype(int)

        log.info("Fold %d: train=%d (%s), test=%d (%s)",
                 fold_idx, len(train), "+".join(train_seasons), len(test), test_season)

        # Baselines
        bh = baseline_always_home(y_te)
        bh.update({"fold": fold_idx, "test_season": test_season})
        baseline_results["always_home"].append(bh)

        bf = baseline_frequency(y_tr, y_te)
        bf.update({"fold": fold_idx, "test_season": test_season})
        baseline_results["frequency"].append(bf)

        odds_cols = ["b365_p_home", "b365_p_draw", "b365_p_away"]
        bb = baseline_bookmaker(y_te, test[odds_cols])
        bb.update({"fold": fold_idx, "test_season": test_season})
        baseline_results["bookmaker"].append(bb)

        # Models
        for name in MODEL_FACTORIES:
            t0 = time.time()
            model, params = tune_and_fit(name, X_tr, y_tr)
            proba = model.predict_proba(X_te)
            m = compute_metrics(y_te, proba)
            m.update({
                "fold": fold_idx,
                "test_season": test_season,
                "best_params": params,
                "train_time_s": round(time.time() - t0, 1),
            })
            fold_results[name].append(m)
            log.info("  %-9s acc=%.3f ll=%.4f brier=%.4f rps=%.4f (%.1fs)",
                     name, m["accuracy"], m["log_loss"], m["brier"], m["rps"],
                     m["train_time_s"])

    # ---- Aggregate (folds 1-4 main, fold 5 extra) ----
    def _agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}

    summary = {"baselines": {}, "models": {}}

    for bname, rows in baseline_results.items():
        main_rows = [r for r in rows if r["fold"] <= 4]
        extra_rows = [r for r in rows if r["fold"] == 5]
        summary["baselines"][bname] = {
            "folds_1_4": {k: _agg(main_rows, k) for k in ("accuracy", "log_loss", "brier", "rps")},
            "fold_5_extra": extra_rows[0] if extra_rows else None,
            "per_fold": rows,
        }

    for mname, rows in fold_results.items():
        main_rows = [r for r in rows if r["fold"] <= 4]
        extra_rows = [r for r in rows if r["fold"] == 5]
        summary["models"][mname] = {
            "folds_1_4": {k: _agg(main_rows, k) for k in ("accuracy", "log_loss", "brier", "rps")},
            "fold_5_extra": extra_rows[0] if extra_rows else None,
            "per_fold": rows,
        }

    # ---- Choose final model by mean log loss (folds 1-4) ----
    mean_ll = {m: summary["models"][m]["folds_1_4"]["log_loss"]["mean"] for m in MODEL_FACTORIES}
    mean_ll = {k: v for k, v in mean_ll.items() if v is not None}
    best_model_name = min(mean_ll, key=mean_ll.get)
    log.info("Best model by mean log loss: %s (%.4f)", best_model_name, mean_ll[best_model_name])

    # ---- Retrain best model on final training data ----
    log.info("Retraining %s on %s ...", best_model_name, FINAL_TRAIN_SEASONS)
    final_train = feats[feats["season"].isin(FINAL_TRAIN_SEASONS)]
    X_final = final_train[main_features]
    y_final = final_train["result_encoded"].values.astype(int)

    # Tune on final train (TimeSeriesSplit)
    final_pipeline, final_params = tune_and_fit(best_model_name, X_final, y_final)
    log.info("  final tuned params: %s", final_params)

    # Calibrate via TimeSeriesSplit-3 (keeps temporal order)
    log.info("  calibrating (sigmoid, TimeSeriesSplit-3) ...")
    calibrated = CalibratedClassifierCV(
        final_pipeline, method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )
    calibrated.fit(X_final, y_final)
    log.info("  calibration done")

    # ---- Save model + metrics + feature list ----
    model_path = ARTIFACTS / "match_model.joblib"
    joblib.dump(calibrated, model_path)
    log.info("  saved %s", model_path.name)

    # Update match_features.json with the model-specific info
    meta_out = dict(meta)
    meta_out["features_main_used"] = main_features
    meta_out["final_model"] = {
        "name": best_model_name,
        "calibration": "sigmoid (CalibratedClassifierCV cv=TimeSeriesSplit(3))",
        "training_seasons": FINAL_TRAIN_SEASONS,
        "n_train_rows": int(len(final_train)),
        "best_params": final_params,
        "trained_at": datetime.utcnow().isoformat(),
        "versions": {
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (ARTIFACTS / "match_features.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics_out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "seed": SEED,
            "folds": [{"train": t, "test": s} for t, s in FOLDS],
            "n_main_features": len(main_features),
        },
        "baselines": summary["baselines"],
        "models": summary["models"],
        "final_model": meta_out["final_model"],
    }
    (ARTIFACTS / "match_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("  saved match_metrics.json")

    # ---- Print summary ----
    print()
    print("=" * 70)
    print("WALK-FORWARD SUMMARY (folds 1-4, mean ± std)")
    print("=" * 70)
    for name in ["always_home", "frequency", "bookmaker"]:
        s = summary["baselines"][name]["folds_1_4"]
        ll = s["log_loss"]["mean"]
        ll_s = f"{ll:.4f}" if ll is not None else "—"
        acc = s["accuracy"]["mean"]
        acc_s = f"{acc:.4f}" if acc is not None else "—"
        print(f"  {name:<12} acc={acc_s}  ll={ll_s}")
    print()
    for name in ["logistic", "rf", "xgboost"]:
        s = summary["models"][name]["folds_1_4"]
        print(f"  {name:<12} acc={s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.3f}  "
              f"ll={s['log_loss']['mean']:.4f}±{s['log_loss']['std']:.3f}  "
              f"brier={s['brier']['mean']:.4f}  rps={s['rps']['mean']:.4f}")
    print()
    print(f"Best model: {best_model_name}  (ll={mean_ll[best_model_name]:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
    sys.exit(0)