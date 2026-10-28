"""
Train + walk-forward evaluate the player expected-points regressor.

Folds (same as match model minus 2026-27):
  F1: train [2020-21..2021-22] → test 2022-23
  F2: train [2020-21..2022-23] → test 2023-24
  F3: train [2020-21..2023-24] → test 2024-25
  F4: train [2020-21..2024-25] → test 2025-26

Baselines: predict last GW points · predict rolling-5 average.

Metrics: MAE, RMSE, Spearman.
Ranking: for each match, top-1 and top-3 "player of the match" hit rates.

Usage:
    python -m ml.train_player_model
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.features_player import FEATURES, load_and_build

SEED = 42
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FOLDS = [
    (["2020-21", "2021-22"], "2022-23"),
    (["2020-21", "2021-22", "2022-23"], "2023-24"),
    (["2020-21", "2021-22", "2022-23", "2023-24"], "2024-25"),
    (["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"], "2025-26"),
]
FINAL_TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

log = logging.getLogger("train_player")


# --- Baselines --------------------------------------------------------------
def baseline_last_gw(df: pd.DataFrame) -> np.ndarray:
    return df["total_points_shift1"].fillna(2.0).values


def baseline_rolling5(df: pd.DataFrame) -> np.ndarray:
    return df["total_points_roll5"].fillna(2.0).values


# --- Model factories --------------------------------------------------------
def make_ridge():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0, random_state=SEED)),
    ])


def make_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("reg", RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=SEED,
        )),
    ])

def make_xgb():
    return Pipeline([
        # XGBoost handles NaN natively — no imputer
        ("reg", XGBRegressor(
            n_estimators=100, objective="reg:squarederror",
            random_state=SEED, tree_method="hist", verbosity=0, n_jobs=-1,
        )),
    ])


GRIDS = {
    "ridge":   {"reg__alpha": [0.5, 1.0, 5.0]},
    "rf":      {"reg__max_depth": [15, None], "reg__min_samples_leaf": [10, 30]},
    "xgboost": {"reg__max_depth": [4, 6], "reg__learning_rate": [0.05, 0.1]},
}

MODEL_FACTORIES = {"ridge": make_ridge, "rf": make_rf, "xgboost": make_xgb}


# --- Metrics ----------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(rho) if not np.isnan(rho) else 0.0,
        "n": int(len(y_true)),
    }


def ranking_metrics(test_df: pd.DataFrame, y_pred: np.ndarray, top_n: int = 3) -> dict:
    """For each match in the test set, compute player-of-the-match hit rates.

    Candidate set: players with rolling minutes >= 45 (likely starters).
    For each match:
      - actual top-1 = player with the highest actual points
      - pred top-1   = highest predicted
      - pred top-3   = top 3 predicted
    Top-1 hit = actual_top_1 == pred_top_1.
    Top-3 hit = actual_top_1 in pred_top_3.
    """
    df = test_df.copy()
    df["_y_pred"] = y_pred

    top1_hits = 0
    top3_hits = 0
    matches = 0

    for _, group in df.groupby("match_key"):
        if group["match_key"].iloc[0] is None:
            continue
        candidates = group[(group["minutes"] > 0)
                           & (group["minutes_roll5"] >= 45)].copy()
        if len(candidates) < top_n:
            continue
        actual_top1 = candidates["total_points"].idxmax()
        pred_top1 = candidates["_y_pred"].idxmax()
        pred_top_n = candidates.nlargest(top_n, "_y_pred").index

        matches += 1
        if pred_top1 == actual_top1:
            top1_hits += 1
        if actual_top1 in pred_top_n:
            top3_hits += 1

    return {
        "matches_evaluated": matches,
        "top1_hit_rate": top1_hits / matches if matches else 0.0,
        "top3_hit_rate": top3_hits / matches if matches else 0.0,
    }


# --- Training helper --------------------------------------------------------
def tune_and_fit(model_name: str, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=2)
    base = MODEL_FACTORIES[model_name]()
    gs = GridSearchCV(base, GRIDS[model_name], scoring="neg_mean_absolute_error",
                      cv=tscv, n_jobs=1, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


# --- Main -------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading + building player features ...")
    df = load_and_build()
    log.info("  %d rows × %d cols | %d unique players | seasons: %s",
             len(df), df.shape[1], df["element_code"].nunique(),
             sorted(df["season"].unique()))

    # Save full feature table for analysis / debugging (not required by Django)
    df.to_csv(ARTIFACTS / "player_features_full.csv", index=False)
    log.info("  saved player_features_full.csv")

    fold_results = {m: [] for m in MODEL_FACTORIES}
    baseline_results = {"last_gw": [], "rolling5": []}

    for fold_idx, (train_seasons, test_season) in enumerate(FOLDS, start=1):
        train = df[df["season"].isin(train_seasons)]
        test = df[df["season"] == test_season]
        if len(train) == 0 or len(test) == 0:
            log.warning("Skipping fold %d (empty)", fold_idx)
            continue

        X_tr = train[FEATURES]
        y_tr = train["total_points"].values.astype(float)
        X_te = test[FEATURES]
        y_te = test["total_points"].values.astype(float)

        log.info("Fold %d: train=%d, test=%d (%s)",
                 fold_idx, len(train), len(test), test_season)

                # Baselines (now with ranking metrics for honest ML-vs-baseline)
        for bname, fn in [("last_gw", baseline_last_gw),
                          ("rolling5", baseline_rolling5)]:
            preds = fn(test)
            m = regression_metrics(y_te, preds)
            r = ranking_metrics(test.reset_index(drop=True), preds)
            m.update({"fold": fold_idx, "test_season": test_season, "ranking": r})
            baseline_results[bname].append(m)
            log.info("  %-9s mae=%.3f rmse=%.3f rho=%.3f top1=%.3f top3=%.3f",
                     bname, m["mae"], m["rmse"], m["spearman"],
                     r["top1_hit_rate"], r["top3_hit_rate"])

        # Models
        for name in MODEL_FACTORIES:
            t0 = time.time()
            model, params = tune_and_fit(name, X_tr, y_tr)
            preds = model.predict(X_te)
            m = regression_metrics(y_te, preds)
            r = ranking_metrics(test.reset_index(drop=True), preds)
            m.update({
                "fold": fold_idx,
                "test_season": test_season,
                "ranking": r,
                "best_params": params,
                "train_time_s": round(time.time() - t0, 1),
            })
            fold_results[name].append(m)
            log.info("  %-9s mae=%.3f rmse=%.3f rho=%.3f top1=%.3f top3=%.3f (%.1fs)",
                     name, m["mae"], m["rmse"], m["spearman"],
                     r["top1_hit_rate"], r["top3_hit_rate"], m["train_time_s"])

    # --- Aggregate ---
    def _agg(rows, key):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    summary = {"baselines": {}, "models": {}}
    for bname, rows in baseline_results.items():
        summary["baselines"][bname] = {
            k: _agg(rows, k) for k in ("mae", "rmse", "spearman")
        }
        top1_vals = [r["ranking"]["top1_hit_rate"] for r in rows if "ranking" in r]
        top3_vals = [r["ranking"]["top3_hit_rate"] for r in rows if "ranking" in r]
        summary["baselines"][bname]["ranking"] = {
            "top1_mean": float(np.mean(top1_vals)) if top1_vals else 0.0,
            "top3_mean": float(np.mean(top3_vals)) if top3_vals else 0.0,
        }
        summary["baselines"][bname]["per_fold"] = rows

    for mname, rows in fold_results.items():
        summary["models"][mname] = {
            k: _agg(rows, k) for k in ("mae", "rmse", "spearman")
        }
        # Aggregate ranking metrics across folds
        top1_vals = [r["ranking"]["top1_hit_rate"] for r in rows]
        top3_vals = [r["ranking"]["top3_hit_rate"] for r in rows]
        summary["models"][mname]["ranking"] = {
            "top1_mean": float(np.mean(top1_vals)) if top1_vals else 0.0,
            "top3_mean": float(np.mean(top3_vals)) if top3_vals else 0.0,
        }
        summary["models"][mname]["per_fold"] = rows

    # --- Choose best by mean Spearman (ranking is the use case) ---------
    mean_rho = {m: summary["models"][m]["spearman"]["mean"]
                for m in MODEL_FACTORIES}
    mean_rho = {k: v for k, v in mean_rho.items() if v is not None}
    best_name = max(mean_rho, key=mean_rho.get)
    log.info("Best model by mean Spearman: %s (%.4f)", best_name, mean_rho[best_name])

    # --- Retrain on all data ---
    log.info("Retraining %s on all seasons ...", best_name)
    final_train = df[df["season"].isin(FINAL_TRAIN_SEASONS)]
    X_final = final_train[FEATURES]
    y_final = final_train["total_points"].values.astype(float)
    final_model, final_params = tune_and_fit(best_name, X_final, y_final)
    log.info("  final params: %s", final_params)

    # Save model
    joblib.dump(final_model, ARTIFACTS / "player_model.joblib")
    log.info("  saved player_model.joblib")

    # Save feature metadata
    meta = {
        "features": FEATURES,
        "target": "total_points",
        "final_model": {
            "name": best_name,
            "best_params": final_params,
            "training_seasons": FINAL_TRAIN_SEASONS,
            "n_train_rows": int(len(final_train)),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "versions": {
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        },
        "baselines": {
            "last_gameweek": "predict total_points_shift1 (fallback 2.0 if NaN)",
            "rolling_5": "predict total_points_roll5 (fallback 2.0 if NaN)",
        },
    }
    (ARTIFACTS / "player_features.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save metrics
    metrics_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(FEATURES),
        "folds": [{"train": t, "test": s} for t, s in FOLDS],
        "baselines": summary["baselines"],
        "models": summary["models"],
        "final_model": meta["final_model"],
    }
    (ARTIFACTS / "player_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Print summary --------------------------------------------------
    print()
    print("=" * 78)
    print("WALK-FORWARD SUMMARY — Player model (mean ± std over folds 1-4)")
    print("=" * 78)
    for bname in ("last_gw", "rolling5"):
        s = summary["baselines"][bname]
        r = s.get("ranking", {})
        print(f"  {bname:<10} mae={s['mae']['mean']:.3f}  "
              f"rmse={s['rmse']['mean']:.3f}  rho={s['spearman']['mean']:.3f}  "
              f"top1={r.get('top1_mean', 0):.3f}  top3={r.get('top3_mean', 0):.3f}")
    print()
    for name in ("ridge", "rf", "xgboost"):
        s = summary["models"][name]
        r = s["ranking"]
        print(f"  {name:<10} mae={s['mae']['mean']:.3f}±{s['mae']['std']:.3f}  "
              f"rho={s['spearman']['mean']:.3f}  "
              f"top1={r['top1_mean']:.3f}  top3={r['top3_mean']:.3f}")
    print()
    print(f"Best model: {best_name}  (rho={mean_rho[best_name]:.4f})")
    print("=" * 78)


if __name__ == "__main__":
    main()
    sys.exit(0)