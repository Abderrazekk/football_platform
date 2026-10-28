"""
Train + walk-forward evaluate the season-points regressor.

Predicts a player's total FPL points for a whole season, using only
prior-season features. Validates on seasons 2022-23 → 2025-26.

Produces:
  - ml/artifacts/season_model.joblib
  - ml/artifacts/season_predictions_2026_27.csv     (raw predictions)
  - ml/artifacts/season_predictions_2026_27.json    (all Django-ready payloads)
  - ml/artifacts/season_metrics.json

Usage:
    python -m ml.train_season_model
"""

from __future__ import annotations

import json
import logging
import math
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml import features_season
from ml.optimize_squad import optimize_squad

SEED = 42
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

TARGET_SEASON = "2026-27"
VALIDATION_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
TRAIN_SEASONS_ALL = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

log = logging.getLogger("train_season")


# --- Models -----------------------------------------------------------------
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
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            n_jobs=-1, random_state=SEED,
        )),
    ])


def make_xgb():
    return Pipeline([
        ("reg", XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective="reg:squarederror", random_state=SEED,
            tree_method="hist", verbosity=0, n_jobs=-1,
        )),
    ])


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


def top10_overlap(df: pd.DataFrame, pred_col: str, k: int = 10) -> float:
    """Overlap between predicted top-k and actual top-k."""
    if len(df) < k:
        return 0.0
    pred_top = set(df.nlargest(k, pred_col)["element_code"])
    true_top = set(df.nlargest(k, "target_points")["element_code"])
    return len(pred_top & true_top) / k


# --- Main -------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading data ...")
    history = features_season.load_history()
    pgw = pd.read_csv(DATA_PROCESSED / "player_gw.csv")
    mf = pd.read_csv(DATA_PROCESSED / "match_features.csv", parse_dates=["date"])
    birthdates = features_season.load_birthdates()

    meta = features_season.build_meta_from_pgw(pgw)
    team_elo = features_season.build_team_elo(mf)

    log.info("  history: %d rows, %d players, %d seasons",
             len(history), history["element_code"].nunique(), history["season_name"].nunique())
    log.info("  birthdates: %s",
             "loaded" if birthdates is not None else "not available (fallback to proxy)")

    log.info("Building training table ...")
    train_df = features_season.build_training_table(history, meta, team_elo, birthdates)
    log.info("  %d rows × %d cols", len(train_df), train_df.shape[1])

    features = features_season.FEATURES_SEASON

    # ---- Walk-forward ----
    fold_results = {m: [] for m in MODEL_FACTORIES}

    for fold_idx, test_season in enumerate(VALIDATION_SEASONS, start=1):
        train = train_df[train_df["target_season"] < test_season]
        test = train_df[train_df["target_season"] == test_season]
        if len(train) < 50 or len(test) < 20:
            log.warning("Skipping fold %d (train=%d, test=%d)", fold_idx, len(train), len(test))
            continue

        X_tr = train[features]
        y_tr = train["target_points"].values.astype(float)
        X_te = test[features]
        y_te = test["target_points"].values.astype(float)

        log.info("Fold %d: train=%d, test=%d (%s)", fold_idx, len(train), len(test), test_season)

        for name in MODEL_FACTORIES:
            t0 = time.time()
            model = MODEL_FACTORIES[name]()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            test_eval = test.reset_index(drop=True).copy()
            test_eval["_pred"] = preds
            overlap = top10_overlap(test_eval, "_pred", k=10)

            m = regression_metrics(y_te, preds)
            m.update({
                "fold": fold_idx,
                "test_season": test_season,
                "top10_overlap": round(overlap, 3),
                "train_time_s": round(time.time() - t0, 1),
            })
            fold_results[name].append(m)
            log.info("  %-9s mae=%.1f rmse=%.1f rho=%.3f top10=%.2f (%.1fs)",
                     name, m["mae"], m["rmse"], m["spearman"], overlap, m["train_time_s"])

    # ---- Choose best by mean Spearman ----
    def _mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    mean_rho = {m: _mean(rows, "spearman") for m, rows in fold_results.items()}
    best_name = max(mean_rho, key=mean_rho.get)
    log.info("Best model by mean Spearman: %s (%.4f)", best_name, mean_rho[best_name])

    # ---- Retrain on all data ----
    log.info("Retraining %s on all seasons ...", best_name)
    final_train = train_df[train_df["target_season"].isin(VALIDATION_SEASONS + ["2025-26"])].copy()
    # Include all validation-target rows so we use max data
    X_final = final_train[features]
    y_final = final_train["target_points"].values.astype(float)
    final_model = MODEL_FACTORIES[best_name]()
    final_model.fit(X_final, y_final)
    log.info("  trained on %d rows", len(final_train))

    joblib.dump(final_model, ARTIFACTS / "season_model.joblib")

    # ---- Predict 2026-27 ----
    log.info("Predicting %s ...", TARGET_SEASON)
    pred_df = features_season.build_prediction_table(history, meta, team_elo, birthdates, TARGET_SEASON)
    log.info("  %d eligible players", len(pred_df))

    X_pred = pred_df[features]
    raw_pred = final_model.predict(X_pred)
    pred_df["predicted_points"] = np.clip(raw_pred, 0.0, None)

    # Uncertainty from last validation fold
    last_fold = fold_results[best_name][-1] if fold_results[best_name] else {"rmse": 30.0}
    uncertainty_95 = round(1.96 * last_fold["rmse"], 1)
    pred_df["uncertainty"] = uncertainty_95

    # Rank
    pred_df = pred_df.sort_values("predicted_points", ascending=False).reset_index(drop=True)

    # Save raw CSV for optimizer + inspection
    out_cols = [
        "element_code", "name", "prev_position", "prev_team",
        "predicted_points", "uncertainty", "prev_price_end", "age", "n_prior_seasons",
    ]
    pred_df[out_cols].to_csv(ARTIFACTS / "season_predictions_2026_27.csv", index=False)

    # ---- Best player top 10 ----
    top_players = (
        pred_df.nlargest(10, "predicted_points")[out_cols]
        .rename(columns={"prev_position": "position", "prev_team": "team", "prev_price_end": "price"})
        .to_dict("records")
    )

    # ---- Best young top 10 ----
    has_real_age = pred_df["age"].notna().sum() >= 30  # at least ~30 players with real ages
    if has_real_age:
        young_df = pred_df[pred_df["age"] < 22].copy()
        young_criterion = "age < 22 (from FPL API)"
    else:
        young_df = pred_df[pred_df["n_prior_seasons"] <= 2].copy()
        young_criterion = "proxy: n_prior_seasons <= 2 (age unavailable)"

    young_top = (
        young_df.nlargest(10, "predicted_points")[out_cols]
        .rename(columns={"prev_position": "position", "prev_team": "team", "prev_price_end": "price"})
        .to_dict("records")
    ) if not young_df.empty else []

    # ---- Best XI ----
    log.info("Optimizing best squad ...")
    xi_result = optimize_squad(pred_df)
    
    
    # ---- Build payload JSON ----
    payload = {
        "season": TARGET_SEASON,
        "model_version": f"{best_name}-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "uncertainty_95": uncertainty_95,
        "has_real_age": bool(has_real_age),
        "young_criterion": young_criterion,
        "best_player": {
            "top_10": top_players,
            "uncertainty_95": uncertainty_95,
        },
        "best_young": {
            "top_10": young_top,
            "criterion": young_criterion,
            "uncertainty_95": uncertainty_95,
        },
        "best_xi": xi_result,
        "metrics": {
            "model": best_name,
            "mean_spearman": mean_rho[best_name],
            "per_fold": fold_results[best_name],
        },
    }
    def _san(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _san(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_san(v) for v in obj]
        return obj

    payload = _san(payload)
    (ARTIFACTS / "season_predictions_2026_27.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- Metrics file ----
    metrics_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(features),
        "folds": VALIDATION_SEASONS,
        "models": {
            name: {
                "mean_mae": float(np.mean([r["mae"] for r in rows])),
                "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
                "mean_spearman": float(np.mean([r["spearman"] for r in rows])),
                "mean_top10_overlap": float(np.mean([r["top10_overlap"] for r in rows])),
                "per_fold": rows,
            } for name, rows in fold_results.items()
        },
        "final_model": best_name,
        "versions": {
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (ARTIFACTS / "season_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- Print summary ----
    print()
    print("=" * 78)
    print("WALK-FORWARD SUMMARY — Season model")
    print("=" * 78)
    for name in ("ridge", "rf", "xgboost"):
        rows = fold_results[name]
        if not rows:
            continue
        print(f"  {name:<9} mae={np.mean([r['mae'] for r in rows]):6.1f}  "
              f"rmse={np.mean([r['rmse'] for r in rows]):6.1f}  "
              f"rho={np.mean([r['spearman'] for r in rows]):.3f}  "
              f"top10={np.mean([r['top10_overlap'] for r in rows]):.2f}")
    print(f"\nBest model: {best_name}  (mean rho={mean_rho[best_name]:.4f})")
    print(f"Uncertainty (±95%): {uncertainty_95} pts")
    print(f"Young criterion: {young_criterion}")
    print(f"Files saved in {ARTIFACTS}")
    print("=" * 78)


if __name__ == "__main__":
    main()
    sys.exit(0)