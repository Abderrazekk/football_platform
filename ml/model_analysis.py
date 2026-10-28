"""
Post-training analysis: permutation feature importance + calibration curve.

Loads the saved match_model.joblib and produces:
  - ml/artifacts/match_feature_importance.json  (top-20 features by importance)
  - ml/artifacts/match_calibration_curve.png    (reliability diagram)

Usage:
    python -m ml.model_analysis
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

from ml import features

log = logging.getLogger(__name__)
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading model + features ...")
    model_path = ARTIFACTS / "match_model.joblib"
    if not model_path.exists():
        log.error("Missing %s. Run `python -m ml.train_match_model` first.", model_path)
        sys.exit(1)
    model = joblib.load(model_path)

    meta = json.loads((ARTIFACTS / "match_features.json").read_text(encoding="utf-8"))
    feature_names = meta["features_main_used"]
    train_seasons = meta["final_model"]["training_seasons"]

    matches = features.load_matches()
    feats = features.build_features(matches)
    train_df = feats[feats["season"].isin(train_seasons)].sort_values("date")
    X = train_df[feature_names]
    y = train_df["result_encoded"].values.astype(int)
    log.info("  training rows: %d, features: %d", len(X), len(feature_names))

    # ---- Permutation importance (3 repeats for stability) ----
    log.info("Computing permutation importance (10 repeats) ...")
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, scoring="neg_log_loss", n_jobs=1,
    )
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp_df["rank"] = range(1, len(imp_df) + 1)
    imp_df.to_csv(ARTIFACTS / "match_feature_importance.csv", index=False)

    top20 = imp_df.head(20).to_dict(orient="records")
    (ARTIFACTS / "match_feature_importance.json").write_text(
        json.dumps({
            "method": "permutation_importance (n_repeats=10, scoring=neg_log_loss)",
            "top_20": top20,
            "all_features_ranked": imp_df.to_dict(orient="records"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("  Saved match_feature_importance.json")
    log.info("  Top 5 features:")
    for r in top20[:5]:
        log.info("    %2d. %-28s  %+.4f ± %.4f",
                 r["rank"], r["feature"], r["importance_mean"], r["importance_std"])

    # ---- Calibration curve (using held-out fold 4 test set) ----
    log.info("Building calibration curve on 2025-26 hold-out ...")
    test_df = feats[feats["season"] == "2025-26"]
    if len(test_df) == 0:
        log.warning("No 2025-26 test data; skipping calibration plot")
        return
    X_te = test_df[feature_names]
    y_te = test_df["result_encoded"].values.astype(int)

    proba = model.predict_proba(X_te)

    # One-vs-rest calibration per class
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    colors = ["#1f77b4", "#888888", "#d62728"]
    labels = ["Home (H)", "Draw (D)", "Away (A)"]
    for k in range(3):
        y_bin = (y_te == k).astype(int)
        p_k = proba[:, k]
        frac_pos, mean_pred = calibration_curve(y_bin, p_k, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_bin, p_k)
        ax.plot(mean_pred, frac_pos, marker="o", color=colors[k],
                label=f"{labels[k]} (Brier {brier:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (out-of-sample: 2025-26)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = ARTIFACTS / "match_calibration_curve.png"
    plt.savefig(out_png, dpi=110)
    plt.close(fig)
    log.info("  Saved %s", out_png.name)

    # Metrics for the calibration JSON
    test_metrics = {
        "test_season": "2025-26",
        "n": int(len(y_te)),
        "log_loss": float(log_loss(y_te, proba, labels=[0, 1, 2])),
        "brier_home": float(brier_score_loss((y_te == 0).astype(int), proba[:, 0])),
        "brier_draw": float(brier_score_loss((y_te == 1).astype(int), proba[:, 1])),
        "brier_away": float(brier_score_loss((y_te == 2).astype(int), proba[:, 2])),
    }
    (ARTIFACTS / "match_calibration.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    log.info("  Saved match_calibration.json")
    log.info("Done.")


if __name__ == "__main__":
    main()
    sys.exit(0)