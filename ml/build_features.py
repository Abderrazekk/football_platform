"""
Build the match feature table.

Usage:
    python -m ml.build_features
"""

from __future__ import annotations

import logging
import sys

from ml import features


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("build_features")

    log.info("Loading matches ...")
    matches = features.load_matches()
    log.info("  %d matches loaded", len(matches))

    log.info("Building features (chronological) ...")
    feats = features.build_features(matches)
    log.info("  features table: %d rows × %d cols", len(feats), feats.shape[1])

    # Summary stats
    n_cold = int(feats["cold_start"].sum())
    n_nan_xg = int(feats["home_xg_for_5"].isna().sum())
    log.info("  cold_start matches (either team < 3 prior): %d", n_cold)
    log.info("  rows with NaN home_xg_for_5: %d", n_nan_xg)
    log.info("  seasons present: %s", sorted(feats["season"].unique()))

    log.info("Saving artifacts ...")
    features.save_artifacts(feats)
    log.info("Done.")


if __name__ == "__main__":
    main()
    sys.exit(0)