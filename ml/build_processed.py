"""
Generate data/processed/*.csv from data/raw/.

Run with:
    python -m ml.build_processed
"""

from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd

from ml import data_loader, cleaning
from ml.mappings import SEASONS, PLAYER_SEASONS

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_processed")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # --- Matches (all 7 seasons) -----------------------------------------
    log.info("Building matches.csv ...")
    parts = []
    for s in SEASONS:
        raw = data_loader.load_e0(s)
        if raw is None:
            log.warning("E0 missing for %s", s)
            continue
        parts.append(cleaning.clean_e0(raw, s))
    if not parts:
        log.error("No E0 files loaded.")
        raise SystemExit(1)
    matches = pd.concat(parts, ignore_index=True)
    log.info("  %d matches total across %d seasons", len(matches), len(parts))

    # --- Understat --------------------------------------------------------
    log.info("Building understat.csv ...")
    und_parts = []
    for s in SEASONS:
        raw = data_loader.load_understat(s)
        if raw is None:
            log.info("  Understat missing for %s (expected)", s)
            continue
        und_parts.append(cleaning.clean_understat(raw, s))
    understat = pd.concat(und_parts, ignore_index=True) if und_parts else pd.DataFrame()
    log.info("  %d Understat rows", len(understat))

    matches = cleaning.attach_xg_to_matches(matches, understat)
    matches = matches.drop(columns=["date_only"])
    matches.to_csv(PROCESSED / "matches.csv", index=False)
    if not understat.empty:
        understat.drop(columns=["date_only"], errors="ignore").to_csv(
            PROCESSED / "understat.csv", index=False
        )

    # --- Load all history files (for global name fallback) ---------------
    log.info("Loading all history files ...")
    all_hist = {s: data_loader.load_player_history(s) for s in PLAYER_SEASONS}

    # --- Player gameweeks (6 seasons) — per-folder bridge ----------------
    log.info("Building player_gw.csv ...")
    gw_parts = []
    for s in PLAYER_SEASONS:
        raw_gw = data_loader.load_gameweeks(s)
        raw_pgw = data_loader.load_player_gw(s)
        raw_hist = data_loader.load_player_history(s)
        if raw_gw is None or raw_pgw is None or raw_hist is None:
            log.warning("Missing FPL data for %s", s)
            continue

        gw = cleaning.clean_gameweeks(raw_gw, s)
        bridge = cleaning.build_bridge_from_folder_names(raw_pgw, raw_hist)
        gw["element_code"] = gw["element"].map(bridge).astype("Int64")

        coverage = gw["element_code"].notna().mean()
        log.info("  %s: %d rows | bridge coverage %.1f%%", s, len(gw), coverage * 100)
        gw_parts.append(gw)

    if not gw_parts:
        log.error("No player_gw data loaded.")
        raise SystemExit(1)

    player_gw = pd.concat(gw_parts, ignore_index=True)

    # --- Global name fallback (pfn-prefix based) -------------------------
    log.info("Applying global name fallback ...")
    name_map = cleaning.build_global_name_to_code(all_hist)
    log.info("  global name map: %d unique names", len(name_map))

    unmatched_before = player_gw["element_code"].isna().sum()
    if unmatched_before:
        mask = player_gw["element_code"].isna()
        player_gw.loc[mask, "element_code"] = (
            player_gw.loc[mask, "name"]
            .apply(lambda n: name_map.get(cleaning.normalize_name(n)))
            .astype("Int64")
        )
        filled = (~player_gw.loc[mask, "element_code"].isna()).sum()
        log.info("  name fallback filled %d / %d unmatched rows (%.1f%%)",
                 filled, unmatched_before, 100.0 * filled / max(unmatched_before, 1))

    log.info("  overall element_code coverage: %.2f%%",
             100.0 * player_gw["element_code"].notna().mean())

    player_gw = cleaning.attach_match_key(player_gw)
    log.info("  match_key coverage %.1f%%",
             100.0 * player_gw["match_key"].notna().mean())
    player_gw.to_csv(PROCESSED / "player_gw.csv", index=False)

    # --- Player history (curated + derived for 2025-26) ------------------
    log.info("Building player_history.csv ...")
    latest = data_loader.load_player_history("2026-27")
    if latest is None:
        latest = data_loader.load_player_history("2025-26")
    if latest is None:
        log.error("No player history file (2026-27 or 2025-26).")
        raise SystemExit(1)

    curated = cleaning.clean_player_history(latest, keep_seasons=PLAYER_SEASONS)
    log.info("  curated: %d rows, %d players, seasons=%s",
             len(curated), curated["element_code"].nunique(),
             sorted(curated["season_name"].unique()))

    derived = cleaning.derive_season_aggregates_from_gw(player_gw)
    log.info("  derived from gw: %d rows, %d players, seasons=%s",
             len(derived), derived["element_code"].nunique() if not derived.empty else 0,
             sorted(derived["season_name"].unique()) if not derived.empty else [])

    history = (
        pd.concat([curated, derived], ignore_index=True)
        .drop_duplicates(subset=["element_code", "season_name"], keep="first")
        .sort_values(["season_name", "element_code"])
        .reset_index(drop=True)
    )
    name_lookup = (
        curated[curated["name"] != ""]
        .drop_duplicates("element_code")
        .set_index("element_code")["name"]
        .to_dict()
    )
    history["name"] = history.apply(
        lambda r: name_lookup.get(r["element_code"], "") if not r["name"] else r["name"],
        axis=1,
    )
    log.info("  merged: %d rows, %d players, %d seasons",
             len(history), history["element_code"].nunique(),
             history["season_name"].nunique())
    history.to_csv(PROCESSED / "player_history.csv", index=False)

    # --- Teams + Players (derived) --------------------------------------
    log.info("Building teams.csv / players.csv ...")
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    pd.DataFrame({"name": teams}).to_csv(PROCESSED / "teams.csv", index=False)

    latest_gw = (
        player_gw.dropna(subset=["element_code"])
        .sort_values(["season", "gameweek"])
        .groupby("element_code").tail(1)
    )
    players = pd.DataFrame({
        "element_code": latest_gw["element_code"],
        "name": latest_gw["name"],
        "position": latest_gw["position"],
        "current_team": latest_gw["team"],
    }).drop_duplicates("element_code")
    players.to_csv(PROCESSED / "players.csv", index=False)
    log.info("  teams: %d | players: %d", len(teams), len(players))

    log.info("Done. Files in %s", PROCESSED)


if __name__ == "__main__":
    main()