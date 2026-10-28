"""
One-shot fetch of FPL player birth dates.

The raw CSV exports don't include birth_date (Phase 1 finding). This script
pulls them from the public FPL bootstrap-static endpoint once and caches them
in data/raw/player_birthdates.csv.

Run once:
    python -m ml.fetch_player_birthdates

If offline or the API changes, the script exits gracefully and the season
model falls back to a documented proxy (n_prior_seasons <= 2) for "young".

Idempotent: re-run only if you want to refresh ages mid-season.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "player_birthdates.csv"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def main() -> None:
    print(f"Fetching {URL} ...")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 FIP-Academic/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"ERROR: {e}")
        print("Falling back to proxy criterion (n_prior_seasons <= 2).")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        if not OUT.exists():
            with OUT.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["element_code", "name", "web_name", "birth_date", "position"])
            print(f"Created empty {OUT.name} with header.")
        sys.exit(0)

    elements = data.get("elements", [])
    if not elements:
        print("WARNING: FPL API returned no elements.")
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["element_code", "name", "web_name", "birth_date", "position"])
        n = 0
        for p in elements:
            code = p.get("code")
            if not code:
                continue
            name = f"{p.get('first_name','')} {p.get('second_name','')}".strip()
            w.writerow([
                code,
                name,
                p.get("web_name", ""),
                p.get("birth_date", ""),
                POSITION_MAP.get(p.get("element_type"), ""),
            ])
            n += 1

    print(f"Saved {n} players to {OUT}")


if __name__ == "__main__":
    main()