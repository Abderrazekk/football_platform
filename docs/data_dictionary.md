# Data Dictionary — Football Intelligence Platform

Status: **Phase 1 (exploration)** — v1.0
Last updated: Phase 1 completion
Confidence: high for E0, FPL and Understat; **`master_xP_projections.id`
still unconfirmed** (see Open question Q3).

Column-level documentation for every file the project consumes.
All files live under `data/raw/<season>/` where `<season>` is `YYYY-YY`.

---

## 1. `E0_Season_YYYYYYYY.csv` — match results + bookmaker odds

One row per Premier League match. Source: football-data.co.uk, **renamed
columns** (not the original `B365H` / `FTHG` codes).

| Column | Meaning | Type | Used for |
|---|---|---|---|
| `Division` | Always `E0` (Premier League) | str | sanity check |
| `Match_Date` | Match date, **`DD/MM/YYYY`** | str → datetime (dayfirst=True) | temporal split, H2H, rolling form |
| `Kickoff_Time` | Local kickoff `HH:MM`, no tz | str | secondary |
| `Home_Team` | Home club name (short form, e.g. `Man United`) | str | target setup, joins |
| `Away_Team` | Away club name (short form) | str | target setup, joins |
| `Full_Time_Home_Goals` | Final home goals | int | **target**, xG fallback |
| `Full_Time_Away_Goals` | Final away goals | int | **target**, xG fallback |
| `Full_Time_Result` | **`H` / `D` / `A`** | str | **target variable (y)** |
| `Half_Time_Home_Goals` | HT home goals | int | optional feature |
| `Half_Time_Away_Goals` | HT away goals | int | optional feature |
| `Half_Time_Result` | HT result | str | optional feature |
| `Referee` | Referee name | str | optional |
| `Home_Shots`, `Away_Shots` | Total shots | int | post-match stats (never a feature) |
| `Home_Shots_On_Target`, `Away_Shots_On_Target` | SoT | int | post-match stats |
| `Home_Fouls`, `Away_Fouls` | Fouls | int | post-match stats |
| `Home_Corners`, `Away_Corners` | Corners | int | post-match stats |
| `Home_Yellow_Cards`, `Away_Yellow_Cards` | YC | int | post-match stats |
| `Home_Red_Cards`, `Away_Red_Cards` | RC | int | post-match stats |
| `Bet365_*`, `Betway_*`, `Pinnacle_*`, `WH*`, `VC*`, `IWH*`, `IW*` | **Opening** odds (H / D / A) per bookmaker | float | baseline 3 (bookmaker implied), optional feature (separate experiment) |
| `Max_Market_*`, `Avg_Market_*` | Max / average opening odds across market | float | stronger baseline |
| `Bet365_Over_2.5_Goals`, `Bet365_Under_2.5_Goals`, `Pinnacle_*`, `Max_*`, `Avg_*` | O/U 2.5 opening odds | float | secondary experiment |
| `Asian_Handicap_Size`, `*_Asian_Handicap_*` | Asian handicap opening lines | float | unused |
| `*_Closing_*` | **Closing** odds (same bookmakers) | float | stronger baseline if needed |

**Rule:** odds are used **only as a benchmark**, never as features in the
main model (cahier des charges §6.2). The main model must beat baselines
without seeing the market.

Post-match statistics (`*_Shots`, `*_Fouls`, `*_Cards`, `*_Goals`) describe
the match we are trying to predict → **never features**. They are only used
to compute *rolling* statistics of **past** matches.

---

## 2. `master_gameweeks.csv` — FPL per-player-per-gameweek rows

One row per player per gameweek. This is **FPL data enriched with
`name / position / team / xP`**, NOT a fixture calendar (contra cahier des
charges §4.1, which describes it as `dates, échéances, scores moyens`).

| Column | Meaning | Type | Used for |
|---|---|---|---|
| `name` | Player display name (`"Aaron Connolly"`) | str | display |
| `position` | `GK` / `DEF` / `MID` / `FWD` | str | player features |
| `team` | Club name, **string** (`"Brighton"`) | str | features |
| `xP` | Expected FPL points (pre-match projection) | float | **extra feature**, but leak risk → verify availability time |
| `assists` | Assists in this GW | int | player model target components |
| `bonus` | FPL bonus points | int | player features |
| `bps` | Bonus Points System score | int | player features |
| `clean_sheets` | Clean-sheet flag (0/1) | int | player features |
| `creativity` | FPL creativity metric (ICT) | float | player features |
| `element` | **Per-season FPL player ID** (unstable across seasons) | int | joins inside one season only |
| `fixture` | FPL fixture ID (per-season) | int | joins inside one season only |
| `goals_conceded`, `goals_scored` | Counters | int | player features (rolling) |
| `ict_index` | ICT index (sum of influence/creativity/threat) | float | player features |
| `influence` | FPL influence metric | float | player features |
| `kickoff_time` | ISO-8601 **with `Z`** (`2020-09-14T19:15:00Z`) | str → datetime (UTC) | temporal ordering |
| `minutes` | Minutes played | int | player features, filter |
| `opponent_team` | **FPL numeric team ID** (e.g. `5`) | int | join to team mapping |
| `own_goals`, `penalties_missed`, `penalties_saved` | Incident counts | int | player features |
| `red_cards`, `yellow_cards` | Cards | int | player features |
| `round` | Gameweek number (1..38) | int | alignment with E0 |
| `saves` | Saves (GK) | int | GK features |
| `selected` | FPL ownership (transfers count) | int | unused |
| `team_a_score`, `team_h_score` | Goals in this fixture (FPL view) | int | cross-check with E0 |
| `threat` | FPL threat metric | float | player features |
| `total_points` | FPL points earned in this GW | int | **main player target** |
| `transfers_balance`, `transfers_in`, `transfers_out` | FPL transfer activity | int | unused |
| `value` | Player price in this GW (×10, e.g. `55` = £5.5m) | int | player features (price) |
| `was_home` | `"True"` / `"False"` **string** | str → bool | home/away feature |
| `season` | Injected by loader (folder name) | str | grouping |

**No xG/xA per player.** FPL data does not include expected goals/assists in
these files. If needed for §6.3 features, they must come from a separate
Understat player export (not currently supplied).

---

## 3. `master_player_gw.csv` — FPL per-GW rows (same payload, different layout)

Contains the **same 30 columns** as `master_gameweeks.csv` **minus**
`name / position / team / xP` **plus**:

| Column | Meaning | Type | Used for |
|---|---|---|---|
| `player_folder_name` | `"{SanitizedName}_{element}"`, e.g. `"Aaron_Connolly_78"` | str | **bridge to `master_player_history`** |

Row order appears to match `master_gameweeks.csv` on `(element, fixture)`.
**Join on `element` (int)**, not position.

---

## 4. `master_player_history.csv` — season-level per-player aggregates

One row per **player per past season**. The same player contributes one row
per season they've played in the Premier League.

| Column | Meaning | Type | Used for |
|---|---|---|---|
| `element_code` | **Stable FPL player code** (e.g. `233425`) — **canonical player key across seasons** | int | primary player identity |
| `player_folder_name` | Same format as in `master_player_gw.csv` | str | bridge from `master_player_gw` |
| `season_name` | `"2019/20"` (**slash separator**) | str | must normalise to `"2019-20"` |
| `start_cost`, `end_cost` | FPL price start/end of season (×10) | int | price feature |
| `total_points` | FPL season total points | int | **main target for season model** |
| `minutes`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`, `saves`, `bonus`, `bps`, `ict_index`, `influence`, `creativity`, `threat` | Season aggregates | int / float | season features |
| `own_goals`, `penalties_missed`, `penalties_saved`, `red_cards`, `yellow_cards` | Season incident counts | int | season features |

**Includes 2019/20**, i.e. one season **before** the project's training
window. This gives free priors for cold-start promoted teams and first-
season players.

---

## 5. `master_understat_teams.csv` — Understat per-team-per-match

One row per **team per match**. To reconstruct a single match, pair the two
teams' rows on `date`.

| Column | Meaning | Type | Used for |
|---|---|---|---|
| `h_a` | `"h"` (home) / `"a"` (away) | str | pair reconstruction |
| `xG`, `xGA` | Expected goals for / against | float | **rolling team features** |
| `npxG`, `npxGA` | Non-penalty xG / xGA | float | rolling team features |
| `ppda` | **Dict-string** `"{'att': 307, 'def': 14}"` | str → parse | PPDA (passes per defensive action) — parse in Phase 2 |
| `ppda_allowed` | Same structure for opponent | str → parse | defensive pressure feature |
| `deep`, `deep_allowed` | Deep completions for / against | int | team features |
| `scored`, `missed` | Goals scored / conceded | int | cross-check with E0 |
| `xpts` | Expected points (Understat model) | float | comparison baseline |
| `result` | `w` / `d` / `l` | str | cross-check |
| `date` | Match datetime, ISO `YYYY-MM-DD HH:MM:SS` | str → datetime | join to E0 |
| `wins`, `draws`, `loses`, `pts` | Cumulative season stats | int | unused (E0 has these) |
| `npxGD` | Non-penalty xG difference | float | team feature |
| `team_name` | **Long-form** club name (`"Manchester United"`) | str | **join key — needs mapping to E0** |

**Missing for 2025-26 and 2026-27** (cahier des charges §4.1). Phase 2 must
implement an imputation or a fallback model without xG.

---

## 6. `master_xP_projections.csv` — expected FPL points

| Column | Meaning | Type | Used for |
|---|---|---|---|
| `id` | **Ambiguous** — likely `element` but unconfirmed | int | ? |
| `xP` | Expected FPL points | float | extra player feature |
| `season` | Injected by loader | str | grouping |

**Open question Q3:** `id` may be `element` (season-scoped), a per-fixture
row identifier, or a per-player-history id. Must confirm with a full file
before use. Fallback: drop this file if unresolvable (it's listed as
"feature complémentaire" in the cahier des charges §4.1).

---

## Identity resolution rules (Phase 2 will implement)

| Entity | Inside one season | Across seasons |
|---|---|---|
| **Team** | `E0.Home_Team` / `E0.Away_Team` (short name) or `gameweeks.team` (short) | map via `ml/mappings.py` |
| **Player (seasonal)** | `element` (int) | **do not rely on** |
| **Player (stable)** | `element_code` (from `player_history`) | **canonical PK** |
| **Player (cross-file)** | `player_folder_name` | stable within a season folder only |
| **Fixture** | `fixture` (int) inside one season | not portable |

---

## Column conventions & parsing rules (Phase 2)

- `E0.Match_Date`: `pd.to_datetime(..., dayfirst=True)`
- `gameweeks.kickoff_time`: `pd.to_datetime(..., utc=True)`
- `understat.date`: `pd.to_datetime(..., format="%Y-%m-%d %H:%M:%S")`
- `was_home`: `"True"`/`"False"` strings → `bool`
- `ppda` / `ppda_allowed`: `ast.literal_eval` the dict string → ratio `att/def`
- `player_history.season_name`: `"2019/20"` → `"2019-20"`
- `value`, `start_cost`, `end_cost`: FPL price ×10 → divide by 10 to get £m

---

## Open questions

- **Q1** (fixture calendar): no dedicated fixture calendar file supplied.
  `master_gameweeks` is player-GW data; a gameweek calendar can be
  **derived** by grouping `fixture` + `kickoff_time` + `team_h_score` /
  `team_a_score` per round.
- **Q2** (FPL team ID): `opponent_team` (int) has no supplied name mapping.
  Must be built from observed `(team, opponent_team)` pairs per season, or
  enriched from a public FPL teams snapshot.
- **Q3** (`xP_projections.id`): see §6.
- **Q4** (2026-27): the file must be inspected to know how many matches are
  played vs. upcoming.
- **Q5** (age / DOB): not present in any file. Needed for
  `F-S2 Meilleur jeune joueur`. Plan A: enrich via FPL API
  `bootstrap-static` (`birth_date`). Plan B: proxy = years since first
  `player_history.season_name`. Plan C: manual lookup for top-N young
  candidates.