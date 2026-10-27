#!/usr/bin/env python3
"""
Football Elo ranking system
===========================

Reads match results from a CSV and produces:
  * rankings.csv        - current table, one row per team
  * rating_history.csv  - every match with ratings before/after and the change

How a rating changes after each match
-------------------------------------
Each match moves ratings for two reasons, blended together:

1. RESULT (win / draw / loss)
   Classic Elo. From the rating gap we get an expected result
       E = 1 / (1 + 10 ** (-(R_team - R_opp) / 400))
   Actual result S is 1 for a win, 0.5 for a draw, 0 for a loss.
   Result surprise = S - E   (range -1 .. +1)

2. SCORE (expected-goals model)
   The rating gap also predicts a goal difference:
       expected_gd = (R_team - R_opp) / POINTS_PER_GOAL
   Score surprise = tanh((actual_gd - expected_gd) / MARGIN_SOFTNESS)
   tanh keeps it in -1 .. +1, so beating expectations by 1-2 goals matters a
   lot but a 9-0 is not wildly different from a 5-0.

   Example: a team rated 200 above its opponent is expected to win by ~1 goal.
   Winning 1-0 is roughly "as expected" on the score side; winning 4-0 earns
   extra; scraping a 2-2 draw costs on both sides.

   change = K * (RESULT_WEIGHT * result_surprise + SCORE_WEIGHT * score_surprise)

Fairness for teams with different numbers of games / opponents
--------------------------------------------------------------
* Opponent strength is built in: beating a strong team is a bigger surprise
  than beating a weak one, so it earns more.
* Provisional K-factor: a team's K starts at K_BASE * PROVISIONAL_MULTIPLIER
  and slides down to K_BASE over its first PROVISIONAL_GAMES matches, so new
  teams reach their true level quickly.
* Reliability damping: when you play a provisional (barely-known) team, your
  own rating moves less, because their rating is not yet trustworthy.

Usage
-----
    python elo.py                      # reads all results from RESULTS_URL (GitHub)
    python elo.py --out-dir output --min-games 5 --dayfirst
    python elo.py --predict "Team A" "Team B"

CSV format (header names are case-insensitive)
----------------------------------------------
    date,game_order,team1,team2,goals1,goals2
    2026-08-01,1,Leeds,York,2,1
    2026-08-01,2,Leeds,Hull,0,0

Accepted aliases: team1/home/home_team, team2/away/away_team,
goals1/score1/home_goals, goals2/score2/away_goals,
game_order/game/order/match_no/sequence.
Matches are processed by date, then game_order within that date (so a team
playing several games in one day carries its updated rating into the next).
game_order is optional; without it, same-date games keep their CSV order.
Rows missing a game_order run after that day's numbered games.
Home advantage is deliberately NOT used: team1/team2 are interchangeable.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Tunable settings
# ---------------------------------------------------------------------------
# Every game lives in this one file; the script always reads the latest version.
RESULTS_URL = "https://github.com/rogers1000/fm_draft/blob/main/draft_game_results.csv"
START_RATING = 1500          # every new team starts here
K_BASE = 30                  # K for an established team
PROVISIONAL_MULTIPLIER = 2.0 # new teams' K is this many times K_BASE...
PROVISIONAL_GAMES = 15       # ...falling to K_BASE over this many games
RESULT_WEIGHT = 0.6          # share of the change driven by W/D/L
SCORE_WEIGHT = 0.4           # share driven by goal difference vs expectation
POINTS_PER_GOAL = 200        # rating gap that predicts a 1-goal margin
MARGIN_SOFTNESS = 2.0        # larger = goal surprises saturate more slowly
MIN_RELIABILITY_DAMPING = 0.5  # your K vs a brand-new opponent is scaled to this

COLUMN_ALIASES = {
    "date": ["date", "match_date", "played"],
    "game_order": ["game_order", "game", "game_no", "game_number", "order",
                   "match_no", "match_number", "sequence", "seq"],
    "team1": ["team1", "team_1", "home", "home_team", "team_a"],
    "team2": ["team2", "team_2", "away", "away_team", "team_b"],
    "goals1": ["goals1", "score1", "home_goals", "home_score", "goals_a"],
    "goals2": ["goals2", "score2", "away_goals", "away_score", "goals_b"],
}


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------
def expected_result(r_team: float, r_opp: float) -> float:
    """Probability-like expected score (win=1, draw=0.5) for r_team."""
    return 1.0 / (1.0 + 10 ** (-(r_team - r_opp) / 400.0))


def expected_goal_diff(r_team: float, r_opp: float) -> float:
    return (r_team - r_opp) / POINTS_PER_GOAL


def reliability(games: int) -> float:
    """0 for a brand-new team, 1 once it has played PROVISIONAL_GAMES."""
    return min(1.0, games / PROVISIONAL_GAMES)


def k_factor(games: int, opp_games: int) -> float:
    k = K_BASE * (1 + (PROVISIONAL_MULTIPLIER - 1) * (1 - reliability(games)))
    damping = MIN_RELIABILITY_DAMPING + (1 - MIN_RELIABILITY_DAMPING) * reliability(opp_games)
    return k * damping


def rating_change(r_team, r_opp, goals_for, goals_against, games, opp_games):
    """Return (change, details) for one team in one match."""
    gd = goals_for - goals_against
    actual = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
    exp_res = expected_result(r_team, r_opp)
    exp_gd = expected_goal_diff(r_team, r_opp)

    result_surprise = actual - exp_res
    score_surprise = math.tanh((gd - exp_gd) / MARGIN_SOFTNESS)
    k = k_factor(games, opp_games)
    change = k * (RESULT_WEIGHT * result_surprise + SCORE_WEIGHT * score_surprise)
    return change, {"expected_result": exp_res, "expected_gd": exp_gd, "k": k}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def to_raw_url(source: str) -> str:
    """Turn a GitHub page link (.../blob/...) into the raw file link pandas can read."""
    if source.startswith("https://github.com/") and "/blob/" in source:
        return (source.replace("https://github.com/", "https://raw.githubusercontent.com/", 1)
                      .replace("/blob/", "/", 1))
    return source


def load_results(source: str, dayfirst: bool) -> pd.DataFrame:
    df = pd.read_csv(to_raw_url(str(source)))
    lower = {c.strip().lower(): c for c in df.columns}
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        found = next((lower[a] for a in aliases if a in lower), None)
        if found is None:
            if canon in ("date", "game_order"):
                continue  # optional; fall back to row order
            sys.exit(f"Missing column for '{canon}'. Expected one of: {aliases}")
        rename[found] = canon
    df = df.rename(columns=rename)[list(rename.values())].copy()

    df["team1"] = df["team1"].astype(str).str.strip()
    df["team2"] = df["team2"].astype(str).str.strip()
    df["goals1"] = pd.to_numeric(df["goals1"], errors="coerce")
    df["goals2"] = pd.to_numeric(df["goals2"], errors="coerce")
    df["csv_row"] = df.index + 2  # spreadsheet-style row number incl. header

    bad = df[df["goals1"].isna() | df["goals2"].isna() | (df["team1"] == df["team2"])]
    if not bad.empty:
        print(f"Skipping {len(bad)} invalid row(s) (missing score or team vs itself): "
              f"CSV rows {bad['csv_row'].tolist()}", file=sys.stderr)
        df = df.drop(bad.index)

    if "game_order" in df.columns:
        df["game_order"] = pd.to_numeric(df["game_order"], errors="coerce")
        missing = df["game_order"].isna()
        if missing.any():
            print(f"{missing.sum()} row(s) have no game order; they run after the "
                  f"numbered games that day, in CSV order: "
                  f"CSV rows {df.loc[missing, 'csv_row'].tolist()}", file=sys.stderr)

    sort_cols = []
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=dayfirst, errors="coerce")
        undated = df["date"].isna()
        if undated.any():
            print(f"Skipping {undated.sum()} row(s) with unreadable dates: "
                  f"CSV rows {df.loc[undated, 'csv_row'].tolist()}", file=sys.stderr)
            df = df[~undated]
        sort_cols.append("date")
    if "game_order" in df.columns:
        sort_cols.append("game_order")
        dup = df.duplicated(subset=[c for c in ("date", "game_order") if c in df.columns],
                            keep=False) & df["game_order"].notna()
        if dup.any():
            print(f"Warning: duplicate game order numbers on the same date "
                  f"(CSV order used to break ties): CSV rows {df.loc[dup, 'csv_row'].tolist()}",
                  file=sys.stderr)
    sort_cols.append("csv_row")
    df = df.sort_values(sort_cols, kind="stable", na_position="last")
    df[["goals1", "goals2"]] = df[["goals1", "goals2"]].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Running the system
# ---------------------------------------------------------------------------
def new_team():
    return {"rating": START_RATING, "games": 0, "W": 0, "D": 0, "L": 0,
            "GF": 0, "GA": 0, "peak": START_RATING, "last_played": None}


def run(df: pd.DataFrame):
    teams = {}
    history = []
    for m in df.itertuples(index=False):
        a = teams.setdefault(m.team1, new_team())
        b = teams.setdefault(m.team2, new_team())
        ra, rb = a["rating"], b["rating"]

        ch_a, da = rating_change(ra, rb, m.goals1, m.goals2, a["games"], b["games"])
        ch_b, db = rating_change(rb, ra, m.goals2, m.goals1, b["games"], a["games"])

        date = getattr(m, "date", None)
        for t, gf, ga, ch in ((a, m.goals1, m.goals2, ch_a), (b, m.goals2, m.goals1, ch_b)):
            t["rating"] += ch
            t["peak"] = max(t["peak"], t["rating"])
            t["games"] += 1
            t["GF"] += gf
            t["GA"] += ga
            t["W" if gf > ga else "D" if gf == ga else "L"] += 1
            t["last_played"] = date

        history.append({
            "date": date.date() if date is not None else None,
            "game_order": (int(m.game_order) if hasattr(m, "game_order")
                           and pd.notna(m.game_order) else None),
            "team1": m.team1, "team2": m.team2,
            "goals1": m.goals1, "goals2": m.goals2,
            "team1_before": round(ra, 1), "team2_before": round(rb, 1),
            "team1_win_expectancy": round(da["expected_result"], 3),
            "expected_gd_team1": round(da["expected_gd"], 2),
            "team1_change": round(ch_a, 1), "team2_change": round(ch_b, 1),
            "team1_after": round(a["rating"], 1), "team2_after": round(b["rating"], 1),
        })
    columns = ["date", "game_order", "team1", "team2", "goals1", "goals2",
               "team1_before", "team2_before", "team1_win_expectancy",
               "expected_gd_team1", "team1_change", "team2_change",
               "team1_after", "team2_after"]
    return teams, pd.DataFrame(history, columns=columns)


def build_table(teams, min_games: int) -> pd.DataFrame:
    rows = []
    for name, t in teams.items():
        rows.append({
            "team": name,
            "rating": round(t["rating"], 1),
            "played": t["games"], "W": t["W"], "D": t["D"], "L": t["L"],
            "GF": t["GF"], "GA": t["GA"], "GD": t["GF"] - t["GA"],
            "peak_rating": round(t["peak"], 1),
            "provisional": t["games"] < PROVISIONAL_GAMES,
            "last_played": t["last_played"].date() if t["last_played"] is not None else None,
        })
    columns = ["team", "rating", "played", "W", "D", "L", "GF", "GA", "GD",
               "peak_rating", "provisional", "last_played"]
    table = pd.DataFrame(rows, columns=columns).sort_values("rating", ascending=False)
    table = table[table["played"] >= min_games].reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def predict(teams, t1, t2):
    r1 = teams.get(t1, new_team())["rating"]
    r2 = teams.get(t2, new_team())["rating"]
    e = expected_result(r1, r2)
    gd = expected_goal_diff(r1, r2)
    print(f"\n{t1} ({r1:.0f}) vs {t2} ({r2:.0f})")
    print(f"  Win expectancy for {t1}: {e:.1%}  (draw counted as half)")
    print(f"  Expected goal difference: {gd:+.2f} to {t1 if gd >= 0 else t2}")
    for name in (t1, t2):
        if name not in teams:
            print(f"  Note: {name} has no games yet - using start rating {START_RATING}.")


def main():
    p = argparse.ArgumentParser(description="Football Elo rankings from a results CSV.")
    p.add_argument("--out-dir", type=Path, default=Path("."), help="where to write outputs")
    p.add_argument("--min-games", type=int, default=0,
                   help="hide teams with fewer games from rankings.csv")
    p.add_argument("--dayfirst", action="store_true",
                   help="read dates as DD/MM/YYYY (UK style)")
    p.add_argument("--predict", nargs=2, metavar=("TEAM1", "TEAM2"),
                   help="show the expected outcome of a match after processing")
    args = p.parse_args()

    # The GitHub Action sets ELO_RESULTS_FILE to its fresh checkout of the CSV,
    # because raw GitHub links can lag a few minutes behind a push.
    source = os.environ.get("ELO_RESULTS_FILE") or RESULTS_URL
    df = load_results(source, args.dayfirst)
    teams, history = run(df)
    table = build_table(teams, args.min_games)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "rankings.csv", index=False)
    history.to_csv(args.out_dir / "rating_history.csv", index=False)

    if table.empty:
        print("No games recorded yet.")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(table.to_string(index=False))
    print(f"\n{len(df)} matches, {len(teams)} teams. "
          f"Wrote rankings.csv and rating_history.csv to {args.out_dir.resolve()}")

    if args.predict:
        predict(teams, *args.predict)


if __name__ == "__main__":
    main()
