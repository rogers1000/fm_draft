#!/usr/bin/env python3
"""
Football Elo ranking system
===========================

Reads match results from a CSV and produces:
  * rankings.csv        - current table, one row per team: the RESULTS rating
                          (from scores) plus a PERFORMANCE rating (from xG, when the
                          sheet's optional xg1/xg2 columns are filled in)
  * rating_history.csv  - every match with ratings before/after and the change
  * predictions.csv     - every possible pairing of teams, with win expectancy
                          and expected goal difference from current ratings

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
* Match weight: K depends on the match's competition (COMPETITION_K), so a
  Streamer Showdown game moves ratings twice as much as a standard game.
* Same K for both teams and for every team (as in World Football Elo). New teams
  get no faster rating changes, and playing a new team has no special effect.
* Provisional label: teams with fewer than PROVISIONAL_GAMES matches are
  flagged as provisional in the outputs. The flag never changes the maths.
* Zero-sum: in every match, one team gains exactly what the other loses, so
  the average rating across all teams stays at START_RATING.

Usage
-----
    python elo.py                      # reads all results from RESULTS_URL (Google Sheet)
    python elo.py --out-dir output --min-games 5
    python elo.py --predict "Team A" "Team B"

CSV format (header names are case-insensitive)
----------------------------------------------
    date,game_order,team1,team2,goals1,goals2,competition
    2026-08-01,1,Leeds,York,2,1,Streamer Showdown
    2026-08-01,2,Leeds,Hull,0,0,Fun

competition is optional; without it (or when blank) every game uses K.

Accepted aliases: team1/home/home_team, team2/away/away_team,
goals1/score1/home_goals, goals2/score2/away_goals,
game_order/game/order/match_no/sequence.
Matches are processed by date, then game_order within that date (so a team
playing several games in one day carries its updated rating into the next).
game_order is optional; without it, same-date games keep their CSV order.
Rows missing a game_order run after that day's numbered games.
team1 is the HOME side and gets HOME_ADVANTAGE (30) points in the expected
result (set it to 0 to switch off),
except in neutral games (NEUTRAL_COMPETITIONS, or a "neutral" column).
stage (e.g. Group / KO) is optional and recorded for analysis only.
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Tunable settings
# ---------------------------------------------------------------------------
# Every game lives in this Google Sheet; the script always reads the latest version.
# The sheet must be shared as "Anyone with the link can view".
RESULTS_URL = ("https://docs.google.com/spreadsheets/d/"
               "1fiPOeL43SXmzuryRHTDn2l9Qw8KXhIXiQ7cLvQbwNpQ/edit?usp=sharing")
START_RATING = 1500          # every new team starts here
K = 30                       # match weight for any competition not listed below

# Match weight by competition (World Football Elo style: big tournaments count
# more). Matched against the sheet's "competition" column, ignoring capitals and
# spaces; a name only has to contain the key, so "Streamer Showdown 2" matches
# "streamer showdown". Blank or unlisted competitions use K above.
COMPETITION_K = {
    "streamer showdown": 60,
}

# Home advantage (World Football Elo style, but 30 points instead of 100; 30 fits
# the group-stage results best: about +4% win chance between equal teams).
# team1 is the home side. When working out the expected result, team1 is treated
# as HOME_ADVANTAGE points stronger; ratings themselves are untouched. So a home
# win earns less and a home draw/loss costs more. Neutral games get no bonus.
HOME_ADVANTAGE = 30          # points added to team1 (home) in the expected result; 0 = off
# Competitions played at a neutral venue (same matching rules as COMPETITION_K).
# Empty: every competition so far, Streamer Showdown included, has a home side.
NEUTRAL_COMPETITIONS = []
# Stages played at a neutral venue, per competition (Benjy Discord rules: every
# knockout game is neutral; the group stage has a home side). Stage names are the
# cleaned ones from the sheet's round/stage column, e.g. "Group" or "KO".
NEUTRAL_STAGES = {
    "benjy discord": ["KO"],
}
# An optional "neutral" column in the sheet overrides all of this for single games:
# yes/y/true/1/x = neutral, no/n/false/0 = team1 at home, blank = use the rules above.

PROVISIONAL_GAMES = 15       # teams with fewer games are labelled provisional (label only)
RESULT_WEIGHT = 0.6          # share of the change driven by W/D/L
SCORE_WEIGHT = 0.4           # share driven by goal difference vs expectation
POINTS_PER_GOAL = 200        # rating gap that predicts a 1-goal margin
MARGIN_SOFTNESS = 2.0        # larger = goal surprises saturate more slowly
TOSS_UP_MARGIN = 0.05        # predictions within 45-55% are "Too close to call"

COLUMN_ALIASES = {
    "date": ["date", "match_date", "played"],
    "game_order": ["game_order", "game", "game_no", "game_number", "order",
                   "match_no", "match_number", "sequence", "seq"],
    "team1": ["team1", "team_1", "home", "home_team", "team_a"],
    "team2": ["team2", "team_2", "away", "away_team", "team_b"],
    "goals1": ["goals1", "score1", "home_goals", "home_score", "goals_a"],
    "goals2": ["goals2", "score2", "away_goals", "away_score", "goals_b"],
    "competition": ["competition", "comp", "event", "tournament", "match_type", "type"],
    "neutral": ["neutral", "neutral_venue", "is_neutral"],
    # Recorded for analysis only (e.g. Group / KO); it does not change ratings.
    "stage": ["stage", "round", "phase"],
    # Optional xG for each side, from the match report. Used only by the performance rating.
    "xg1": ["xg1", "xg_1", "team1_xg", "home_xg", "xg_home"],
    "xg2": ["xg2", "xg_2", "team2_xg", "away_xg", "xg_away"],
}
OPTIONAL_COLUMNS = ("date", "game_order", "competition", "neutral", "stage", "xg1", "xg2")


def _norm_name(text) -> str:
    return " ".join(str(text).lower().split())


STAGE_SPELLINGS = {
    "group": "Group", "groups": "Group", "group stage": "Group",
    "ko": "KO", "k.o.": "KO", "knockout": "KO", "knock out": "KO", "knock-out": "KO",
    "knockouts": "KO",
}


def clean_stage(stage):
    """Standardise common spellings (group -> Group, knockout -> KO); keep anything else as typed."""
    if stage is None or pd.isna(stage) or not str(stage).strip():
        return None
    text = " ".join(str(stage).split())
    return STAGE_SPELLINGS.get(text.lower(), text)


def is_neutral(competition, override=None, stage=None) -> bool:
    """True when the game has no home side."""
    if override is not None and not pd.isna(override) and str(override).strip():
        flag = str(override).strip().lower()
        if flag in ("yes", "y", "true", "1", "x", "neutral"):
            return True
        if flag in ("no", "n", "false", "0", "home"):
            return False
    if competition is None or pd.isna(competition):
        return False
    name = _norm_name(competition)
    if any(_norm_name(key) in name for key in NEUTRAL_COMPETITIONS):
        return True
    if stage is not None and not pd.isna(stage):
        for key, stages in NEUTRAL_STAGES.items():
            if _norm_name(key) in name and _norm_name(stage) in {_norm_name(s) for s in stages}:
                return True
    return False


def match_k(competition) -> float:
    """Match weight for a competition name (blank -> K)."""
    if competition is None or pd.isna(competition):
        return K
    name = " ".join(str(competition).lower().split())
    for key, k in COMPETITION_K.items():
        if " ".join(key.lower().split()) in name:
            return k
    return K


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------
def expected_result(r_team: float, r_opp: float) -> float:
    """Probability-like expected score (win=1, draw=0.5) for r_team."""
    return 1.0 / (1.0 + 10 ** (-(r_team - r_opp) / 400.0))


def expected_goal_diff(r_team: float, r_opp: float) -> float:
    return (r_team - r_opp) / POINTS_PER_GOAL


def rating_change(r_team, r_opp, goals_for, goals_against, k=None):
    """Return (change, details) for r_team. The opponent's change is exactly -change."""
    k = K if k is None else k
    gd = goals_for - goals_against
    actual = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
    exp_res = expected_result(r_team, r_opp)
    exp_gd = expected_goal_diff(r_team, r_opp)

    result_surprise = actual - exp_res
    score_surprise = math.tanh((gd - exp_gd) / MARGIN_SOFTNESS)
    change = k * (RESULT_WEIGHT * result_surprise + SCORE_WEIGHT * score_surprise)
    return change, {"expected_result": exp_res, "expected_gd": exp_gd}


# ---------------------------------------------------------------------------
# Performance rating (xG Elo)
# ---------------------------------------------------------------------------
# A second, separate rating that learns from chances instead of the scoreline:
# the "result" is the chance each side had of winning given its xG (draw = half),
# and the margin is the xG difference. Same home advantage, K and maths as the
# results rating. Games with no xG entered fall back to the actual score.
def xg_result(xg_for: float, xg_against: float, max_goals: int = 12) -> float:
    """Expected result (win = 1, draw = 0.5) from both sides' xG, treating goals as Poisson."""
    pf = [math.exp(-xg_for) * xg_for ** k / math.factorial(k) for k in range(max_goals)]
    pa = [math.exp(-xg_against) * xg_against ** k / math.factorial(k) for k in range(max_goals)]
    win = sum(pf[i] * pa[j] for i in range(max_goals) for j in range(i))
    draw = sum(pf[i] * pa[i] for i in range(max_goals))
    return win + 0.5 * draw


def performance_change(r_team, r_opp, result, margin, k=None):
    """Like rating_change, but with a fractional result and an xG margin."""
    k = K if k is None else k
    exp_res = expected_result(r_team, r_opp)
    exp_gd = expected_goal_diff(r_team, r_opp)
    return k * (RESULT_WEIGHT * (result - exp_res)
                + SCORE_WEIGHT * math.tanh((margin - exp_gd) / MARGIN_SOFTNESS))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def to_raw_url(source: str) -> str:
    """Turn a Google Sheets or GitHub page link into a direct CSV link pandas can read."""
    sheet = re.match(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)", source)
    if sheet:
        gid = re.search(r"[#&?]gid=(\d+)", source)  # a specific tab, if the link names one
        url = f"https://docs.google.com/spreadsheets/d/{sheet.group(1)}/export?format=csv"
        return url + (f"&gid={gid.group(1)}" if gid else "")
    if source.startswith("https://github.com/") and "/blob/" in source:
        return (source.replace("https://github.com/", "https://raw.githubusercontent.com/", 1)
                      .replace("/blob/", "/", 1))
    return source


def parse_dates(dates: pd.Series, dayfirst: bool) -> pd.Series:
    """Year-first dates (2026-10-28) are read as written; anything else
    (28/10/2026, 28 Oct 2026) is read day-first unless dayfirst is False."""
    text = dates.astype(str).str.strip()
    year_first = text.str.extract(r"^(\d{4}-\d{1,2}-\d{1,2})", expand=False)
    iso = pd.to_datetime(year_first, format="%Y-%m-%d", errors="coerce")
    other = iso.isna()
    if other.any():
        iso[other] = pd.to_datetime(text[other], format="mixed", dayfirst=dayfirst,
                                    errors="coerce")
    return iso


def load_results(source: str, dayfirst: bool) -> pd.DataFrame:
    df = pd.read_csv(to_raw_url(str(source)))
    lower = {c.strip().lower(): c for c in df.columns}
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        found = next((lower[a] for a in aliases if a in lower), None)
        if found is None:
            if canon in OPTIONAL_COLUMNS:
                continue  # optional; fall back to row order
            sys.exit(f"Missing column for '{canon}'. Expected one of: {aliases}")
        rename[found] = canon
    df = df.rename(columns=rename)[list(rename.values())].copy()
    df = df.dropna(how="all")  # spreadsheets export their empty rows; ignore them

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
        df["date"] = parse_dates(df["date"], dayfirst)
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

    if "competition" not in df.columns:
        df["competition"] = None
    df["competition"] = df["competition"].map(
        lambda c: None if pd.isna(c) or not str(c).strip() else " ".join(str(c).split())
    ).astype(object)
    df["match_weight"] = df["competition"].map(match_k)
    if "stage" not in df.columns:
        df["stage"] = None
    df["stage"] = df["stage"].map(clean_stage).astype(object)
    if "neutral" not in df.columns:
        df["neutral"] = None
    df["neutral"] = [is_neutral(c, o, s) for c, o, s in
                     zip(df["competition"], df["neutral"], df["stage"])]
    summary = (df.assign(competition=df["competition"].fillna("(blank)"))
                 .groupby("competition", sort=False)
                 .agg(weight=("match_weight", "first"), games=("match_weight", "size"),
                      neutral=("neutral", "sum")))
    for comp, row in summary.iterrows():
        if HOME_ADVANTAGE:
            venue = (", neutral" if row["neutral"] == row["games"] else
                     ", team1 at home" if row["neutral"] == 0 else
                     f", {row['neutral']} neutral, rest team1 at home")
        else:
            venue = ""
        print(f"Competition '{comp}': {row['games']} game(s) at weight {row['weight']:g}"
              f"{venue}", file=sys.stderr)

    for c in ("xg1", "xg2"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else float("nan")
    with_xg = (df["xg1"].notna() & df["xg2"].notna()).sum()
    if len(df):
        print(f"xG entered for {with_xg} of {len(df)} game(s) "
              f"(the rest use the score in the performance rating)", file=sys.stderr)

    if len(df):
        counts = df["stage"].fillna("(blank)").value_counts()
        print("Stages: " + ", ".join(f"{k} {v}" for k, v in counts.items()), file=sys.stderr)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Running the system
# ---------------------------------------------------------------------------
def new_team():
    return {"rating": START_RATING, "games": 0, "W": 0, "D": 0, "L": 0,
            "GF": 0, "GA": 0, "peak": START_RATING, "last_played": None,
            "perf": START_RATING, "xg_games": 0, "xGF": 0.0, "xGA": 0.0}


def run(df: pd.DataFrame):
    teams = {}
    history = []
    for m in df.itertuples(index=False):
        a = teams.setdefault(m.team1, new_team())
        b = teams.setdefault(m.team2, new_team())
        ra, rb = a["rating"], b["rating"]

        home = 0 if m.neutral else HOME_ADVANTAGE
        ch_a, da = rating_change(ra + home, rb, m.goals1, m.goals2, m.match_weight)
        ch_b = -ch_a  # zero-sum: whatever one team gains, the other loses

        # Performance rating: from xG when both sides have it, otherwise from the score.
        pa, pb = a["perf"], b["perf"]
        has_xg = pd.notna(m.xg1) and pd.notna(m.xg2)
        if has_xg:
            p_res, p_margin = xg_result(m.xg1, m.xg2), m.xg1 - m.xg2
            for t, xf, xa in ((a, m.xg1, m.xg2), (b, m.xg2, m.xg1)):
                t["xg_games"] += 1; t["xGF"] += xf; t["xGA"] += xa
        else:
            gd = m.goals1 - m.goals2
            p_res, p_margin = (1.0 if gd > 0 else 0.5 if gd == 0 else 0.0), gd
        pch = performance_change(pa + home, pb, p_res, p_margin, m.match_weight)
        a["perf"] += pch
        b["perf"] -= pch

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
            "competition": m.competition, "match_weight": m.match_weight,
            "stage": m.stage,
            "neutral": bool(m.neutral), "home_advantage": home,
            "team1": m.team1, "team2": m.team2,
            "goals1": m.goals1, "goals2": m.goals2,
            "team1_before": round(ra, 1), "team2_before": round(rb, 1),
            "team1_win_expectancy": round(da["expected_result"], 3),
            "expected_gd_team1": round(da["expected_gd"], 2),
            "team1_change": round(ch_a, 1), "team2_change": round(ch_b, 1),
            "team1_after": round(a["rating"], 1), "team2_after": round(b["rating"], 1),
            "xg1": m.xg1 if has_xg else None, "xg2": m.xg2 if has_xg else None,
            "team1_perf_before": round(pa, 1), "team2_perf_before": round(pb, 1),
            "team1_perf_change": round(pch, 1), "team2_perf_change": round(-pch, 1),
        })
    columns = ["date", "game_order", "competition", "match_weight",
               "stage", "neutral", "home_advantage",
               "team1", "team2", "goals1", "goals2",
               "team1_before", "team2_before", "team1_win_expectancy",
               "expected_gd_team1", "team1_change", "team2_change",
               "team1_after", "team2_after",
               "xg1", "xg2", "team1_perf_before", "team2_perf_before",
               "team1_perf_change", "team2_perf_change"]
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
            "performance_rating": round(t["perf"], 1),
            "xg_games": t["xg_games"],
            "xGF": round(t["xGF"], 2), "xGA": round(t["xGA"], 2),
        })
    columns = ["team", "rating", "played", "W", "D", "L", "GF", "GA", "GD",
               "peak_rating", "provisional", "last_played",
               "performance_rating", "xg_games", "xGF", "xGA"]
    table = pd.DataFrame(rows, columns=columns).sort_values("rating", ascending=False)
    table = table[table["played"] >= min_games].reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    # Teams with no xG games have no performance data (their figure would just copy the
    # results rating), so they get a blank performance rating and rank.
    no_xg = table["xg_games"] == 0
    table["performance_rating"] = table["performance_rating"].mask(no_xg)
    perf_rank = table["performance_rating"].rank(ascending=False, method="min").astype("Int64")
    table.insert(table.columns.get_loc("performance_rating") + 1, "performance_rank", perf_rank)
    for c in ("xGF", "xGA"):
        table[c] = table[c].mask(no_xg)
    return table


def build_predictions(teams, team_names) -> pd.DataFrame:
    """Every pairing of the given teams, once each, predicted at a neutral venue
    (no home advantage), using ratings that already allow for home advantage."""
    ordered = sorted(team_names, key=lambda n: -teams[n]["rating"])
    rows = []
    for i, t1 in enumerate(ordered):
        for t2 in ordered[i + 1:]:
            r1, r2 = teams[t1]["rating"], teams[t2]["rating"]
            e = expected_result(r1, r2)
            gd = expected_goal_diff(r1, r2)
            fav = t1 if r1 >= r2 else t2
            rows.append({
                "team1": t1, "team2": t2,
                "team1_rating": round(r1, 1), "team2_rating": round(r2, 1),
                "team1_win_expectancy": round(e, 3),
                "team2_win_expectancy": round(1 - e, 3),
                "team1_home_win_expectancy": round(expected_result(r1 + HOME_ADVANTAGE, r2), 3),
                "team2_home_win_expectancy": round(expected_result(r2 + HOME_ADVANTAGE, r1), 3),
                "expected_gd_team1": round(gd, 2),
                "favourite": fav if abs(e - 0.5) > TOSS_UP_MARGIN else "Too close to call",
                "either_provisional": teams[t1]["games"] < PROVISIONAL_GAMES
                                      or teams[t2]["games"] < PROVISIONAL_GAMES,
            })
    columns = ["team1", "team2", "team1_rating", "team2_rating",
               "team1_win_expectancy", "team2_win_expectancy",
               "team1_home_win_expectancy", "team2_home_win_expectancy",
               "expected_gd_team1", "favourite", "either_provisional"]
    # Predictions are for a neutral venue only; the home columns are left out of the output.
    return pd.DataFrame(rows, columns=columns).drop(
        columns=["team1_home_win_expectancy", "team2_home_win_expectancy"])


def predict(teams, t1, t2):
    r1 = teams.get(t1, new_team())["rating"]
    r2 = teams.get(t2, new_team())["rating"]
    print(f"\n{t1} ({r1:.0f}) vs {t2} ({r2:.0f})   (win expectancy, draw counted as half)")
    venues = [("Neutral venue", 0)]
    if HOME_ADVANTAGE:
        venues += [(f"{t1} at home", HOME_ADVANTAGE), (f"{t2} at home", -HOME_ADVANTAGE)]
    for venue, home in venues:
        e = expected_result(r1 + home, r2)
        gd = expected_goal_diff(r1 + home, r2)
        print(f"  {venue:<28} {t1} {e:.1%} / {t2} {1 - e:.1%}, "
              f"expected goal difference {abs(gd):.2f} to {t1 if gd >= 0 else t2}")
    for name in (t1, t2):
        if name not in teams:
            print(f"  Note: {name} has no games yet - using start rating {START_RATING}.")


def main():
    p = argparse.ArgumentParser(description="Football Elo rankings from a results CSV.")
    p.add_argument("--out-dir", type=Path, default=Path("."), help="where to write outputs")
    p.add_argument("--min-games", type=int, default=0,
                   help="hide teams with fewer games from rankings.csv")
    p.add_argument("--monthfirst", action="store_true",
                   help="read slash dates as MM/DD/YYYY (US style); default is UK DD/MM/YYYY")
    p.add_argument("--predict", nargs=2, metavar=("TEAM1", "TEAM2"),
                   help="show the expected outcome of a match after processing")
    args = p.parse_args()

    # ELO_RESULTS_FILE overrides the sheet with a local CSV (handy for testing).
    source = os.environ.get("ELO_RESULTS_FILE") or RESULTS_URL
    df = load_results(source, dayfirst=not args.monthfirst)
    teams, history = run(df)
    table = build_table(teams, args.min_games)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "rankings.csv", index=False)
    history.to_csv(args.out_dir / "rating_history.csv", index=False)
    predictions = build_predictions(teams, table["team"].tolist())
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)

    if table.empty:
        print("No games recorded yet.")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(table.to_string(index=False))
    print(f"\n{len(df)} matches, {len(teams)} teams. "
          f"Wrote rankings.csv, rating_history.csv and predictions.csv "
          f"({len(predictions)} pairings) to {args.out_dir.resolve()}")

    if args.predict:
        predict(teams, *args.predict)


if __name__ == "__main__":
    main()
