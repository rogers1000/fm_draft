"""
FM Draft dashboard — results, rankings and predictions.

Reads every game live from the Google Sheet (RESULTS_URL in fm_draft_elo.py) and
calculates ratings with fm_draft_elo.py, so all three tabs are always current and
consistent. If the sheet can't be reached, falls back to draft_game_results.csv
in the repo root.

Run locally from the repo root:
    pip install -r streamlitdashboard/requirements.txt
    streamlit run streamlitdashboard/dashboard.py

Test with another results file instead of the sheet (from the repo root):
    ELO_RESULTS_FILE=dummy_draft_game_results.csv streamlit run streamlitdashboard/dashboard.py
  Windows PowerShell:
    $env:ELO_RESULTS_FILE="dummy_draft_game_results.csv"; streamlit run streamlitdashboard/dashboard.py

Streamlit Community Cloud: main file path = streamlitdashboard/dashboard.py
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# fm_draft_elo.py lives in the repo root, one level up from this folder.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import fm_draft_elo as elo  # noqa: E402

LOCAL_RESULTS = REPO_ROOT / "draft_game_results.csv"
CACHE_SECONDS = 300  # re-read the sheet at most every 5 minutes

# Column names are matched case-insensitively, ignoring spaces/underscores.
TEAM1_NAMES = ("team1", "home", "hometeam")
TEAM2_NAMES = ("team2", "away", "awayteam")
PROVISIONAL_NAMES = ("either_provisional", "provisional")

# Readable headers: column name → (label, hover tooltip). Columns not listed
# get an automatic label ("peak_rating" → "Peak Rating").
LABELS = {
    # Predictions
    "team1": ("Team 1", None),
    "team2": ("Team 2", None),
    "team1_rating": ("Team 1 Rating", "Current Elo rating"),
    "team2_rating": ("Team 2 Rating", "Current Elo rating"),
    "team1_win_expectancy": ("Team 1 Win Chance", "Expected result for Team 1, with a draw counted as half a win"),
    "team2_win_expectancy": ("Team 2 Win Chance", "Expected result for Team 2, with a draw counted as half a win"),
    "expected_gd_team1": ("Expected Goal Difference", "Which team the ratings expect to win, and by how many goals"),
    "favourite": ("Favourite", "'Too close to call' when the win chances are within 45–55%"),
    "either_provisional": ("Provisional?", "Ticked if either team is still provisional (fewer than 15 games), so the prediction is less reliable"),
    # Rankings
    "rank": ("Rank", None),
    "team": ("Team", None),
    "rating": ("Rating", "Current Elo rating (everyone starts at 1500)"),
    "played": ("Played", None),
    "W": ("Won", None),
    "D": ("Drawn", None),
    "L": ("Lost", None),
    "GF": ("Goals For", None),
    "GA": ("Goals Against", None),
    "GD": ("Goal Difference", None),
    "peak_rating": ("Peak Rating", "Highest rating this team has reached"),
    "provisional": ("Provisional?", "Fewer than 15 games played, so the rating is still settling"),
    "last_played": ("Last Played", None),
    # Draft game results
    "date": ("Date", None),
    "game_order": ("Game Order", "Order of games played on the same day"),
    "goals1": ("Team 1 Score", None),
    "goals2": ("Team 2 Score", None),
}
RESULTS_COLUMN_ORDER = ["date", "game_order", "team1", "goals1", "goals2", "team2"]
PCT_COLS = {"team1_win_expectancy", "team2_win_expectancy"}


def label(col: str) -> str:
    return LABELS.get(col, (str(col).replace("_", " ").title(), None))[0]


def column_config(df: pd.DataFrame) -> dict:
    """Readable header, tooltip and number/date format for every column."""
    cfg = {}
    for col in df.columns:
        name = label(col)
        tip = LABELS.get(col, (None, None))[1]
        if col in PCT_COLS:
            cfg[col] = st.column_config.NumberColumn(name, help=tip, format="percent")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            cfg[col] = st.column_config.DateColumn(name, help=tip, format="DD MMM YYYY")
        else:
            cfg[col] = st.column_config.Column(name, help=tip)
    return cfg


# ---------------------------------------------------------------------------
# Helpers (pure pandas — no Streamlit calls)
# ---------------------------------------------------------------------------
def _norm(name: str) -> str:
    return str(name).lower().replace(" ", "").replace("_", "")


def find_col(df: pd.DataFrame, candidates) -> str | None:
    lookup = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in lookup:
            return lookup[_norm(cand)]
    return None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            if "date" in col.lower() or col.lower() == "last_played":
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
    return df


def team_list(df: pd.DataFrame, c1: str, c2: str) -> list[str]:
    teams = pd.concat([df[c1], df[c2]]).dropna().astype(str).str.strip()
    return sorted(teams.unique(), key=str.lower)


def filter_teams(df: pd.DataFrame, c1: str, c2: str, teams: list[str]) -> pd.DataFrame:
    """
    No teams  → every row.
    One team  → every row that team is in, as Team 1 or Team 2.
    Two+ teams → head to head: only rows where BOTH sides are selected teams.
    """
    if not teams:
        return df
    side1 = df[c1].astype(str).str.strip().isin(teams)
    side2 = df[c2].astype(str).str.strip().isin(teams)
    return df[side1 | side2] if len(teams) == 1 else df[side1 & side2]


def head_to_head(results: pd.DataFrame, a: str, b: str) -> dict:
    """Record between two teams from a results table (goals1/goals2 columns)."""
    rec = {"played": 0, a: 0, "draws": 0, b: 0, f"{a}_goals": 0, f"{b}_goals": 0}
    for t1, t2, g1, g2 in results[["team1", "team2", "goals1", "goals2"]].itertuples(index=False):
        ga, gb = (g1, g2) if t1 == a else (g2, g1)
        rec["played"] += 1
        rec[f"{a}_goals"] += ga
        rec[f"{b}_goals"] += gb
        rec[a if ga > gb else b if gb > ga else "draws"] += 1
    return rec


def filter_value(df: pd.DataFrame, col: str, choice: str) -> pd.DataFrame:
    if choice == "All":
        return df
    return df[df[col].astype(str) == choice]


def describe_gd(gd: float, team1: str, team2: str) -> str:
    """Team 1-relative goal difference → e.g. 'Kingsmead wins by 0.31' (always positive)."""
    if pd.isna(gd):
        return ""
    if abs(gd) < 0.005:
        return "Level"
    winner = team1 if gd > 0 else team2
    return f"{winner} wins by {abs(gd):.2f}"


def compute(raw: pd.DataFrame):
    """Run the Elo system on cleaned results → (results, rankings, predictions)."""
    teams, _history = elo.run(raw)
    rankings = elo.build_table(teams, min_games=0)
    predictions = elo.build_predictions(teams, rankings["team"].tolist())
    predictions["expected_gd_team1"] = [
        describe_gd(gd, t1, t2)
        for gd, t1, t2 in zip(predictions["expected_gd_team1"], predictions["team1"], predictions["team2"])
    ]
    results = raw.drop(columns=["csv_row"], errors="ignore")
    return clean(results), clean(rankings), clean(predictions)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Loading results…")
def load_all():
    note = None
    test_file = os.environ.get("ELO_RESULTS_FILE")  # same override fm_draft_elo.py uses
    if test_file:
        raw = elo.load_results(test_file, dayfirst=True)
        results, rankings, predictions = compute(raw)
        note = f"Test mode: showing {Path(test_file).name} (ELO_RESULTS_FILE is set), not the Google Sheet."
        return results, rankings, predictions, Path(test_file).name, note, datetime.now(timezone.utc)
    try:
        raw = elo.load_results(elo.RESULTS_URL, dayfirst=True)
        source = "Google Sheet"
    except (Exception, SystemExit) as err:  # load_results calls sys.exit on bad columns
        raw = elo.load_results(str(LOCAL_RESULTS), dayfirst=True)
        source = f"repo copy ({LOCAL_RESULTS.name})"
        note = f"Couldn't read the Google Sheet ({err}). Showing {LOCAL_RESULTS.name} from the repo instead."
    results, rankings, predictions = compute(raw)
    loaded_at = datetime.now(timezone.utc)
    return results, rankings, predictions, source, note, loaded_at


# ---------------------------------------------------------------------------
# Streamlit pieces
# ---------------------------------------------------------------------------
def show_table(df: pd.DataFrame, total: int, name: str, key: str):
    st.caption(f"Showing **{len(df):,}** of {total:,} rows")
    st.dataframe(df, width="stretch", hide_index=True, column_config=column_config(df))
    st.download_button(
        "Download filtered CSV",
        df.rename(columns=label).to_csv(index=False).encode("utf-8"),
        file_name=f"{name}_filtered.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )


def team_filter(df: pd.DataFrame, key: str, container) -> tuple[pd.DataFrame, list[str]]:
    c1, c2 = find_col(df, TEAM1_NAMES), find_col(df, TEAM2_NAMES)
    if not (c1 and c2):
        container.warning(f"Couldn't find team columns. Columns found: {', '.join(df.columns)}")
        return df, []
    teams = container.multiselect(
        "Team",
        team_list(df, c1, c2),
        placeholder="All teams — type to search",
        key=f"teams_{key}",
        help="One team: all its games, as Team 1 or Team 2. "
             "Two teams: head to head, only games between them.",
    )
    if len(teams) >= 2:
        container.caption("Head to head: " + " vs ".join(teams))
    return filter_teams(df, c1, c2, teams), teams


def show_head_to_head(df: pd.DataFrame, a: str, b: str):
    rec = head_to_head(df, a, b)
    if not rec["played"]:
        st.info(f"{a} and {b} haven't played each other yet.")
        return
    cols = st.columns(4)
    cols[0].metric("Played", rec["played"])
    cols[1].metric(f"{a} Wins", rec[a])
    cols[2].metric("Draws", rec["draws"])
    cols[3].metric(f"{b} Wins", rec[b])
    st.caption(f"Goals: {a} {rec[f'{a}_goals']} – {rec[f'{b}_goals']} {b}")


def generic_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """One filter widget per column, chosen by data type."""
    out = df
    cols = st.columns(3)
    for i, col in enumerate(df.columns):
        box = cols[i % 3]
        s = df[col]
        wkey = f"{key}_{col}"

        if pd.api.types.is_bool_dtype(s):
            pick = box.multiselect(label(col), [True, False], placeholder="All", key=wkey,
                                   format_func=lambda v: "Yes" if v else "No")
            if pick:
                out = out[out[col].isin(pick)]

        elif pd.api.types.is_numeric_dtype(s):
            lo, hi = s.min(), s.max()
            if pd.isna(lo) or lo == hi:
                continue
            is_int = pd.api.types.is_integer_dtype(s)
            lo, hi = (int(lo), int(hi)) if is_int else (float(lo), float(hi))
            rng = box.slider(label(col), lo, hi, (lo, hi), key=wkey)
            if rng != (lo, hi):
                out = out[out[col].between(*rng)]

        elif pd.api.types.is_datetime64_any_dtype(s):
            lo, hi = s.min(), s.max()
            if pd.isna(lo):
                continue
            rng = box.date_input(label(col), (lo.date(), hi.date()), key=wkey)
            if len(rng) == 2 and rng != (lo.date(), hi.date()):
                out = out[out[col].between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1))]

        elif s.nunique() <= 100:
            opts = sorted(s.dropna().astype(str).unique(), key=str.lower)
            pick = box.multiselect(label(col), opts, placeholder="All", key=wkey)
            if pick:
                out = out[out[col].astype(str).isin(pick)]

        else:
            text = box.text_input(label(col), placeholder="Contains…", key=wkey)
            if text:
                out = out[out[col].astype(str).str.contains(text, case=False, na=False, regex=False)]
    return out


def empty(df: pd.DataFrame) -> bool:
    if df.empty:
        st.info("No games recorded yet.")
        return True
    return False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="FM Draft Dashboard", page_icon="🏆", layout="wide")
    st.title("🏆 FM Draft Dashboard")

    results, rankings, predictions, source, note, loaded_at = load_all()

    info, button = st.columns([4, 1])
    info.caption(
        f"{len(results):,} matches · {len(rankings):,} teams · from {source} · "
        f"loaded {loaded_at:%H:%M} UTC (refreshes every {CACHE_SECONDS // 60} min)"
    )
    if button.button("↻ Refresh now", width="stretch"):
        load_all.clear()
        st.rerun()
    if note:
        st.warning(note)

    tab_rank, tab_pred, tab_res = st.tabs(["Rankings", "Predictions", "Results"])

    # --- Tab 1 · Rankings: every column -----------------------------------
    with tab_rank:
        if not empty(rankings):
            with st.expander("Filters", expanded=True):
                df = generic_filters(rankings, "rank")
            show_table(df, len(rankings), "rankings", "rank")

    # --- Tab 2 · Predictions: team (either column) + provisional ----------
    with tab_pred:
        if not empty(predictions):
            left, right = st.columns([3, 2])
            df, _teams = team_filter(predictions, "pred", left)
            prov = find_col(predictions, PROVISIONAL_NAMES)
            if prov:
                values = sorted(predictions[prov].dropna().astype(str).unique())
                choice = right.radio(
                    "Either team provisional", ["All", *values], horizontal=True, key="prov",
                    format_func=lambda v: {"True": "Yes", "False": "No"}.get(v, v),
                )
                df = filter_value(df, prov, choice)
            show_table(df, len(predictions), "predictions", "pred")

    # --- Tab 3 · Results: team (either column) ----------------------------
    with tab_res:
        if not empty(results):
            left, _ = st.columns([3, 2])
            df, teams = team_filter(results, "res", left)
            if len(teams) == 2:
                show_head_to_head(df, *teams)
            order = [c for c in ("date", "game_order") if c in df.columns]
            if order:
                df = df.sort_values(order, ascending=False)
            # Date, Game Order, Team 1, Team 1 Score, Team 2 Score, Team 2 (anything else after)
            first = [c for c in RESULTS_COLUMN_ORDER if c in df.columns]
            df = df[first + [c for c in df.columns if c not in first]]
            show_table(df, len(results), "results", "res")


if __name__ == "__main__":
    main()
