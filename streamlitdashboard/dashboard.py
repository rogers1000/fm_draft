"""
FM Draft dashboard — results, rankings and predictions.

Reads every game live from the Google Sheet (RESULTS_URL in fm_draft_elo.py) and
calculates ratings with fm_draft_elo.py, so all three tabs are always current and
consistent. If the sheet can't be reached, falls back to draft_game_results.csv
in the repo root.

Run locally from the repo root:
    pip install -r streamlitdashboard/requirements.txt
    streamlit run streamlitdashboard/dashboard.py

Streamlit Community Cloud: main file path = streamlitdashboard/dashboard.py
"""

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

# Nicer display for known columns (unknown columns display as-is).
PCT = st.column_config.NumberColumn(format="percent")
COLUMN_CONFIG = {
    "team1_win_expectancy": PCT,
    "team2_win_expectancy": PCT,
    "expected_gd_team1": st.column_config.NumberColumn(format="%+.2f"),
    "date": st.column_config.DateColumn(format="DD MMM YYYY"),
    "last_played": st.column_config.DateColumn(format="DD MMM YYYY"),
}


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
    """Keep rows where any selected team appears in EITHER team column."""
    if not teams:
        return df
    mask = df[c1].astype(str).str.strip().isin(teams) | df[c2].astype(str).str.strip().isin(teams)
    return df[mask]


def filter_value(df: pd.DataFrame, col: str, choice: str) -> pd.DataFrame:
    if choice == "All":
        return df
    return df[df[col].astype(str) == choice]


def compute(raw: pd.DataFrame):
    """Run the Elo system on cleaned results → (results, rankings, predictions)."""
    teams, _history = elo.run(raw)
    rankings = elo.build_table(teams, min_games=0)
    predictions = elo.build_predictions(teams, rankings["team"].tolist())
    results = raw.drop(columns=["csv_row"], errors="ignore")
    return clean(results), clean(rankings), clean(predictions)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Loading results…")
def load_all():
    note = None
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
    config = {c: cfg for c, cfg in COLUMN_CONFIG.items() if c in df.columns}
    st.dataframe(df, width="stretch", hide_index=True, column_config=config)
    st.download_button(
        "Download filtered CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}_filtered.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )


def team_filter(df: pd.DataFrame, key: str, container) -> pd.DataFrame:
    c1, c2 = find_col(df, TEAM1_NAMES), find_col(df, TEAM2_NAMES)
    if not (c1 and c2):
        container.warning(f"Couldn't find team columns. Columns found: {', '.join(df.columns)}")
        return df
    teams = container.multiselect(
        "Team (matches team1 or team2)",
        team_list(df, c1, c2),
        placeholder="All teams — type to search",
        key=f"teams_{key}",
    )
    return filter_teams(df, c1, c2, teams)


def generic_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """One filter widget per column, chosen by data type."""
    out = df
    cols = st.columns(3)
    for i, col in enumerate(df.columns):
        box = cols[i % 3]
        s = df[col]
        wkey = f"{key}_{col}"

        if pd.api.types.is_bool_dtype(s):
            pick = box.multiselect(col, [True, False], placeholder="All", key=wkey)
            if pick:
                out = out[out[col].isin(pick)]

        elif pd.api.types.is_numeric_dtype(s):
            lo, hi = s.min(), s.max()
            if pd.isna(lo) or lo == hi:
                continue
            is_int = pd.api.types.is_integer_dtype(s)
            lo, hi = (int(lo), int(hi)) if is_int else (float(lo), float(hi))
            rng = box.slider(col, lo, hi, (lo, hi), key=wkey)
            if rng != (lo, hi):
                out = out[out[col].between(*rng)]

        elif pd.api.types.is_datetime64_any_dtype(s):
            lo, hi = s.min(), s.max()
            if pd.isna(lo):
                continue
            rng = box.date_input(col, (lo.date(), hi.date()), key=wkey)
            if len(rng) == 2 and rng != (lo.date(), hi.date()):
                out = out[out[col].between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1))]

        elif s.nunique() <= 100:
            opts = sorted(s.dropna().astype(str).unique(), key=str.lower)
            pick = box.multiselect(col, opts, placeholder="All", key=wkey)
            if pick:
                out = out[out[col].astype(str).isin(pick)]

        else:
            text = box.text_input(col, placeholder="Contains…", key=wkey)
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

    tab_pred, tab_rank, tab_res = st.tabs(["Predictions", "Rankings", "Draft game results"])

    # --- Predictions: team (either column) + provisional -------------------
    with tab_pred:
        if not empty(predictions):
            left, right = st.columns([3, 2])
            df = team_filter(predictions, "pred", left)
            prov = find_col(predictions, PROVISIONAL_NAMES)
            if prov:
                values = sorted(predictions[prov].dropna().astype(str).unique())
                choice = right.radio(
                    "Either team provisional", ["All", *values], horizontal=True, key="prov",
                    format_func=lambda v: {"True": "Yes", "False": "No"}.get(v, v),
                )
                df = filter_value(df, prov, choice)
            show_table(df, len(predictions), "predictions", "pred")

    # --- Rankings: every column -------------------------------------------
    with tab_rank:
        if not empty(rankings):
            with st.expander("Filters", expanded=True):
                df = generic_filters(rankings, "rank")
            show_table(df, len(rankings), "rankings", "rank")

    # --- Draft game results: team (either column) --------------------------
    with tab_res:
        if not empty(results):
            left, _ = st.columns([3, 2])
            df = team_filter(results, "res", left)
            order = [c for c in ("date", "game_order") if c in df.columns]
            if order:
                df = df.sort_values(order, ascending=False)
            show_table(df, len(results), "draft_game_results", "res")


if __name__ == "__main__":
    main()
