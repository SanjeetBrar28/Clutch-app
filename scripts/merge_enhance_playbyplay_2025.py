#!/usr/bin/env python3
"""
Merge and enhance 2024-25 play-by-play data for all NBA teams.

This script:
1. Loads every processed team CSV for the 2024-25 season.
2. Adds league-wide feature columns ready for model training.
3. Deduplicates overlapping events and standardises schema.
4. Saves a unified dataset plus summary statistics.
"""

import logging
import math
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams


PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("data/logs")
OUTPUT_PATH = PROCESSED_DIR / "playbyplay_2025_all_teams_enhanced.csv"
LOG_PATH = LOG_DIR / "merge_enhance_2025.log"
SEASON_LABEL = "2024-25"
TOTAL_REGULATION_SECONDS = 48 * 60
TEAM_ID_MAP_PATH = Path("data/team_id_map.json")
TEAM_ABBR_TO_ID: Optional[Dict[str, int]] = None


def load_team_abbr_map() -> Dict[str, int]:
    """Load or build a mapping of team abbreviations to IDs."""
    global TEAM_ABBR_TO_ID
    if TEAM_ABBR_TO_ID:
        return TEAM_ABBR_TO_ID
    
    TEAM_ID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TEAM_ID_MAP_PATH.exists():
        with open(TEAM_ID_MAP_PATH, "r") as f:
            TEAM_ABBR_TO_ID = {k.upper(): int(v) for k, v in json.load(f).items()}
    else:
        teams = nba_teams.get_teams()
        TEAM_ABBR_TO_ID = {team["abbreviation"].upper(): int(team["id"]) for team in teams}
        with open(TEAM_ID_MAP_PATH, "w") as f:
            json.dump(TEAM_ABBR_TO_ID, f, indent=2)
    return TEAM_ABBR_TO_ID


def setup_logging() -> None:
    """Configure logging to file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler()
        ],
    )


def gather_input_files() -> List[Path]:
    """Return all processed team CSVs for the 2024-25 season."""
    pattern = "playbyplay_*_2024_25_cleaned.csv"
    files = sorted(PROCESSED_DIR.glob(pattern))
    logging.info("Located %d candidate files with pattern '%s'", len(files), pattern)
    if not files:
        logging.warning("No files found. Did you run the batch collection pipeline?")
    return files


def build_team_mappings() -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    Build dictionaries for mapping team IDs and slugs to abbreviations.

    Returns:
        Tuple of (team_id_to_abbrev, team_slug_to_abbrev)
    """
    teams_list = nba_teams.get_teams()
    id_to_abbrev = {team["id"]: team["abbreviation"] for team in teams_list}
    slug_to_abbrev = {}
    for team in teams_list:
        slug = team["full_name"].lower().replace(" ", "_").replace(".", "")
        slug_to_abbrev[slug] = team["abbreviation"]
        slug_to_abbrev[team["abbreviation"].lower()] = team["abbreviation"]
    logging.info("Loaded %d team mappings from nba_api", len(teams_list))
    return id_to_abbrev, slug_to_abbrev


def load_and_concat(files: List[Path], slug_to_abbrev: Dict[str, str]) -> Tuple[pd.DataFrame, int]:
    """
    Load all files, annotate with source team, and concatenate.

    Returns:
        Tuple of (combined_dataframe, duplicates_removed_count)
    """
    frames = []
    total_rows = 0
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Failed to load %s: %s", path.name, exc)
            continue

        total_rows += len(df)
        team_slug = path.name.replace("playbyplay_", "").replace("_2024_25_cleaned.csv", "")
        df["SOURCE_TEAM"] = team_slug
        df["SOURCE_TEAM_ABBREV"] = slug_to_abbrev.get(team_slug.lower(), "UNKNOWN")
        df["SEASON"] = SEASON_LABEL
        frames.append(df)
        logging.info("Loaded %s (%d rows)", path.name, len(df))

    if not frames:
        logging.error("No data loaded. Exiting early.")
        return pd.DataFrame(), 0

    combined = pd.concat(frames, ignore_index=True, sort=False)
    before = len(combined)
    
    # When deduplicating, prefer keeping events from the team's own perspective
    # For each (GAME_ID, EVENTNUM), if both teams have the same event, we want to keep
    # the one where the event's player is on the tracked team (where tracked_team_is_home matches)
    # But actually, for model training, we don't need both perspectives - we need ONE canonical
    # event per (GAME_ID, EVENTNUM). However, we should prefer keeping the perspective where
    # the player involved is on the tracked team.
    
    # Strategy: Sort by SOURCE_TEAM (consistent order), then keep first per (GAME_ID, EVENTNUM)
    # This ensures we always keep the same team's perspective for consistency
    # Actually better: For each (GAME_ID, EVENTNUM), if PLAYER1_NAME matches SOURCE_TEAM players,
    # prefer that row. Otherwise, just take first.
    
    # For now, simpler approach: Sort by SOURCE_TEAM alphabetically, then drop duplicates
    # This ensures consistent deduplication but may lose some team-specific events
    # Better would be to check if PLAYER1_TEAM_ABBREVIATION matches SOURCE_TEAM_ABBREV
    
    if "PLAYER1_TEAM_ABBREVIATION" in combined.columns and "SOURCE_TEAM_ABBREV" in combined.columns:
        # Add a preference column: prefer rows where player is on source team
        combined["_prefer"] = (
            combined["PLAYER1_TEAM_ABBREVIATION"] == combined["SOURCE_TEAM_ABBREV"]
        ).astype(int)
        # Sort so preferred rows come first, then by SOURCE_TEAM for consistency
        combined = combined.sort_values(
            ["_prefer", "SOURCE_TEAM"], ascending=[False, True]
        )
        combined = combined.drop(columns=["_prefer"])
    
    combined = combined.drop_duplicates(subset=["GAME_ID", "EVENTNUM"], keep="first")
    duplicates_removed = before - len(combined)
    logging.info(
        "Concatenated %d rows across %d files (removed %d duplicates)",
        before,
        len(frames),
        duplicates_removed,
    )
    return combined, duplicates_removed


def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Cast selected columns to numeric, coercing errors to NaN."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def extract_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Populate home_score and away_score from SCORE_HOME/AWAY or descriptions."""
    if "SCORE_HOME" in df.columns and "SCORE_AWAY" in df.columns:
        df["home_score"] = pd.to_numeric(df["SCORE_HOME"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["SCORE_AWAY"], errors="coerce")
        df["home_score"] = df.groupby("GAME_ID")["home_score"].ffill().bfill()
        df["away_score"] = df.groupby("GAME_ID")["away_score"].ffill().bfill()
    else:
        score_pattern = re.compile(r"(\d+)\s*-\s*(\d+)")

        def parse_score(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
            text = f"{row.get('HOMEDESCRIPTION', '')} {row.get('VISITORDESCRIPTION', '')}"
            match = score_pattern.search(str(text))
            if match:
                return float(match.group(1)), float(match.group(2))
            return (np.nan, np.nan)

        scores = df.apply(parse_score, axis=1, result_type="expand")
        df["home_score"] = scores[0]
        df["away_score"] = scores[1]
        df["home_score"] = df.groupby("GAME_ID")["home_score"].ffill().bfill()
        df["away_score"] = df.groupby("GAME_ID")["away_score"].ffill().bfill()
    return df


def lookup_player_ids(df: pd.DataFrame) -> pd.Series:
    """Map PLAYER1_NAME to NBA player ID using nba_api, with caching."""
    cache: Dict[str, Optional[int]] = {}

    def fetch_id(name: str) -> Optional[int]:
        if not name or name.upper() == "UNKNOWN":
            return None
        key = name.strip().lower()
        if key in cache:
            return cache[key]
        matches = nba_players.find_players_by_full_name(name)
        player_id = matches[0]["id"] if matches else None
        cache[key] = player_id
        return player_id

    return df["PLAYER1_NAME"].apply(fetch_id)


def determine_opponent(df: pd.DataFrame, id_to_abbrev: Dict[int, str]) -> pd.Series:
    """Infer opponent abbreviation using team IDs and event team."""
    home_abbrev = df["TEAM_ID_HOME"].map(id_to_abbrev)
    away_abbrev = df["TEAM_ID_AWAY"].map(id_to_abbrev)

    event_team = df["PLAYER1_TEAM_ABBREVIATION"].replace({np.nan: "UNKNOWN"})
    event_team = np.where(
        event_team.eq("UNKNOWN") | event_team.eq(""),
        df["SOURCE_TEAM_ABBREV"],
        event_team,
    )

    opponent = np.where(
        event_team == home_abbrev,
        away_abbrev,
        np.where(event_team == away_abbrev, home_abbrev, np.nan),
    )
    opponent = np.where(pd.isna(opponent), away_abbrev, opponent)
    opponent = np.where(pd.isna(opponent), home_abbrev, opponent)
    opponent = np.where(pd.isna(opponent), "UNKNOWN", opponent)
    return opponent


def infer_shot_and_points(df: pd.DataFrame) -> pd.DataFrame:
    """Create shot_made and points_scored columns."""
    descriptions = (
        df["HOMEDESCRIPTION"].fillna("").str.cat(df["VISITORDESCRIPTION"].fillna(""), sep=" ")
    )
    descriptions_lower = descriptions.str.lower()

    df["shot_made"] = np.where(
        (df["event_category"] == "SHOT")
        & (
            descriptions_lower.str.contains("makes")
            | descriptions_lower.str.contains("made")
            | descriptions_lower.str.contains("pts")
        )
        & (~descriptions_lower.str.contains("miss")),
        1,
        0,
    )

    def derive_points(row, desc_lower: str) -> int:
        event_category = getattr(row, "event_category", None)
        if event_category == "FREE_THROW":
            if "miss" in desc_lower or "misses" in desc_lower:
                return 0
            return 1
        shot_made_flag = getattr(row, "shot_made", 0)
        if shot_made_flag == 1:
            if "3pt" in desc_lower or "3-pt" in desc_lower:
                return 3
            return 2
        return 0

    df["points_scored"] = [
        derive_points(row, desc_lower)
        for row, desc_lower in zip(df.itertuples(index=False), descriptions_lower.tolist())
    ]
    return df


def infer_possession(df: pd.DataFrame) -> pd.DataFrame:
    """Assign possession_team and possession_change columns per game."""
    def process_game(game_df: pd.DataFrame) -> pd.DataFrame:
        last_offensive_team = None
        possession = []
        change_flags = []

        for row in game_df.itertuples(index=False):
            event_team = getattr(row, "PLAYER1_TEAM_ABBREVIATION", "UNKNOWN")
            if not event_team or str(event_team).upper() == "UNKNOWN":
                event_team = getattr(row, "SOURCE_TEAM_ABBREV", "UNKNOWN")

            if row.event_category in {"SHOT", "TURNOVER", "FOUL", "FREE_THROW"} and event_team != "UNKNOWN":
                last_offensive_team = event_team

            possession.append(last_offensive_team)

            descriptions = f"{getattr(row, 'HOMEDESCRIPTION', '')} {getattr(row, 'VISITORDESCRIPTION', '')}".lower()
            is_def_rebound = (
                row.event_category == "REBOUND"
                and ("defensive" in descriptions)
            )
            change_flags.append(
                (row.event_category == "TURNOVER")
                or (getattr(row, "shot_made", 0) == 1)
                or is_def_rebound
            )

        game_df = game_df.copy()
        game_df["possession_team"] = possession
        game_df["possession_change"] = change_flags
        return game_df

    df = df.groupby("GAME_ID", group_keys=False).apply(process_game)
    return df


def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns specified in requirements."""
    df["GAME_TYPE"] = "Regular"
    df["GAME_DATE"] = ""
    df["TEAM_ABBREV_OPPONENT"] = determine_opponent(df, TEAM_ID_TO_ABBREV)
    df = extract_scores(df)
    df = infer_shot_and_points(df)
    df = infer_possession(df)

    # Ensure TEAM_ID_EVENT is always populated (in case older processed files lack new column)
    if 'TEAM_ID_EVENT' not in df.columns:
        df['TEAM_ID_EVENT'] = pd.NA

    if df['TEAM_ID_EVENT'].isna().any():
        if 'TEAM_ID' in df.columns:
            df['TEAM_ID_EVENT'] = df['TEAM_ID_EVENT'].fillna(df['TEAM_ID'])
        if 'PLAYER1_TEAM_ID' in df.columns:
            df['TEAM_ID_EVENT'] = df['TEAM_ID_EVENT'].fillna(df['PLAYER1_TEAM_ID'])
        if df['TEAM_ID_EVENT'].isna().any() and 'PLAYER1_TEAM_ABBREVIATION' in df.columns:
            team_map = load_team_abbr_map()
            df['PLAYER1_TEAM_ABBREVIATION'] = df['PLAYER1_TEAM_ABBREVIATION'].str.upper()
            df['TEAM_ID_EVENT'] = df['TEAM_ID_EVENT'].fillna(
                df['PLAYER1_TEAM_ABBREVIATION'].map(team_map)
            )
    if df['TEAM_ID_EVENT'].isna().any() and 'is_home' in df.columns:
        fallback_values = pd.Series(
            np.where(
                df['is_home'].astype(bool),
                df['TEAM_ID_HOME'],
                df['TEAM_ID_AWAY']
            ),
            index=df.index
        )
        df['TEAM_ID_EVENT'] = df['TEAM_ID_EVENT'].fillna(fallback_values)
    if df['TEAM_ID_EVENT'].isna().any():
        df['TEAM_ID_EVENT'] = df['TEAM_ID_EVENT'].fillna(df['TEAM_ID_HOME'])
    if df['TEAM_ID_EVENT'].isna().any():
        missing = df[df['TEAM_ID_EVENT'].isna()][['GAME_ID', 'EVENTNUM']].head()
        raise ValueError(
            "TEAM_ID_EVENT still missing after inference. Re-run processing with updated logic. "
            f"Sample rows:\n{missing}"
        )
    df['TEAM_ID_EVENT'] = df['TEAM_ID_EVENT'].astype('Int64')
    df["turnover_flag"] = df["event_category"].eq("TURNOVER")
    df["foul_flag"] = df["event_category"].eq("FOUL")
    df["abs_score_margin"] = df["score_margin_int"].abs()
    df["game_clock_ratio"] = df["total_seconds_remaining"] / TOTAL_REGULATION_SECONDS
    df["game_clock_ratio"] = df["game_clock_ratio"].clip(lower=0, upper=1)

    df["leverage_index"] = (1 / (df["abs_score_margin"] + 1)) * (1 - df["game_clock_ratio"])
    df["leverage_index"] = df["leverage_index"].fillna(0)

    df["PLAYER1_NAME"] = df["PLAYER1_NAME"].fillna("UNKNOWN")
    df["PLAYER1_TEAM_ABBREVIATION"] = df["PLAYER1_TEAM_ABBREVIATION"].fillna("UNKNOWN")
    df["PLAYER1_ID"] = lookup_player_ids(df)


    return df


def finalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dtypes and ordering."""
    numeric_columns = [
        "EVENTNUM",
        "PERIOD",
        "total_seconds_remaining",
        "score_margin_int",
        "points_scored",
        "TEAM_ID_EVENT",
    ]
    df = ensure_numeric(df, numeric_columns)
    df["points_scored"] = df["points_scored"].fillna(0).astype(int)
    df["shot_made"] = df["shot_made"].fillna(0).astype(int)
    df["turnover_flag"] = df["turnover_flag"].fillna(False).astype(bool)
    df["foul_flag"] = df["foul_flag"].fillna(False).astype(bool)
    df["possession_change"] = df["possession_change"].fillna(False).astype(bool)

    sort_columns = ["GAME_ID", "EVENTNUM"] if {"GAME_ID", "EVENTNUM"}.issubset(df.columns) else df.columns
    df = df.sort_values(sort_columns).reset_index(drop=True)
    return df


def summarise(df: pd.DataFrame, duplicates_removed: int) -> None:
    """Print summary statistics to stdout and logs."""
    total_rows = len(df)
    unique_games = df["GAME_ID"].nunique()
    unique_players = df["PLAYER1_NAME"].nunique()
    unique_teams = df["SOURCE_TEAM"].nunique()
    avg_points_per_game = (
        df.groupby("GAME_ID")["points_scored"].sum().mean()
        if unique_games and "points_scored" in df.columns
        else 0
    )

    summary_lines = [
        "=== Merge Summary ===",
        f"Total rows/events: {total_rows:,}",
        f"Unique games: {unique_games:,}",
        f"Unique players: {unique_players:,}",
        f"Unique source teams: {unique_teams:,}",
        f"Average points scored per game: {avg_points_per_game:.2f}",
        f"Duplicates removed: {duplicates_removed:,}",
        f"Output path: {OUTPUT_PATH}",
    ]

    for line in summary_lines:
        logging.info(line)
        print(line)


def main() -> None:
    """Entry-point."""
    setup_logging()
    logging.info("Starting merge and enhancement pipeline for %s", SEASON_LABEL)

    files = gather_input_files()
    if not files:
        logging.error("No input files found. Aborting.")
        return

    df, duplicates_removed = load_and_concat(files, TEAM_SLUG_TO_ABBREV)
    if df.empty:
        logging.error("Combined dataframe empty, nothing to do.")
        return

    df = compute_additional_features(df)
    df = finalise_dataframe(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info("Saved merged dataset to %s (%d rows)", OUTPUT_PATH, len(df))

    summarise(df, duplicates_removed)


if __name__ == "__main__":
    TEAM_ID_TO_ABBREV, TEAM_SLUG_TO_ABBREV = build_team_mappings()
    main()

