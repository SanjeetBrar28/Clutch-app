#!/usr/bin/env python3
"""
Fetch and process play-by-play data for every NBA team in the 2024-25 season.

This wraps the existing batch_collect helper functions but locks the season to 2024-25
so we can rebuild the league-wide dataset reliably.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from nba_api.stats.static import teams as nba_teams

# Reuse processing helper but provide our own fetch implementation so we can skip playoffs.
from scripts.batch_collect_all_teams import check_data_exists, process_team_season  # type: ignore

from backend.services.data_ingestion import GameUtils, PlayByPlayFetcher


SEASON = "2024-25"
RAW_DIR = Path("data/raw/playbyplay")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("data/logs")
LOG_PATH = LOG_DIR / f"batch_fetch_process_{SEASON.replace('-', '_')}.log"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch + process NBA play-by-play for all teams in the 2024-25 season.",
    )
    parser.add_argument(
        "--teams",
        nargs="+",
        help="Optional list of team names or abbreviations to restrict processing.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.6,
        help="Delay between NBA API calls (seconds).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per game fetch.",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Re-fetch even if raw files already exist.",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Re-process even if cleaned file already exists.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )


def filter_teams(selected: List[str] = None) -> List[Dict]:
    """Return NBA team metadata, optionally filtered by name/abbreviation."""
    all_teams = nba_teams.get_teams()
    if not selected:
        return all_teams

    selected_lower = {item.lower() for item in selected}
    filtered = [
        team
        for team in all_teams
        if team["full_name"].lower() in selected_lower
        or team["abbreviation"].lower() in selected_lower
    ]

    missing = selected_lower.difference(
        {team["full_name"].lower() for team in filtered}.union(
            {team["abbreviation"].lower() for team in filtered}
        )
    )
    if missing:
        logging.warning("Requested teams not found: %s", ", ".join(sorted(missing)))

    return filtered


def summarise(results: List[Tuple[str, Dict, Dict]]) -> None:
    """Print and log a summary after processing."""
    total_teams = len(results)
    fetch_success = sum(1 for _, fetch_res, _ in results if fetch_res.get("success"))
    process_success = sum(1 for _, _, process_res in results if process_res.get("success"))
    skipped_fetch = sum(1 for _, fetch_res, _ in results if fetch_res.get("skipped"))
    skipped_process = sum(1 for _, _, process_res in results if process_res.get("skipped"))

    logging.info("=== SUMMARY (%s season) ===", SEASON)
    logging.info("Teams attempted: %d", total_teams)
    logging.info("Fetch success: %d (skipped: %d)", fetch_success, skipped_fetch)
    logging.info("Process success: %d (skipped: %d)", process_success, skipped_process)

    failed_fetch = [(team, res) for team, res, _ in results if not res.get("success")]
    failed_process = [(team, res) for team, _, res in results if not res.get("success")]

    if failed_fetch:
        logging.warning("Fetch failures:")
        for team, res in failed_fetch:
            logging.warning("  - %s: %s", team, res.get("error", "Unknown error"))

    if failed_process:
        logging.warning("Process failures:")
        for team, res in failed_process:
            logging.warning("  - %s: %s", team, res.get("error", "Unknown error"))


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_regular_season_games(
    team: Dict,
    season: str,
    raw_dir: str,
    rate_limit: float,
    max_retries: int,
    skip_existing: bool = True,
) -> Dict:
    """
    Fetch only regular-season games (GAME_ID starts with '002') for the given team.
    Mirrors fetch_team_season but excludes playoff games which frequently time out.
    """
    team_name = team["full_name"]
    result = {
        "team": team_name,
        "season": season,
        "success": False,
        "games_fetched": 0,
        "games_failed": 0,
        "total_plays": 0,
        "error": None,
        "skipped": False,
    }

    try:
        raw_exists, _ = check_data_exists(team_name, season, raw_dir, "")
        if skip_existing and raw_exists:
            result["skipped"] = True
            result["success"] = True

            safe_team_name = team_name.replace(" ", "_").replace(".", "").lower()
            safe_season = season.replace("-", "_")
            raw_dir_path = Path(raw_dir) / safe_team_name / safe_season
            if raw_dir_path.exists():
                result["games_fetched"] = len(list(raw_dir_path.glob("*.csv")))
            return result

        game_utils = GameUtils(rate_limit_delay=rate_limit)
        fetcher = PlayByPlayFetcher(output_dir=raw_dir, rate_limit_delay=rate_limit)

        games = game_utils.fetch_game_ids(team_name, season, max_retries=max_retries)
        if not games:
            result["error"] = "No games found"
            return result

        regular_games = [game for game in games if str(game["GAME_ID"]).startswith("002")]
        logging.info(
            "Fetched %d game IDs for %s (%d regular-season games, %d skipped as non-regular)",
            len(games),
            team_name,
            len(regular_games),
            len(games) - len(regular_games),
        )

        if not regular_games:
            result["error"] = "No regular-season games found after filtering"
            return result

        for game in regular_games:
            game_id = game["GAME_ID"]
            df = fetcher.fetch_game_playbyplay(game_id, max_retries=max_retries)
            if df is None or df.empty:
                result["games_failed"] += 1
                continue
            fetcher.save_game_data(df, game_id, team_name, season)
            result["games_fetched"] += 1
            result["total_plays"] += len(df)

        result["success"] = result["games_fetched"] > 0
        if result["games_failed"]:
            result["error"] = f"{result['games_failed']} games failed"

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Unexpected error fetching %s: %s", team_name, exc)
        result["error"] = str(exc)

    return result


def main() -> None:
    args = parse_arguments()
    setup_logging()
    ensure_directories()

    logging.info("Starting 2024-25 league-wide fetch/process pipeline at %s", datetime.now())
    logging.info("Rate limit: %.2fs | Max retries: %d", args.rate_limit, args.max_retries)

    teams_to_run = filter_teams(args.teams)
    logging.info("Teams to process: %d", len(teams_to_run))

    results: List[Tuple[str, Dict, Dict]] = []

    for team in teams_to_run:
        team_name = team["full_name"]
        logging.info("=== %s ===", team_name)

        # Fetch step
        fetch_res = fetch_regular_season_games(
            team,
            SEASON,
            RAW_DIR.as_posix(),
            args.rate_limit,
            args.max_retries,
            skip_existing=not args.force_fetch,
        )

        if fetch_res.get("success"):
            logging.info(
                "Fetch complete: %d games fetched (skipped=%s)",
                fetch_res.get("games_fetched", 0),
                fetch_res.get("skipped"),
            )
        else:
            logging.error("Fetch failed for %s: %s", team_name, fetch_res.get("error"))

        # Process step only if fetch succeeded or raw data already existed.
        process_res = process_team_season(
            team,
            SEASON,
            RAW_DIR.as_posix(),
            PROCESSED_DIR.as_posix(),
            skip_existing=not args.force_process,
        )

        if process_res.get("success"):
            logging.info(
                "Process complete: %d rows across %d games",
                process_res.get("rows_processed", 0),
                process_res.get("games_processed", 0),
            )
        else:
            logging.error("Process failed for %s: %s", team_name, process_res.get("error"))

        results.append((team_name, fetch_res, process_res))

    summarise(results)


if __name__ == "__main__":
    main()

