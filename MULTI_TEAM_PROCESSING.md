# Multi-Team and Multi-Season Processing Guide

## Overview

The Clutch system now supports processing play-by-play data for all NBA teams across multiple seasons. This document explains how the system handles duplicate games (same game from different team perspectives) and ensures correct WPA attribution.

## Key Features

### 1. Team Perspective Tracking
- Each row in the WPA data includes `tracked_team_abbrev` and `tracked_team_name` to identify which team's perspective the data is from
- This allows the system to distinguish between the same game processed from different team perspectives

### 2. WPA Attribution Logic
- WPA is calculated from the **tracked team's perspective**
- When the tracked team is away, `WPA_weighted` is flipped BEFORE attribution
- This ensures all `player_wpa` values are from the tracked team's perspective
- Attribution rules (positive for good plays, negative for bad plays) work correctly regardless of home/away status

### 3. Deduplication
- When processing multiple teams, the same game appears from different perspectives
- The `aggregate_player_wpa()` function can deduplicate by `(player, game_id)`
- For each `(player, game)` combination, the system prefers the row where the player is on the tracked team
- This ensures we use WPA from the player's own team's perspective

## Processing Workflow

### Single Team Processing
```bash
python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25" --retrain
```

This processes one team's data and generates:
- `playbyplay_wpa.csv`: Event-level WPA data
- `player_wpa_summary.csv`: Player-level aggregated metrics

### Multi-Team Processing

**Option 1: Process teams separately, then merge**
1. Process each team separately (same command as above, different teams)
2. Each team's output includes `tracked_team_abbrev` and `tracked_team_name`
3. When merging/aggregating across teams, use `aggregate_player_wpa()` with `deduplicate_by_game=True`

**Option 2: Use batch processing script**
```bash
python -m scripts.batch_collect_all_teams --seasons 5
```

Then process each team's data separately.

### Aggregation Across Teams

When you have WPA data from multiple teams (same games from different perspectives):

```python
from models.clutch_score import ClutchScoreCalculator

# Load WPA data from multiple teams
all_wpa_data = pd.concat([
    pd.read_csv('data/processed/team1_playbyplay_wpa.csv'),
    pd.read_csv('data/processed/team2_playbyplay_wpa.csv'),
    # ... more teams
])

# Aggregate with deduplication
calculator = ClutchScoreCalculator(None)  # WP model not needed for aggregation
player_stats = calculator.aggregate_player_wpa(all_wpa_data, deduplicate_by_game=True)
```

## How Deduplication Works

When `deduplicate_by_game=True`:

1. **Identify tracked team rows**: For each `(player, game)` combination, identify rows where `PLAYER1_TEAM_ABBREVIATION == tracked_team_abbrev`

2. **Prefer tracked team perspective**: Sort rows so tracked team rows come first, then take the first row per `(player, game)` group

3. **Deduplicate event counts**: Since the same events appear from both perspectives, event counts are deduplicated to avoid double-counting

4. **Aggregate player stats**: Sum WPA, average leverage, count unique games, etc.

## Example: Team A vs Team B Game

When processing "Team A vs Team B":

### From Team A's perspective:
- Team A player scores: `player_wpa > 0` (helps Team A)
- Team B player scores: `player_wpa < 0` (hurts Team A)
- Team A player turnover: `player_wpa < 0` (hurts Team A)

### From Team B's perspective:
- Team A player scores: `player_wpa < 0` (hurts Team B)
- Team B player scores: `player_wpa > 0` (helps Team B)
- Team B player turnover: `player_wpa < 0` (hurts Team B)

### When aggregating:
- For Team A players: Use WPA from Team A perspective (where they're tracked team)
- For Team B players: Use WPA from Team B perspective (where they're tracked team)

## Data Schema

### Event-Level WPA Data (`playbyplay_wpa.csv`)
- `tracked_team_abbrev`: Team abbreviation (e.g., "IND")
- `tracked_team_name`: Full team name (e.g., "Indiana Pacers")
- `tracked_team_is_home`: Boolean indicating if tracked team is home for this game
- `player_wpa`: WPA from tracked team's perspective
- All other WPA-related columns

### Player-Level Summary (`player_wpa_summary.csv`)
- `player_name`: Player name
- `player_total_WPA`: Sum of all WPA (deduplicated by game)
- `player_avg_WPA`: Mean WPA per event
- `player_event_count`: Number of events (deduplicated)
- `games_played`: Unique games (deduplicated)
- `player_leverage_index`: Mean leverage weight

## Best Practices

1. **Process teams separately**: Run `compute_clutch_scores` for each team separately to maintain clear team perspectives

2. **Use deduplication when merging**: Always use `deduplicate_by_game=True` when aggregating across multiple teams

3. **Preserve team context**: Keep `tracked_team_abbrev` and `tracked_team_name` in your data for traceability

4. **Verify event counts**: After deduplication, verify that event counts are reasonable (not double-counted)

5. **Check player totals**: When aggregating, ensure player WPA totals make sense (e.g., players shouldn't have impossibly high/low totals)

## Troubleshooting

### Issue: Double-counted games
**Solution**: Ensure `deduplicate_by_game=True` when aggregating across teams

### Issue: Wrong WPA signs for away teams
**Solution**: Verify that `tracked_team_is_home` is correctly set and WPA attribution logic is working

### Issue: Missing team context
**Solution**: Ensure `tracked_team_abbrev` and `tracked_team_name` are being saved in WPA data

### Issue: Event counts too high
**Solution**: Check that deduplication is working correctly - event counts should be the same per `(player, game)` regardless of perspective

## Future Enhancements

- Batch processing script that automatically deduplicates across teams
- Database storage with proper indexing for faster queries
- Incremental processing (only process new games)
- Parallel processing for multiple teams/seasons

