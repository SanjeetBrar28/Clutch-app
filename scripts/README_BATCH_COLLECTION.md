# Batch Data Collection Guide

## Overview

The `batch_collect_all_teams.py` script collects and processes play-by-play data for all NBA teams across multiple seasons. This is essential for training a robust Win Probability model with league-wide data.

## Why Batch Collection?

**Problem**: Training on only one team (Pacers) resulted in poor AUC (0.52) because:
- Limited training data (~100 games)
- Home/away bias (team identity confounded with home court)
- Can't learn general win probability patterns

**Solution**: Collect data for all 30 teams across 5 seasons (~15,000 games) to:
- Train on diverse game contexts
- Separate home court advantage from team quality
- Improve model performance (target: AUC 0.65-0.75+)

## Usage

### Basic Usage (All Teams, 5 Seasons)

```bash
# Collect last 5 seasons for all 30 teams
python -m scripts.batch_collect_all_teams --seasons 5
```

**Estimated Time**: ~25 hours (running continuously)
- 30 teams × 5 seasons = 150 team-seasons
- ~100 games per team/season × 0.6s per game = 60s per team/season
- Total: ~2.5 hours per season × 5 = ~12.5 hours minimum
- Plus processing time: ~25 hours total

### Test with Specific Teams First

```bash
# Test with 2-3 teams first to verify everything works
python -m scripts.batch_collect_all_teams --seasons 3 --teams "Lakers" "Warriors" "Celtics"
```

### Resume After Interruption

The script automatically skips existing data:

```bash
# Run again - it will skip what's already fetched
python -m scripts.batch_collect_all_teams --seasons 5
```

### Only Process Existing Raw Data

If you already have raw data and just want to process it:

```bash
python -m scripts.batch_collect_all_teams --only-process
```

### Force Re-fetch Everything

```bash
# Re-fetch even if data exists
python -m scripts.batch_collect_all_teams --seasons 5 --force-refetch
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--seasons` | Number of seasons to collect | `5` |
| `--teams` | Specific teams (name or abbreviation) | All 30 teams |
| `--raw-dir` | Directory for raw play-by-play files | `data/raw/playbyplay` |
| `--processed-dir` | Directory for processed files | `data/processed` |
| `--rate-limit` | Delay between API calls (seconds) | `0.6` |
| `--max-retries` | Max retry attempts per game | `3` |
| `--skip-fetch` | Skip fetching (only process) | `False` |
| `--only-process` | Only process existing raw data | `False` |
| `--force-refetch` | Re-fetch even if exists | `False` |
| `--force-reprocess` | Re-process even if exists | `False` |

## Data Structure

### Raw Data
```
data/raw/playbyplay/
├── indiana_pacers/
│   ├── 2024_25/
│   │   ├── 0012300001.csv
│   │   ├── 0012300002.csv
│   │   └── ...
│   ├── 2023_24/
│   └── ...
├── los_angeles_lakers/
│   ├── 2024_25/
│   └── ...
└── ...
```

### Processed Data
```
data/processed/
├── playbyplay_indiana_pacers_2024_25_cleaned.csv
├── playbyplay_indiana_pacers_2023_24_cleaned.csv
├── playbyplay_los_angeles_lakers_2024_25_cleaned.csv
└── ...
```

## Recommended Workflow

### Phase 1: Small Test (30 minutes)
```bash
# Test with 2 teams, 1 season
python -m scripts.batch_collect_all_teams --seasons 1 --teams "Lakers" "Warriors"
```

**Verify**:
- ✅ Data is being fetched correctly
- ✅ Files are saved in correct structure
- ✅ Processing works without errors
- ✅ Output files are valid

### Phase 2: Single Season for All Teams (5 hours)
```bash
# Collect just 2024-25 season for all teams
python -m scripts.batch_collect_all_teams --seasons 1
```

**Benefits**:
- Tests the full pipeline on all teams
- Produces enough data to train a better model
- Can validate improvements before full collection

### Phase 3: Full Collection (Run Overnight)
```bash
# Collect all 5 seasons for all teams
# Run this overnight or over a weekend
python -m scripts.batch_collect_all_teams --seasons 5
```

**Tips**:
- Start on Friday evening, let it run over the weekend
- Monitor logs in `data/logs/batch_collect_*.log`
- Script can be interrupted and resumed (it skips existing data)

## Progress Tracking

The script shows:
- Real-time progress bar with current team/season
- Status indicators (✓ = success, ✗ = failed, skipped = already exists)
- Summary statistics at the end

### Logs

All activity is logged to:
```
data/logs/batch_collect_YYYYMMDD_HHMMSS.log
```

Check logs for:
- Failed fetches and errors
- Rate limit issues
- API connection problems

## Error Handling

The script handles errors gracefully:
- **API Failures**: Retries up to `--max-retries` times
- **Network Issues**: Logs error and continues with next team/season
- **Missing Data**: Skips gracefully if raw data is missing

**If failures occur**:
1. Check logs for error messages
2. Re-run the script (it will skip successful fetches)
3. For persistent failures, fetch those teams/seasons individually:
   ```bash
   python -m scripts.fetch_pbp_season_v3 --team "Team Name" --season "2024-25"
   ```

## Storage Requirements

**Estimated storage**:
- Raw data: ~50-100 MB per team/season = ~75-150 GB total (5 seasons × 30 teams)
- Processed data: ~10-20 MB per team/season = ~1.5-3 GB total

**Ensure you have enough disk space!**

## Next Steps After Collection

Once all data is collected:

1. **Merge all processed files** into a single dataset:
   ```python
   # TODO: Create merge script
   # Combines all team/season CSVs into one file
   ```

2. **Train improved WP model**:
   ```bash
   python -m scripts.compute_clutch_scores --team "ALL" --season "2019-2024" --retrain
   ```

3. **Validate improvements**:
   - AUC should improve to 0.65-0.75+
   - Better player rankings
   - More reliable WPA scores

## Troubleshooting

### Rate Limiting Issues
If you get rate limit errors, increase the delay:
```bash
python -m scripts.batch_collect_all_teams --seasons 5 --rate-limit 1.0
```

### Out of Memory
Processing can be memory-intensive. If you run out of memory:
- Process in batches (fewer teams/seasons at once)
- Or process after each team/season fetch

### API Connection Issues
If API is temporarily unavailable:
- Script will log errors and continue
- Re-run later to fetch missing data
- Check NBA API status if persistent

## Summary

**Quick Start**:
```bash
# Test first
python -m scripts.batch_collect_all_teams --seasons 1 --teams "Lakers" "Warriors"

# Then full collection (run overnight)
python -m scripts.batch_collect_all_teams --seasons 5
```

**Expected Outcome**:
- ~15,000 games of play-by-play data
- Improved Win Probability model (AUC 0.65-0.75+)
- More reliable player WPA scores
- Better player rankings and insights
