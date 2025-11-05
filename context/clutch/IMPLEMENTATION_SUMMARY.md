# Clutch WP/WPA Implementation Summary

**Date:** 2025-01-29  
**Status:** ✅ Complete

---

## Implementation Overview

Successfully implemented the Win Probability (WP) and Win Probability Added (WPA) baseline model for Clutch v1.0. All modules are complete and ready for testing.

---

## Files Created

### 1. `features/build_leverage_features.py`
- **Purpose:** Compute leverage index and contextual weights for WPA calculations
- **Key Functions:**
  - `compute_leverage_weight()`: Calculates leverage weight based on score margin and time remaining
  - `build_wp_features()`: Builds feature matrix for WP model (score_margin, seconds_remaining, is_home, period, interaction)
  - `validate_features()`: Validates feature DataFrame completeness

### 2. `models/wp_model.py`
- **Purpose:** Train and evaluate logistic regression Win Probability model
- **Key Classes:**
  - `WinProbabilityModel`: Main WP model class
  - Methods: `train()`, `predict()`, `evaluate_calibration()`, `plot_calibration()`, `save_model()`, `load_model()`
- **Features:** Logistic regression with one-hot encoded period, calibration plotting, model persistence

### 3. `models/clutch_score.py`
- **Purpose:** Compute per-event WPA and aggregate to player-level scores
- **Key Classes:**
  - `ClutchScoreCalculator`: Main WPA calculation class
  - Methods: `compute_event_wpa()`, `attribute_wpa_to_players()`, `aggregate_player_wpa()`, `save_wpa_data()`, `save_player_summary()`
- **Features:** Event-level WPA calculation, player attribution logic, player-level aggregation

### 4. `scripts/compute_clutch_scores.py`
- **Purpose:** End-to-end orchestrator for WP/WPA pipeline
- **Key Functions:**
  - `fetch_game_outcomes()`: Fetches game outcomes from NBA API
  - `infer_game_outcomes_from_data()`: Fallback for inferring outcomes from data
  - `main()`: Main orchestrator function
- **Features:** CLI interface, model training/loading, WPA computation, output generation, calibration plots

### 5. `context/clutch/metrics.md`
- **Purpose:** Comprehensive documentation of WP/WPA metrics v1.0
- **Contents:**
  - WP definition and model specification
  - WPA formula and leverage weighting
  - Player attribution rules
  - Example calculations
  - Validation methods
  - Future roadmap

---

## Dependencies Added

Updated `requirements.txt` with:
- `scikit-learn>=1.3.0,<2.0.0` - Machine learning library
- `matplotlib>=3.7.0` - Plotting for calibration curves
- `joblib>=1.3.0` - Model persistence

---

## Usage

### Basic Usage

```bash
# Compute Clutch Scores for Indiana Pacers 2024-25 season
python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25"
```

### Advanced Usage

```bash
# Retrain WP model
python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25" --retrain

# Use custom data path
python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25" \
  --data-path data/processed/playbyplay_indiana_pacers_2024_25_cleaned.csv

# Use inferred outcomes (skip API calls)
python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25" \
  --no-use-api-outcomes
```

---

## Output Files

The pipeline generates:

1. **`data/processed/playbyplay_wpa.csv`**
   - Event-level WPA data with columns:
     - `WP_before`, `WP_after`, `delta_wp`
     - `leverage_weight`, `WPA_weighted`, `player_wpa`
     - `attribution_type`

2. **`data/processed/player_wpa_summary.csv`**
   - Player-level aggregated metrics:
     - `player_total_WPA`, `player_avg_WPA`
     - `player_leverage_index`, `player_event_count`, `games_played`

3. **`data/processed/wp_calibration_plot.png`**
   - Calibration curve showing predicted vs actual win rates

4. **`models/artifacts/win_prob_model.pkl`**
   - Saved trained WP model (for reuse)

---

## Next Steps

1. **Test the pipeline:**
   ```bash
   python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25"
   ```

2. **Validate outputs:**
   - Check that WP values are reasonable (0-1 range, S-shaped curves)
   - Verify player rankings match intuitive expectations
   - Review calibration plot for accuracy

3. **Iterate on model:**
   - Adjust leverage weighting parameters if needed
   - Refine attribution rules
   - Add more features (opponent strength, lineup effects)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  scripts/compute_clutch_scores.py (Orchestrator)            │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ features/       │ │ models/     │ │ context/clutch/ │
│ build_leverage_ │ │ wp_model.py │ │ metrics.md      │
│ features.py     │ │             │ │                 │
└─────────────────┘ │ clutch_     │ └──────────────────┘
                    │ score.py    │
                    └─────────────┘
```

---

## Key Design Decisions

1. **Modular Structure:** Each module has a single, clear responsibility
2. **Baseline First:** Started with logistic regression (simple, interpretable)
3. **Leverage Weighting:** Emphasizes late-game, close-score situations
4. **Player Attribution:** Simple rules first, can be refined later
5. **Validation:** Calibration plots and baseline comparisons

---

## Known Limitations

1. **Assist Attribution:** Not implemented in v1.0 (planned for v1.1)
2. **Defensive Events:** Limited coverage (steals, blocks not fully captured)
3. **Lineup Effects:** No adjustment for teammate quality
4. **Model Simplicity:** Logistic regression is baseline (future: XGBoost, neural networks)

---

## Testing Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run pipeline: `python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25"`
- [ ] Verify outputs exist: `playbyplay_wpa.csv`, `player_wpa_summary.csv`
- [ ] Check WP values are reasonable (0-1 range)
- [ ] Review calibration plot
- [ ] Validate player rankings (top players should rank highly)
- [ ] Check for errors in logs

---

## Status

✅ **All modules implemented and ready for testing**

Next action: Run the pipeline and validate outputs against success criteria.

