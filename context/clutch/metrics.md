# Clutch Metrics v1.0

## Overview

This document defines the Win Probability (WP) and Win Probability Added (WPA) metrics used in Clutch v1.0. These metrics form the statistical foundation for quantifying player impact on winning.

**Version:** 1.0  
**Date:** 2025-01-29  
**Status:** Active

---

## Win Probability (WP)

### Definition

Win Probability (WP) is the estimated probability that the home team will win the game, given the current game state.

**Formula:**
```
WP = P(home_win | score_margin, time_remaining, period, is_home)
```

### Model Specification

**Baseline Model:** Logistic Regression (v1.0)

**Features:**
- `score_margin`: Score margin from home team perspective (points)
- `seconds_remaining`: Total seconds remaining in game
- `is_home`: Binary indicator (1 if home team, 0 if away)
- `period`: Game period (1-4 for regulation, 5+ for overtime)
- `margin_time_interaction`: Score margin / sqrt(seconds_remaining + 1)

**Target:**
- `home_win`: Binary outcome (1 if home team won, 0 if away team won)

**Training:**
- Model trained on historical play-by-play data
- Uses logistic regression with L2 regularization
- Evaluated on holdout test set (20% split)
- Calibration validated via calibration curves

**Model Output:**
- `WP`: Probability between 0 and 1
- `WP = 0.5` means equal chance for both teams
- `WP = 0.75` means home team has 75% chance to win

---

## Win Probability Added (WPA)

### Definition

Win Probability Added (WPA) measures the change in win probability caused by a single event.

**Formula:**
```
WPA = (WP_after - WP_before) × leverage_weight
```

Where:
- `WP_before`: Win probability before the event
- `WP_after`: Win probability after the event
- `leverage_weight`: Contextual weight emphasizing late-game, close-score situations

### Leverage Weighting

To emphasize high-impact moments (late-game, close-score situations), we apply a leverage weight:

**Formula:**
```
leverage_weight = 1 + α × (1 / (abs(score_margin) + 1)) + β × (1 / (seconds_remaining / 720 + 1))
leverage_weight = clip(leverage_weight, 1.0, 3.0)
```

**Parameters (v1.0):**
- `α = 0.6`: Weight for score margin component
- `β = 0.4`: Weight for time remaining component
- `max_weight = 3.0`: Maximum leverage weight cap

**Interpretation:**
- `leverage_weight = 1.0`: Baseline (no emphasis)
- `leverage_weight = 2.0`: 2× emphasis on high-leverage moments
- `leverage_weight = 3.0`: 3× emphasis (maximum cap)

**Example:**
- Late 4th quarter, tied game: `leverage_weight ≈ 2.5`
- Early 1st quarter, 20-point lead: `leverage_weight ≈ 1.2`
- Final seconds, 1-point lead: `leverage_weight ≈ 3.0`

---

## Player Attribution Rules

### General Principle

WPA credit is assigned to players based on their actions during events. Attribution rules vary by event type.

### Attribution by Event Type

#### Made Shots
- **Shooter:** Gets full credit (1.0 × WPA)
- **Assister (future):** Gets 15% credit (0.15 × WPA) - *not implemented in v1.0*

#### Missed Shots
- **Shooter:** Gets negative credit (-1.0 × WPA) - missed opportunity cost

#### Free Throws
- **Made FT:** Shooter gets full credit (1.0 × WPA)
- **Missed FT:** Shooter gets negative credit (-0.5 × WPA)

#### Turnovers
- **Player committing turnover:** Gets full blame (-1.0 × WPA)

#### Rebounds
- **Rebounder:** Gets full credit (1.0 × WPA)

#### Fouls
- **Player committing foul:** Gets half blame (-0.5 × WPA)

#### Other Events
- **Primary player (PLAYER1_NAME):** Gets full credit if player identified, otherwise 0

### Attribution Logic

```python
# Made shot: shooter gets full credit
if event_category == 'SHOT' and made:
    player_wpa = WPA_weighted

# Missed shot: shooter gets negative credit
if event_category == 'MISS':
    player_wpa = -WPA_weighted

# Turnover: full blame
if event_category == 'TURNOVER':
    player_wpa = -WPA_weighted

# Rebound: full credit
if event_category == 'REBOUND':
    player_wpa = WPA_weighted

# Foul: half blame
if event_category == 'FOUL':
    player_wpa = -WPA_weighted * 0.5

# Other: default to PLAYER1_NAME if available
if event_category == 'OTHER' and PLAYER1_NAME:
    player_wpa = WPA_weighted
```

---

## Player Clutch Score

### Definition

Player Clutch Score aggregates WPA across all events for a player.

**Metrics:**
1. **Total WPA:** `player_total_WPA = sum(player_wpa)` across all events
2. **Average WPA:** `player_avg_WPA = mean(player_wpa)` per event
3. **Leverage Index:** `player_leverage_index = mean(leverage_weight)` for player's events
4. **Event Count:** `player_event_count` = number of events attributed to player
5. **Games Played:** `games_played` = number of games player appeared in

### Interpretation

**Positive WPA:**
- Player's actions increased team's win probability
- Higher values indicate greater positive impact

**Negative WPA:**
- Player's actions decreased team's win probability
- Lower values indicate greater negative impact

**Example:**
- Player A: `total_WPA = +2.5` → Added 2.5 wins worth of probability
- Player B: `total_WPA = -1.2` → Subtracted 1.2 wins worth of probability

**Leverage Index:**
- `leverage_index = 1.5` → Player's events occurred in moderate-leverage situations
- `leverage_index = 2.5` → Player's events occurred in high-leverage situations (clutch moments)

---

## Example Calculation Walkthrough

### Scenario: Clutch 3-Pointer

**Game State:**
- Score: Home 98, Away 100 (margin = -2)
- Time: 15 seconds remaining in 4th quarter
- Period: 4
- `is_home = True`

**Event:**
- Player makes a 3-pointer
- Score changes: Home 101, Away 100 (margin = +1)

**Calculation:**

1. **Compute WP_before:**
   - Features: `score_margin = -2`, `seconds_remaining = 15`, `period = 4`, `is_home = 1`
   - `WP_before = 0.35` (home team has 35% chance to win)

2. **Compute WP_after:**
   - Features: `score_margin = +1`, `seconds_remaining = 15`, `period = 4`, `is_home = 1`
   - `WP_after = 0.75` (home team now has 75% chance to win)

3. **Compute delta_wp:**
   - `delta_wp = 0.75 - 0.35 = 0.40` (40% increase in win probability)

4. **Compute leverage_weight:**
   - `leverage_weight = 1 + 0.6 × (1 / (|-2| + 1)) + 0.4 × (1 / (15/720 + 1))`
   - `leverage_weight = 1 + 0.6 × 0.33 + 0.4 × 0.98 ≈ 2.5`

5. **Compute WPA_weighted:**
   - `WPA_weighted = 0.40 × 2.5 = 1.00`

6. **Attribute to player:**
   - Made shot → shooter gets full credit
   - `player_wpa = 1.00`

**Result:** Player's clutch 3-pointer added 1.00 win probability (weighted), making it a highly impactful play.

---

## Validation & Calibration

### Model Calibration

The WP model is validated via calibration curves:
- **Perfect calibration:** Predicted WP should match actual win rates
- **Evaluation:** Plot predicted WP (x-axis) vs actual win rate (y-axis)
- **Target:** Points should fall close to diagonal line (y = x)

### Baseline Comparisons

Clutch Score is validated against traditional metrics:
- **Rank correlation:** Compare player rankings vs PER, BPM, +/- 
- **Expert priors:** All-stars should rank highly
- **Stability:** Scores should be consistent across games/seasons

### Data Quality Checks

- **Missing values:** All numeric fields filled with 0 if missing
- **Type enforcement:** All numeric fields cast to integers
- **Event filtering:** Empty/meaningless rows dropped
- **Team ID continuity:** Forward-fill for missing team IDs

---

## Future Roadmap

### v1.1 (Planned)
- **Assist attribution:** Implement 15% credit for assisters
- **Defensive events:** Add WPA for steals, blocks
- **Lineup effects:** Adjust for teammate quality

### v2.0 (Planned)
- **Model upgrade:** Replace logistic regression with gradient boosting (XGBoost)
- **Sequence modeling:** Add LSTM/Transformer for sequential dependencies
- **Contextual features:** Opponent strength, lineup synergy, player fatigue

### v3.0 (Future)
- **Neural WP model:** Deep learning architecture for win probability
- **Real-time inference:** Live game WP updates
- **Causal inference:** Causal attribution beyond correlation

---

## Assumptions & Limitations

### Assumptions

1. **Independence:** Events are treated as independent (future: sequential dependencies)
2. **Attribution:** Simple attribution rules (future: multi-player attribution)
3. **Home/Away:** Home court advantage captured in model, but not adjusted for opponent quality
4. **Leverage:** Leverage weight is heuristic (future: learned from data)

### Limitations

1. **Baseline model:** Logistic regression is simple (future: non-linear models)
2. **No lineup data:** Cannot adjust for teammate effects (future: lineup +/-)
3. **No defensive tracking:** Defensive events not fully captured (future: steals, blocks)
4. **Single season:** Trained on limited data (future: multi-season training)

---

## References

- **Win Probability:** Common in sports analytics (baseball, football, basketball)
- **WPA:** Similar to Win Probability Added in baseball (FanGraphs)
- **Clutch metrics:** Leverage-weighted performance metrics
- **Player impact:** Similar to Real Plus-Minus (RPM), Box Plus-Minus (BPM)

---

## Changelog

### v1.0 (2025-01-29)
- Initial release
- Baseline logistic regression WP model
- Event-level WPA calculation
- Player-level aggregation
- Leverage weighting

---

## Contact & Support

For questions or issues with metric definitions:
- See `context/clutch/decisions.md` for architectural decisions
- See `context/clutch/assumptions.md` for active assumptions
- See `context/clutch/journal/` for development logs

