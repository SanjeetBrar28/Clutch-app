# Win Probability Model Diagnostic Analysis

## Summary

**AUC: 0.5205** (barely better than random 0.5) - Model is essentially not learning any predictive signal.

## Key Findings

### 1. ⚠️ **CRITICAL: Score Margin Distribution Problem**

```
Positive margins:  7,445  (15.2%)
Negative margins:  5,124  (10.5%)
Zero margins:     36,343  (74.3%)  ← PROBLEM!
```

**Root Cause:** The `score_margin_int` column is only updated when a score changes. Most events (74.3%) have `score_margin_int = 0` because they don't directly change the score (rebounds, fouls, turnovers, missed shots, etc.).

**Impact:** The model can't learn from score margin for 74% of the data. When the margin is 0, the model has no signal about which team is leading.

### 2. ⚠️ **Model Always Predicts Home Win**

```
Confusion Matrix:
  True Negatives:  0      ← Model never predicts away win (0)
  False Positives: 4,776  ← Predicted home, but away won
  False Negatives: 0      ← Model never predicts away win
  True Positives:  10,064 ← Correctly predicted home win
```

**Diagnosis:** The model learned to always predict `home_win=1` because:
- The label distribution is skewed (62% home wins)
- When margin is 0 (74% of data), the model defaults to the majority class
- The model can't distinguish between home/away when margin is 0

### 3. ✓ **Good News: Data Quality is OK**

- ✅ Label consistency: Each game has exactly one `home_win` value
- ✅ Margin sign symmetry: Both positive and negative margins exist
- ✅ Time monotonicity: `total_seconds_remaining` decreases correctly
- ✅ No missing values in key columns

### 4. ⚠️ **Feature-Target Relationship Broken**

The model is predicting probabilities in a very narrow range:
- **Predicted values:** [0.5934, 0.6023] - very little variation
- **Actual fractions:** [0.6713, 0.6978] - higher than predicted

This suggests the model is learning a constant (home team bias) rather than using score margin and time.

## Root Cause Analysis

### Primary Issue: Score Margin Not Forward-Filled

The `score_margin_int` column in the processed data is only updated on scoring events. For non-scoring events (rebounds, turnovers, fouls, etc.), it remains 0 or the last score value.

**Example from the data:**
```
Row 4:  score_margin_int=2   (made shot - score updated)
Row 5:  score_margin_int=0   (made shot - score updated to tie)
Row 6:  score_margin_int=0   (missed shot - margin stays 0)
Row 7:  score_margin_int=0   (rebound - margin stays 0)
Row 8:  score_margin_int=3   (made shot - score updated)
```

**What should happen:**
```
Row 4:  score_margin_int=2   (made shot)
Row 5:  score_margin_int=0   (made shot - tied)
Row 6:  score_margin_int=0   (missed shot - still tied)
Row 7:  score_margin_int=0   (rebound - still tied)
Row 8:  score_margin_int=3   (made shot - away leads by 3)
```

### Secondary Issue: Score Margin Perspective

The `score_margin_int` is from the **tracked team's perspective** (Indiana Pacers), not from the home team's perspective. The conversion logic tries to fix this, but when the margin is 0 (74% of the time), it doesn't matter.

## Recommended Fixes

### Fix #1: Forward-Fill Score Margin (HIGH PRIORITY)

Add a step to forward-fill `score_margin_int` within each game so every event has the current score margin:

```python
# In data processing pipeline
df['score_margin_int'] = df.groupby('GAME_ID')['score_margin_int'].fillna(method='ffill')
# Or use .ffill() for newer pandas
df['score_margin_int'] = df.groupby('GAME_ID')['score_margin_int'].ffill()
```

This ensures every event knows the current score state, not just scoring events.

### Fix #2: Verify Score Margin Calculation

Check that `score_margin_int` is calculated correctly:
- When home team scores, margin should increase
- When away team scores, margin should decrease
- The margin should reflect `home_score - away_score`

### Fix #3: Feature Engineering

Consider adding:
- **Score margin per minute remaining**: `score_margin / (seconds_remaining / 60)`
- **Momentum features**: Recent score changes (last N possessions)
- **Possession features**: Who has the ball, time on shot clock

### Fix #4: Model Training

Once score margin is forward-filled:
1. The model should see meaningful margin values for all events
2. AUC should improve significantly (target > 0.7)
3. The model should learn that positive margins favor home team

## Expected Results After Fix

- **AUC should increase to 0.65-0.85** (depending on data quality)
- **Model should predict both home and away wins** (not just home)
- **Confusion matrix should be balanced** (not all predictions = 1)
- **Score margin should be the most important feature**

## Next Steps

1. ✅ Verify score margin calculation in data processing
2. ✅ Add forward-fill step for score margin
3. ✅ Re-run diagnostic script
4. ✅ Retrain WP model
5. ✅ Validate AUC improves to > 0.7

