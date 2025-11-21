# Win Probability vs Win Probability Added: Temporal Dynamics Analysis

## Your Insight: Can Non-Scoring Events Be Valued?

**Question:** If we forward-fill score margin, will rebounds, fouls, and assists that don't directly change the score still be properly valued by the model?

**Short Answer:** Yes, but through a different mechanism than direct score changes. The model can learn the causal chain indirectly through the WPA calculation.

## The Two-Layer System

### 1. **Win Probability (WP) Model** - Current Game State
- **Purpose:** Given current game state (score margin, time remaining), what's the probability of winning?
- **Input:** `score_margin`, `seconds_remaining`, `is_home`, `period`
- **Output:** Win probability (0-1)
- **Forward-filling is CORRECT here** because WP should reflect the CURRENT game state at any moment

### 2. **Win Probability Added (WPA) Calculation** - Change Attribution
- **Purpose:** How much did this specific action change the win probability?
- **Formula:** `WPA = WP_after - WP_before`
- **Forward-filling enables this** because we can track how actions affect future game states

## How WPA Captures Non-Scoring Events

Looking at the code in `models/clutch_score.py`:

```python
# For each event except the last, WP_after = next event's WP_before
for i in range(len(game_indices) - 1):
    current_idx = game_indices[i]
    next_idx = game_indices[i + 1]
    wpa_df.loc[current_idx, 'WP_after'] = wpa_df.loc[next_idx, 'WP_before']
```

### Example: Rebound → Score Chain

**Scenario:** Team down by 5, 2 minutes left, gets defensive rebound

```
Event N:   Defensive Rebound (margin=-5, time=120s)
  WP_before = WP(margin=-5, time=120s) = 0.15 (15% chance to win)
  
Event N+1: Made Shot (margin=-3, time=119s)  
  WP_before = WP(margin=-3, time=119s) = 0.22 (22% chance to win)
  
Event N+2: Made Shot (margin=-1, time=118s)
  WP_before = WP(margin=-1, time=118s) = 0.30 (30% chance to win)
```

**WPA Calculation:**
- Rebound: `WP_after = 0.22` (next event's WP), so `WPA = 0.22 - 0.15 = +0.07`
- The rebound gets credit for the score increase, even though it didn't directly change the score!

## Can the Model Learn This?

### ✅ **Yes, Through Temporal Connection**

The model learns indirectly:

1. **Direct Learning (Scoring Events):**
   - Made shot with margin=+5 → margin=+7
   - Model sees: `WP(margin=+5) → WP(margin=+7)`
   - Large positive WPA

2. **Indirect Learning (Non-Scoring Events):**
   - Rebound with margin=+5 → next event often has margin=+7 (if team scores)
   - Model sees: `WP(margin=+5) → WP(margin=+7)`
   - Also gets positive WPA through the temporal chain

3. **The Key:** WPA is calculated as `WP_after - WP_before`, where `WP_after = next_event.WP_before`
   - Even if margin is forward-filled, the rebound's `WP_after` captures the score change that happens in the next event
   - The attribution system (in `attribute_wpa_to_players`) gives the rebound player credit

### ⚠️ **Limitation: Current Model Type**

**Logistic Regression (current model):**
- ✅ Can learn: "Higher margin → higher WP"
- ✅ Can learn: "Less time → margin matters more"
- ❌ Cannot explicitly learn: "Rebounds at margin=+5 lead to margin=+7 more often"
- ✅ But it doesn't need to! The WPA calculation already captures this

**Why it works:**
- The WP model doesn't need to know about rebounds specifically
- It just needs to know: "margin=+7 is better than margin=+5"
- The WPA attribution system handles giving credit to rebounds

## The Causal Chain Problem

### Your Observation is Correct

Non-scoring events have **indirect, delayed effects**:
- Rebound → Next possession → Potential score → Margin change
- Foul → Free throws → Potential score → Margin change
- Assist → Made shot → Score → Margin change

### Forward-Filling Doesn't Break This

**With forward-filled margin:**

```
Event 1: Rebound (margin=+5, time=120s)
  WP_before = 0.65
  
Event 2: Made Shot (margin=+7, time=119s)
  WP_before = 0.72
  
Rebound WPA = 0.72 - 0.65 = +0.07 ✅
```

**Without forward-filling (current broken state):**

```
Event 1: Rebound (margin=0, time=120s)  ← WRONG!
  WP_before = 0.50 (no signal)
  
Event 2: Made Shot (margin=+7, time=119s)
  WP_before = 0.72
  
Rebound WPA = 0.72 - 0.50 = +0.22 (overestimated due to wrong baseline)
```

## Will It "Even Out" Over Large Data?

### ❌ **Not for WP Model Training**

The WP model needs to see the CURRENT score state at each moment. If 74% of events have margin=0, the model can't learn:
- "When margin=+10, home team wins 80% of the time"
- "When margin=-5 with 2 min left, away team wins 30% of the time"

It can only learn: "When margin=0, home team wins 62% of the time" (just the base rate).

### ✅ **Yes for WPA Attribution**

Over large data, rebounds/fouls will show patterns:
- Defensive rebounds → next possession often leads to score increase
- Offensive fouls → turnover → opponent score increase
- The WPA attribution system will capture these patterns

## Recommended Approach

### Phase 1: Fix WP Model (Current Priority)
1. **Forward-fill score margin** so WP model sees correct game state
2. **Retrain WP model** to learn proper margin → WP relationships
3. **Expected AUC: 0.65-0.85** (vs current 0.52)

### Phase 2: Enhance WPA Attribution (Future)
1. **Possession-level features:**
   - Who has possession? (rebound → team gets possession)
   - Shot clock time
   - Recent possession outcomes

2. **Sequential modeling:**
   - LSTM/Transformer to learn: "rebound → next possession → score"
   - Could explicitly model the causal chain

3. **Expected WPA attribution improvements:**
   - More accurate credit for rebounds (currently works but could be better)
   - Better handling of assists (currently not fully attributed)
   - Context-aware foul valuation

## Summary

**Your insight is correct:** Non-scoring events have indirect effects that take time to manifest.

**Forward-filling is still necessary** because:
1. WP model needs correct current game state (forward-fill fixes this)
2. WPA attribution already captures indirect effects through temporal connection
3. The model doesn't need to explicitly learn causal chains (WPA handles it)

**Future improvements:**
- Sequential models could learn explicit causal chains
- Possession-level features would improve attribution
- But the current approach (forward-fill + WPA attribution) should work reasonably well

**Bottom line:** Forward-fill margin now to fix WP model. The WPA attribution will properly value non-scoring events through the temporal connection. Enhancements can come later.

