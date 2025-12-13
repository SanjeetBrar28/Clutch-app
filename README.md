# Clutch — Event-Level Win Probability and Win Probability Added (WPA)

Clutch is a solo end-to-end sports analytics project that estimates event-level win probability (WP) from NBA play-by-play data and attributes player impact using Win Probability Added (WPA). The project compares a traditional baseline model with a sequential LSTM model that captures full-game context.

## Data Availability
Due to file size constraints, the NBA play-by-play data used in this project is not included in the repository. The modeling and evaluation scripts assume access to locally stored play-by-play data generated during development. The provided pipeline documents the full preprocessing, training, and evaluation workflow, but re-running the entire pipeline from scratch requires obtaining the same play-by-play data.

## Reproduing Results (Runnable Commands)

### 1. Fetch raw NBA play-by-play data
python scripts/fetch_pbp_season.py --season 2024

### 2. Merge, clean, and feature-engineer play-by-play data
Adds score differential tracking, time remaining, possession indicators, home/away consistency, and categorical vocabularies.
python scripts/merge_enhance_playbyplay.py

### 3. Build variable-length game sequences for LSTM training
Pads games to a common length and creates masks so padded events do not affect loss computation.
python scripts/build_lstm_sequences.py

### 4. Train baseline win probability model (logistic regression)
python scripts/train_wp_baseline.py

### 5. Train sequential LSTM win probability model
Uses masked binary cross-entropy, Adam optimization, and early stopping based on validation performance.
python scripts/train_lstm_wp.py

### 6. Generate win probability predictions and compute player WPA
Aggregates changes in win probability across player-involved events.
python scripts/evaluate_and_wpa.py --model lstm

(Optional baseline comparison)
python scripts/evaluate_and_wpa.py --model baseline

## Evaluation
Models are evaluated using ROC-AUC for predicting the eventual game winner from event-level win probability predictions. The LSTM model achieves higher AUC than the baseline, demonstrating the benefit of sequential modeling.

## Contribution
This project was completed entirely by a single contributor.

Data collection and preprocessing: Sole contributor  
Feature engineering and dataset construction: Sole contributor  
Baseline model implementation: Sole contributor  
LSTM model design and training: Sole contributor  
Evaluation, WPA attribution, and report writing: Sole contributor

