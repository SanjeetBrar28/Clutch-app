# Clutch MVP - Streamlit Dashboard

Quick MVP dashboard to display player Win Probability Added (WPA) scores for Indiana Pacers players.

## Setup

1. Install dependencies:
```bash
pip install streamlit plotly
# Or install all requirements:
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

- **Player Rankings**: Bar chart showing total WPA by player
- **WPA vs Events**: Scatter plot showing relationship between event count and WPA
- **Player Statistics Table**: Detailed stats for each player
- **Filters**: Filter by minimum games played and events

## Data Source

Uses data from:
- `data/processed/player_wpa_summary.csv`

Currently displays Pacers players from the 2024-25 season.

## Quick Start

```bash
# Install streamlit and plotly
pip install streamlit plotly

# Run the app
streamlit run frontend/app.py
```

Then open your browser to the URL shown in the terminal (usually `http://localhost:8501`)

