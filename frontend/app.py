#!/usr/bin/env python3
"""
Clutch MVP - Streamlit Dashboard

Simple MVP dashboard to display player Win Probability Added (WPA) scores.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="Clutch - Player Impact",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Headers */
    h1 {
        color: #FF6B35;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #FAFAFA;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Dividers */
    hr {
        border-color: #262730;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_player_data():
    """Load player WPA summary data."""
    data_path = project_root / "data" / "processed" / "player_wpa_summary.csv"
    if not data_path.exists():
        st.error(f"Data file not found at {data_path}")
        st.stop()
    
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def filter_pacers_players(df):
    """Filter to Pacers players only."""
    pacers_players = [
        'Haliburton', 'Siakam', 'Turner', 'Toppin', 'Nembhard', 
        'Mathurin', 'McConnell', 'Nesmith', 'Smith', 'Brown',
        'Bryant', 'Freeman', 'Champagnie', 'Sheppard', 'Walker'
    ]
    pacers_df = df[df['player_name'].isin(pacers_players)].copy()
    pacers_df = pacers_df.sort_values('player_total_WPA', ascending=False)
    return pacers_df

# Main app
st.title("🏀 Clutch")
st.markdown("### Player Impact Analytics | Indiana Pacers | 2024-25 Season")
st.caption("Win Probability Added (WPA) quantifies each player's impact on team win probability")

# Load data
df = load_player_data()
pacers_df = filter_pacers_players(df)

# Sidebar filters
st.sidebar.markdown("### ⚙️ Filters")
min_games = st.sidebar.slider(
    "Min Games",
    min_value=0,
    max_value=int(pacers_df['games_played'].max()),
    value=10,
    help="Filter by minimum games played"
)

min_events = st.sidebar.slider(
    "Min Events",
    min_value=0,
    max_value=int(pacers_df['player_event_count'].max()),
    value=100,
    help="Filter by minimum number of events"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About WPA**
- Measures impact on win probability
- Higher = more positive impact
- Weighted by game leverage
""")
st.sidebar.caption("Model: Logistic Regression (AUC: 0.52)")

# Apply filters
filtered_df = pacers_df[
    (pacers_df['games_played'] >= min_games) & 
    (pacers_df['player_event_count'] >= min_events)
].copy()

# Main content - Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Players", len(filtered_df))

with col2:
    avg_wpa = filtered_df['player_total_WPA'].mean()
    st.metric("Avg WPA", f"{avg_wpa:.2f}")

with col3:
    top_wpa = filtered_df['player_total_WPA'].max()
    top_player = filtered_df.loc[filtered_df['player_total_WPA'].idxmax(), 'player_name']
    st.metric("Top Player", top_player, f"{top_wpa:.2f}")

with col4:
    st.metric("Games", f"{filtered_df['games_played'].sum():,}")

st.markdown("---")

# Explanation Section
st.markdown("#### 📖 Understanding WPA & Leverage")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.markdown("""
    **Win Probability Added (WPA)**
    
    Total WPA measures a player's cumulative impact on their team's win probability across all plays.
    
    - **2.94 WPA** = Added ~2.94 percentage points to win probability over the season
    - **Positive** = Helped the team win more
    - **Negative** = Hurt the team's chances
    
    *Example: A shot that increases WP from 50% to 52% adds 2% WPA to that play.*
    """)

with col_exp2:
    st.markdown("""
    **Leverage**
    
    Leverage measures how important a game moment is. Higher leverage = more critical situation.
    
    - **Close games** (within 3 points) = Higher leverage
    - **Late game** (last 5 minutes) = Higher leverage
    - **Blowouts** = Lower leverage
    
    *WPA is weighted by leverage, so plays in critical moments count more.*
    """)

st.markdown("---")

# Horizontal Bar Chart: WPA Ranking
st.markdown("#### 📊 Player WPA Rankings")

# Prepare data - sort by WPA (highest first)
filtered_df_display = filtered_df.copy()
filtered_df_display = filtered_df_display.sort_values('player_total_WPA', ascending=False)

# Get min and max for proper color scaling
wpa_min = filtered_df_display['player_total_WPA'].min()
wpa_max = filtered_df_display['player_total_WPA'].max()

# Create color mapping: negative = red, zero = yellow, positive = green
# Normalize WPA values to 0-1 range for colorscale
wpa_values = filtered_df_display['player_total_WPA'].values
wpa_normalized = (wpa_values - wpa_min) / (wpa_max - wpa_min) if wpa_max != wpa_min else wpa_values

# Color scale: red (low) to yellow (middle) to green (high)
colorscale = [[0, '#FF4444'], [0.5, '#FFAA00'], [1, '#00FF88']]

# Create colors array - red for negative, green for positive
colors = []
for wpa in filtered_df_display['player_total_WPA']:
    if wpa < 0:
        # Negative: red to yellow gradient
        ratio = abs(wpa) / abs(wpa_min) if wpa_min < 0 else 0
        colors.append('#FF4444')
    elif wpa > 0:
        # Positive: yellow to green gradient
        ratio = wpa / wpa_max if wpa_max > 0 else 0
        colors.append('#00FF88')
    else:
        colors.append('#FFAA00')

# Extract values explicitly to avoid any index issues
wpa_values = [float(x) for x in filtered_df_display['player_total_WPA'].values.tolist()]
player_names = [str(x) for x in filtered_df_display['player_name'].values.tolist()]
games_played = [int(x) for x in filtered_df_display['games_played'].values.tolist()]

# Debug: Log values to verify
with st.expander("🔍 Debug Info (click to see)", expanded=False):
    st.write(f"**First 5 WPA values:** {wpa_values[:5]}")
    st.write(f"**First 5 players:** {player_names[:5]}")
    st.write(f"**WPA range:** {min(wpa_values):.2f} to {max(wpa_values):.2f}")
    st.write(f"**DataFrame shape:** {filtered_df_display.shape}")
    st.write(f"**Sample from DataFrame:**")
    st.dataframe(filtered_df_display[['player_name', 'player_total_WPA']].head())

# Create horizontal bar chart
fig = go.Figure(data=[
    go.Bar(
        x=wpa_values,  # Explicit list of floats
        y=player_names,  # Explicit list of strings
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f"{wpa:.2f}" for wpa in wpa_values],  # Format as string
        textposition='outside',
        textfont=dict(color='white', size=11),
        hovertemplate='<b>%{y}</b><br>' +
                      'WPA: %{x:.2f}<br>' +
                      'Games: %{customdata}<br>' +
                      '<extra></extra>',
        customdata=games_played
    )
])

fig.update_layout(
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=12),
    xaxis=dict(
        title=dict(text="Total WPA", font=dict(color='white')),
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color='white'),
        showgrid=True,
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.3)',
        zerolinewidth=2
    ),
    yaxis=dict(
        title="",
        tickfont=dict(color='white'),
        categoryorder='array',
        categoryarray=filtered_df_display['player_name'].tolist()  # Highest WPA first (appears at top)
    ),
    margin=dict(l=20, r=80, t=20, b=20),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
st.caption("Players ranked by Win Probability Added (WPA). Green = positive impact, Red = negative impact.")

# Data table
st.markdown("---")
st.markdown("#### 📋 Detailed Statistics")

# Format the dataframe for display
display_df = filtered_df[[
    'player_name',
    'player_total_WPA',
    'player_avg_WPA',
    'player_event_count',
    'games_played'
]].copy()

display_df.columns = [
    'Player',
    'Total WPA',
    'Avg WPA/Event',
    'Events',
    'Games'
]

# Format numbers
display_df['Total WPA'] = display_df['Total WPA'].round(3)
display_df['Avg WPA/Event'] = display_df['Avg WPA/Event'].round(6)
display_df['Events'] = display_df['Events'].apply(lambda x: f"{int(x):,}")

# Style the dataframe
styled_df = display_df.style.format({
    'Total WPA': '{:.3f}',
    'Avg WPA/Event': '{:.6f}'
})

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    height=400
)

