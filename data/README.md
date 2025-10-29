# Clutch Data Ingestion

This module provides NBA data ingestion capabilities for the Clutch sports analytics application.

## Features

- **Player Game Logs**: Fetch comprehensive game statistics for individual players
- **Team Game Logs**: Fetch team-level game data and win/loss records
- **Rate Limiting**: Built-in API rate limiting to respect NBA API limits
- **Error Handling**: Robust retry logic with exponential backoff
- **Data Validation**: Quality checks and verification of fetched data
- **Logging**: Comprehensive logging of all operations

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Fetch player data:
```bash
python scripts/fetch_data.py --player "LeBron James" --season "2023-24"
```

Fetch team data:
```bash
python scripts/fetch_data.py --team "Lakers" --season "2023-24"
```

With data verification:
```bash
python scripts/fetch_data.py --player "Stephen Curry" --season "2022-23" --verify
```

### Programmatic Usage

```python
from backend.services.data_ingestion import NBADataIngestion

# Initialize the service
ingestion = NBADataIngestion()

# Fetch player data
result = ingestion.fetch_and_save_player_data("LeBron James", "2023-24")
print(f"Success: {result['success']}")
print(f"Rows: {result['rows']}")
print(f"File: {result['filepath']}")

# Fetch team data
result = ingestion.fetch_and_save_team_data("Lakers", "2023-24")
```

## Data Structure

### Output Files

Data is saved to `data/raw/` with the following naming convention:
- Player data: `player_{name}_{season}.csv`
- Team data: `team_{name}_{season}.csv`

### Data Columns

Player game logs include:
- Game statistics (points, rebounds, assists, etc.)
- Game metadata (date, opponent, home/away)
- Player metadata (name, ID, season)
- Ingestion metadata (fetched_at timestamp)

Team game logs include:
- Team statistics and game outcomes
- Game metadata (date, opponent, home/away)
- Team metadata (name, ID, season)
- Ingestion metadata (fetched_at timestamp)

## Configuration

### Rate Limiting

The default rate limit is 0.6 seconds between API calls. This can be adjusted:

```python
ingestion = NBADataIngestion(rate_limit_delay=1.0)  # 1 second delay
```

### Data Directory

Specify a custom data directory:

```python
ingestion = NBADataIngestion(data_dir="custom_data")
```

## Error Handling

The system includes comprehensive error handling:

- **Player/Team Not Found**: Clear error messages for invalid names
- **API Failures**: Automatic retry with exponential backoff
- **Network Issues**: Graceful handling of connectivity problems
- **Data Validation**: Verification of data quality and structure

## Logging

All operations are logged to:
- Console output (INFO level)
- Log files in `data/logs/` (daily rotation)
- Metadata file `data/logs/fetch_metadata.txt`

## Data Quality Verification

Use the `--verify` flag or programmatic verification:

```python
quality_report = ingestion.verify_data_quality("path/to/file.csv")
print(f"Readable: {quality_report['is_readable']}")
print(f"Rows: {quality_report['rows']}")
print(f"Columns: {quality_report['columns']}")
```

## Examples

### Fetch Multiple Players

```bash
# Fetch data for multiple players
python scripts/fetch_data.py --player "LeBron James" --season "2023-24"
python scripts/fetch_data.py --player "Stephen Curry" --season "2023-24"
python scripts/fetch_data.py --player "Kevin Durant" --season "2023-24"
```

### Fetch Team Data

```bash
# Fetch data for multiple teams
python scripts/fetch_data.py --team "Lakers" --season "2023-24"
python scripts/fetch_data.py --team "Warriors" --season "2023-24"
python scripts/fetch_data.py --team "Nets" --season "2023-24"
```

## Troubleshooting

### Common Issues

1. **Player/Team Not Found**: Ensure names match exactly (case-insensitive)
2. **API Rate Limits**: Increase `rate_limit_delay` if getting rate limit errors
3. **Network Issues**: Check internet connection and NBA API status
4. **Permission Errors**: Ensure write permissions for data directories

### Debug Mode

Enable verbose logging:

```bash
python scripts/fetch_data.py --player "LeBron James" --season "2023-24" --verbose
```

## API Limits

The NBA API has rate limits. The default configuration respects these limits:
- 0.6 second delay between requests
- Automatic retry with exponential backoff
- Maximum 3 retry attempts per request

For high-volume usage, consider implementing additional caching or batch processing.
