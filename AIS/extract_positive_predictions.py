import pandas as pd
import os
from pathlib import Path
import json

# Configuration
PREDICTIONS_CSV = Path('..') / 'New_CAE' / 'results' / 'champion_test_predictions.csv'
OUTPUT_DIR = Path('..') / 'AIS' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POSITIVES_JSON = OUTPUT_DIR / 'positive_predictions.json'
POSITIVES_CSV = OUTPUT_DIR / 'positive_predictions.csv'
TIME_PREFIX_MAPPING = {
    '00': '2025-06-29',  # Day A (June 29)
    '12': '2025-07-02',  # Day B (July 02)
    '13': '2025-07-02',  # Also July 02
}

# Helper: extract HHMMSS from filename like 122236_T30s.png
def extract_hhmmss(filename: str):
    base = os.path.basename(filename)
    hhmmss = base.split('_')[0]  # '122236'
    if len(hhmmss) != 6 or not hhmmss.isdigit():
        return None
    return hhmmss

# Convert HHMMSS -> seconds of day
def hhmmss_to_seconds(hhmmss: str) -> int:
    h = int(hhmmss[0:2]); m = int(hhmmss[2:4]); s = int(hhmmss[4:6])
    return h*3600 + m*60 + s

def main():
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_CSV}")
    df = pd.read_csv(PREDICTIONS_CSV)
    if not {'filename','ship_probability','prediction'}.issubset(df.columns):
        raise ValueError("champion_test_predictions.csv missing required columns")
    positives = df[df['prediction'] == 1].copy()
    positives['hhmmss'] = positives['filename'].apply(extract_hhmmss)
    positives = positives[positives['hhmmss'].notnull()]
    positives['seconds_of_day'] = positives['hhmmss'].apply(hhmmss_to_seconds)
    positives['prefix'] = positives['hhmmss'].str[0:2]
    positives['ais_date'] = positives['prefix'].map(TIME_PREFIX_MAPPING)

    # Build structured list
    records = []
    for _, row in positives.iterrows():
        records.append({
            'hhmmss': row['hhmmss'],
            'seconds_of_day': int(row['seconds_of_day']),
            'ship_probability': float(row['ship_probability']),
            'filename': row['filename'],
            'ais_date': row['ais_date'],
            'prefix': row['prefix']
        })

    # Save JSON and CSV
    with open(POSITIVES_JSON, 'w') as f:
        json.dump(records, f, indent=2)
    positives[['hhmmss','seconds_of_day','ship_probability','filename','ais_date','prefix']].to_csv(POSITIVES_CSV, index=False)

    print(f"Extracted {len(records)} positive predictions.")
    print(f"Saved JSON: {POSITIVES_JSON}")
    print(f"Saved CSV:  {POSITIVES_CSV}")

if __name__ == '__main__':
    main()