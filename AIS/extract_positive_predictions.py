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
FALSE_NEG_JSON = OUTPUT_DIR / 'false_negative_predictions.json'
FALSE_NEG_CSV = OUTPUT_DIR / 'false_negative_predictions.csv'
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
    required_cols = {'filename','ship_probability','prediction','true_label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"champion_test_predictions.csv missing required columns: {required_cols - set(df.columns)}")
    positives = df[df['prediction'] == 1].copy()
    positives['hhmmss'] = positives['filename'].apply(extract_hhmmss)
    positives = positives[positives['hhmmss'].notnull()]
    positives['seconds_of_day'] = positives['hhmmss'].apply(hhmmss_to_seconds)
    positives['prefix'] = positives['hhmmss'].str[0:2]
    positives['ais_date'] = positives['prefix'].map(TIME_PREFIX_MAPPING)

    # False negatives: true ship but predicted 0
    false_negs = df[(df['true_label'] == 1) & (df['prediction'] == 0)].copy()
    false_negs['hhmmss'] = false_negs['filename'].apply(extract_hhmmss)
    false_negs = false_negs[false_negs['hhmmss'].notnull()]
    false_negs['seconds_of_day'] = false_negs['hhmmss'].apply(hhmmss_to_seconds)
    false_negs['prefix'] = false_negs['hhmmss'].str[0:2]
    false_negs['ais_date'] = false_negs['prefix'].map(TIME_PREFIX_MAPPING)

    # Build structured lists
    pos_records = []
    for _, row in positives.iterrows():
        pos_records.append({
            'hhmmss': row['hhmmss'],
            'seconds_of_day': int(row['seconds_of_day']),
            'ship_probability': float(row['ship_probability']),
            'filename': row['filename'],
            'ais_date': row['ais_date'],
            'prefix': row['prefix'],
            'true_label': int(row['true_label']),
            'prediction': int(row['prediction'])
        })
    fn_records = []
    for _, row in false_negs.iterrows():
        fn_records.append({
            'hhmmss': row['hhmmss'],
            'seconds_of_day': int(row['seconds_of_day']),
            'ship_probability': float(row['ship_probability']),
            'filename': row['filename'],
            'ais_date': row['ais_date'],
            'prefix': row['prefix'],
            'true_label': int(row['true_label']),
            'prediction': int(row['prediction'])
        })

    # Save JSON and CSV
    with open(POSITIVES_JSON, 'w') as f:
        json.dump(pos_records, f, indent=2)
    positives[['hhmmss','seconds_of_day','ship_probability','filename','ais_date','prefix','true_label','prediction']].to_csv(POSITIVES_CSV, index=False)

    with open(FALSE_NEG_JSON, 'w') as f:
        json.dump(fn_records, f, indent=2)
    false_negs[['hhmmss','seconds_of_day','ship_probability','filename','ais_date','prefix','true_label','prediction']].to_csv(FALSE_NEG_CSV, index=False)

    print(f"Extracted {len(pos_records)} positive predictions (pred==1).")
    print(f"Extracted {len(fn_records)} false negatives (true_label==1 & pred==0).")
    print(f"Saved positives JSON: {POSITIVES_JSON}")
    print(f"Saved positives CSV:  {POSITIVES_CSV}")
    print(f"Saved false negatives JSON: {FALSE_NEG_JSON}")
    print(f"Saved false negatives CSV:  {FALSE_NEG_CSV}")

if __name__ == '__main__':
    main()