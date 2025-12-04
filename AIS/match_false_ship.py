import json
import csv
import zipfile
from pathlib import Path
import pandas as pd

# Inputs
FALSE_NEG_JSON = Path('..') / 'AIS' / 'results' / 'false_negative_predictions.json'
ZIP_DIR = Path('..') / 'AIS' / 'zip'
ZIP_MAP = {
    '2025-06-29': ('aisdk-2025-06-29.zip', 'aisdk-2025-06-29.csv'),
    '2025-07-02': ('aisdk-2025-07-02.zip', 'aisdk-2025-07-02.csv'),
}

# Spatial bounds
MIN_LAT = 55.33407820857486
MAX_LAT = 55.349157335271144
MIN_LON = 10.976395248488702
MAX_LON = 11.09285951270881

TIME_TOLERANCE_SEC = 120

OUTPUT_DIR = Path('..') / 'AIS' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHES_CSV = OUTPUT_DIR / 'ais_false_neg_matches.csv'
UNMATCHED_CSV = OUTPUT_DIR / 'ais_false_neg_unmatched.csv'
SUMMARY_TXT = OUTPUT_DIR / 'ais_false_neg_summary.txt'

RELEVANT_COLUMNS = [
    '# Timestamp','Type of mobile','MMSI','Latitude','Longitude','Navigational status','SOG','COG','Heading','Name','Ship type','Width','Length','__sec_of_day','__time_of_day'
]

# Note: Use only '# Timestamp' to derive time-of-day seconds
def timestamp_to_seconds(ts: str) -> int:
    # Expect format like '29/06/2025 00:00:04' or similar
    try:
        parts = ts.split()
        tod = parts[-1]  # 'HH:MM:SS'
        h, m, s = tod.split(':')
        return int(h)*3600 + int(m)*60 + int(s)
    except Exception:
        # Fallback: try to extract HH:MM:SS anywhere in the string
        import re
        m = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", ts)
        if m:
            h = int(m.group(1)); mn = int(m.group(2)); sec = int(m.group(3))
            return h*3600 + mn*60 + sec
        raise

# Load AIS rows from a given date's zip (single CSV inside)
# This now stores the full AIS row (all columns) in each returned dict.
def load_ais_rows_for_date(date_key: str):
    zip_name, csv_name = ZIP_MAP[date_key]
    zip_path = ZIP_DIR / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"AIS zip not found: {zip_path}")
    rows = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(csv_name) as f:
            reader = csv.reader(line.decode('utf-8', errors='ignore') for line in f.readlines())
            header = next(reader)
            # Build column index map
            col_idx = {col: i for i, col in enumerate(header)}
            # Ensure needed columns exist
            required = ['Latitude','Longitude','# Timestamp','MMSI']
            for r in required:
                if r not in col_idx:
                    raise ValueError(f"Missing column '{r}' in AIS CSV {csv_name}")
            for r in reader:
                try:
                    lat = float(r[col_idx['Latitude']]); lon = float(r[col_idx['Longitude']])
                except ValueError:
                    continue
                if lat < MIN_LAT or lat > MAX_LAT or lon < MIN_LON or lon > MAX_LON:
                    continue
                ts_val = r[col_idx['# Timestamp']]
                try:
                    tod_seconds = timestamp_to_seconds(ts_val)
                except Exception:
                    continue
                # Store full AIS row (all columns). Also store parsed time fields.
                row_dict = {header[i]: r[i] for i in range(len(header))}
                row_dict['__time_of_day_seconds'] = str(int(tod_seconds))
                row_dict['__time_of_day'] = ts_val
                rows.append(row_dict)
    return rows

def main():
    if not FALSE_NEG_JSON.exists():
        raise FileNotFoundError(f"False negative predictions JSON not found: {FALSE_NEG_JSON}")
    false_negs = json.loads(FALSE_NEG_JSON.read_text())

    # Group false negatives by ais_date
    by_date = {}
    for rec in false_negs:
        date_key = rec['ais_date']
        if date_key not in by_date:
            by_date[date_key] = []
        by_date[date_key].append(rec)

    # Load AIS rows per date once
    ais_cache = {}
    for date_key in by_date.keys():
        ais_cache[date_key] = load_ais_rows_for_date(date_key)

    match_rows = []
    unmatched_rows = []

    for date_key, preds in by_date.items():
        ais_rows = ais_cache[date_key]
        # For efficiency, build time bucket map
        time_map = {}
        for r in ais_rows:
            sec = int(r['__time_of_day_seconds'])
            time_map.setdefault(sec, []).append(r)

        for p in preds:
            target_sec = p['seconds_of_day']
            # Collect candidates within tolerance
            candidates = []
            for delta in range(-TIME_TOLERANCE_SEC, TIME_TOLERANCE_SEC + 1):
                sec = target_sec + delta
                if sec < 0 or sec > 86400:
                    continue
                if sec in time_map:
                    candidates.extend(time_map[sec])
            if candidates:
                # Find closest time delta
                candidates.sort(key=lambda r: abs(int(r['__time_of_day_seconds']) - target_sec))
                closest = candidates[0]
                closest_delta = int(closest['__time_of_day_seconds']) - target_sec
                mmsi_list = list({c.get('MMSI','') for c in candidates})
                names = list({c.get('Name','') for c in candidates if c.get('Name','')})
                base = {
                    'hhmmss': p['hhmmss'],
                    'prediction_filename': p['filename'],
                    'ais_date': date_key,
                    'prediction_seconds': target_sec,
                    'ship_probability': p['ship_probability'],
                    'matched': True,
                    'num_matches': len(candidates),
                    'mmsi_list': ';'.join(mmsi_list),
                    'names': ';'.join(names),
                    'closest_time_of_day': closest.get('__time_of_day'),
                    'closest_time_delta_sec': closest_delta,
                }
                # Add all AIS columns from the closest AIS record, prefixed with 'ais_'
                for k, v in closest.items():
                    base[f'ais_{k}'] = v
                match_rows.append(base)
            else:
                # No candidates within tolerance.
                # Attach the closest AIS row from the day's AIS (if any) so we still save all ais columns.
                if ais_rows:
                    closest = min(ais_rows, key=lambda r: abs(int(r['__time_of_day_seconds']) - target_sec))
                    closest_delta = int(closest['__time_of_day_seconds']) - target_sec
                    # For unmatched, we report num_matches = 0 and matched = False, but still include AIS fields
                    mmsi_list = list({c.get('MMSI','') for c in ais_rows})
                    names = list({c.get('Name','') for c in ais_rows if c.get('Name','')})
                    base = {
                        'hhmmss': p['hhmmss'],
                        'prediction_filename': p['filename'],
                        'ais_date': date_key,
                        'prediction_seconds': target_sec,
                        'ship_probability': p['ship_probability'],
                        'matched': False,
                        'num_matches': 0,
                        'mmsi_list': ';'.join(mmsi_list),
                        'names': ';'.join(names),
                        'closest_time_of_day': closest.get('__time_of_day'),
                        'closest_time_delta_sec': closest_delta,
                    }
                    for k, v in closest.items():
                        base[f'ais_{k}'] = v
                    unmatched_rows.append(base)
                else:
                    # No AIS rows at all for that date: keep minimal unmatched record
                    unmatched_rows.append({
                        'hhmmss': p['hhmmss'],
                        'prediction_filename': p['filename'],
                        'ais_date': date_key,
                        'prediction_seconds': target_sec,
                        'ship_probability': p['ship_probability'],
                        'matched': False
                    })

    matches_df = pd.DataFrame(match_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)

    matches_df.to_csv(MATCHES_CSV, index=False)
    unmatched_df.to_csv(UNMATCHED_CSV, index=False)

    total_preds = len(false_negs)
    total_matches = len(matches_df)
    match_rate = (total_matches / total_preds) if total_preds else 0.0

    summary_lines = [
        f"Total false negative predictions: {total_preds}",
        f"Total matched (AIS present) false negatives: {total_matches}",
        f"Total unmatched (no AIS within tolerance) false negatives: {len(unmatched_df)}",
        f"Match rate: {match_rate:.2%}",
        "", "Per-date breakdown:" ]
    for date_key in by_date.keys():
        date_preds = len(by_date[date_key])
        date_matches = sum(1 for r in match_rows if r['ais_date'] == date_key)
        summary_lines.append(f"  {date_key}: preds={date_preds}, matches={date_matches}, rate={(date_matches/date_preds if date_preds else 0):.2%}")

    SUMMARY_TXT.write_text('\n'.join(summary_lines))
    print(f"Saved matches:   {MATCHES_CSV}")
    print(f"Saved unmatched: {UNMATCHED_CSV}")
    print(f"Saved summary:   {SUMMARY_TXT}")

if __name__ == '__main__':
    main()