# AIS Pipeline Overview

This folder contains a lightweight pipeline that cross-references model ship detections (from convolutional image classifier results) with AIS (Automatic Identification System) positional reports to assess whether predicted ship presence aligns with real vessel traffic.

## Objectives
1. Extract time-based positive ship predictions from the ML model test results.
2. Map each positive prediction timestamp (HHMMSS) to the appropriate AIS date source based on its leading hour prefix.
3. Load and spatially filter AIS records from compressed daily archives.
4. Perform time-window matching (±30 seconds) between predicted ship timestamps and AIS vessel reports within a defined geographic bounding box.
5. Produce match and non-match reports for downstream analysis and validation.

## Data Inputs
- `../New_CAE/results/champion_test_predictions.csv` — Output of model evaluation on held-out test set.
  - Columns used: `filename`, `ship_probability`, `prediction`.
  - Ship positives are rows where `prediction == 1`.
  - Filenames encode start time as `HHMMSS_T30s.png` (first 6 digits = HHMMSS).
- AIS ZIP archives in `AIS/zip/`:
  - `aisdk-2025-06-29.zip` → contains `aisdk-2025-06-29.csv` (Day A data).
  - `aisdk-2025-07-02.zip` → contains `aisdk-2025-07-02.csv` (Day B data).
- AIS CSV columns (subset used): `# Timestamp`, `Latitude`, `Longitude`, `MMSI`, `Name`, `SOG`, `COG`, plus metadata.

## Date Mapping Logic
| Prediction Prefix | Mapped AIS Date |
|-------------------|-----------------|
| `00`              | 2025-06-29      |
| `12`, `13`        | 2025-07-02      |

The first two digits of the HHMMSS portion of the filename are treated as an hour prefix that determines which AIS day is searched.

## Spatial Filter
Only AIS rows within the bounding box are considered:
- Latitude: 55.33407820857486 ≤ lat ≤ 55.349157335271144
- Longitude: 10.976395248488702 ≤ lon ≤ 11.09285951270881

This reduces noise and focuses matching on the monitored region.

## Temporal Matching
- For each positive prediction timestamp (converted to seconds-of-day), the pipeline searches AIS rows with `# Timestamp` parsed into HH:MM:SS.
- Tolerance: ±30 seconds.
- All candidate rows within the tolerance are collected; the **closest** row (smallest absolute time delta) is flagged for summary attributes.
- Multiple vessel hits are aggregated via distinct MMSI and Name lists.

## Scripts
### 1. `extract_positive_predictions.py`
Parses model test predictions and emits structured positives.
- Output JSON: `AIS/results/positive_predictions.json`
- Output CSV:  `AIS/results/positive_predictions.csv`
- Columns: `hhmmss`, `seconds_of_day`, `ship_probability`, `filename`, `ais_date`, `prefix`

### 2. `match_ais_ships.py`
Loads positives and matches against AIS archives.
- Outputs:
  - `AIS/results/ais_matches.csv`
  - `AIS/results/ais_unmatched.csv`
  - `AIS/results/ais_match_summary.txt`
- Match CSV columns:
  - `hhmmss`, `prediction_filename`, `ais_date`, `prediction_seconds`, `ship_probability`
  - `matched`, `num_matches`, `mmsi_list`, `names`
  - `closest_time_of_day`, `closest_time_delta_sec`, `closest_lat`, `closest_lon`, `closest_sog`, `closest_cog`
- Unmatched CSV columns:
  - `hhmmss`, `prediction_filename`, `ais_date`, `prediction_seconds`, `ship_probability`, `matched=False`

## Usage
From repository root:
```bash
python AIS/extract_positive_predictions.py
python AIS/match_ais_ships.py
```
If using Windows CMD:
```cmd
python AIS\extract_positive_predictions.py
python AIS\match_ais_ships.py
```

## Assumptions & Notes
- Filenames strictly follow `HHMMSS_T30s.png`. Non-conforming names are ignored.
- `# Timestamp` column in AIS CSV is authoritative; custom columns `__sec_of_day` / `__time_of_day` (if present) are ignored for matching.
- Time zone assumed consistent between imagery timestamps and AIS feeds (UTC or local alignment not adjusted here).
- ZIP archives contain a single CSV each; script streams line-by-line decoding UTF-8 with fallback ignore of malformed bytes.
- Probability scores (`ship_probability`) are retained for ranking confidence in matches.
- Bounding box chosen from prior domain analysis; adjust in `match_ais_ships.py` if region changes.

## Extending
- Make tolerance configurable: add `argparse` flag `--tolerance` in `match_ais_ships.py`.
- Introduce radius-based distance filtering: compute haversine distance from a centroid instead of a bounding box.
- Integrate MMSI metadata enrichment (flag vessel type or size categories).
- Add a combined report merging matched & unmatched rows with a status column.

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Empty matches | Spatial bounds too tight or date mapping off | Verify prefixes/date mapping and bounding box values |
| High unmatched count | Time tolerance too strict | Increase ± seconds (e.g. 45–60s) |
| Unicode decode errors | Non-UTF8 bytes in AIS feed | Current code uses `errors='ignore'`; switch to robust decoding if needed |
| Missing Columns Error | AIS CSV schema mismatch | Inspect headers, update `required` list in script |

## Summary
This AIS folder enables quick validation of model ship detections against authoritative vessel movement logs, helping distinguish true positives from potential false alarms and informing threshold tuning or data augmentation strategies.
