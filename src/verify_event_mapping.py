"""Verify event IDs against the original DDIMDL event database.

Run after download_data.py and run_experiment.py. Download the original event.db
from the DDIMDL repository to data/original_ddimdl_event.db before running.
"""
from pathlib import Path
import hashlib
import sqlite3
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "data"
db = data / "original_ddimdl_event.db"
expected = "a25ba8ba3f8bddd07143ae6a6973bf343514bd14898849c0d97335b3f2d45c15"
if not db.exists():
    raise SystemExit("Download DDIMDL event.db into data/original_ddimdl_event.db first.")
if hashlib.sha256(db.read_bytes()).hexdigest() != expected:
    raise SystemExit("Unexpected event.db SHA-256; check its version before auditing.")
with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as con:
    events = pd.read_sql_query('SELECT "index", id1, id2 FROM event', con)
    terms = pd.read_sql_query('SELECT "index", mechanism, action FROM extraction', con)
source = events.merge(terms, on="index", validate="one_to_one")
def key(a, b):
    return "|".join(sorted((str(a), str(b))))
source["pair"] = [key(a, b) for a, b in zip(source.id1, source.id2)]
benchmark = pd.concat([
    pd.read_csv(data / "Deng's_dataset" / "0" / f"ddi_{part}1xiao.csv")
    for part in ("training", "validation", "test")
], ignore_index=True)
benchmark["pair"] = [key(a, b) for a, b in zip(benchmark.d1, benchmark.d2)]
assert len(source) == len(benchmark) == 37264
assert not source.pair.duplicated().any() and not benchmark.pair.duplicated().any()
matched = benchmark.merge(source[["pair", "mechanism", "action"]], on="pair", validate="one_to_one")
assert len(matched) == 37264
unique = matched[["type", "mechanism", "action"]].drop_duplicates()
assert unique.type.nunique() == 65 and len(unique) == 65
full = matched.groupby("type").size().rename("full_n")
sample = pd.read_csv(root / "results_full" / "analysis_data.csv").groupby("type").size().rename("sample_n")
result = unique.set_index("type").join(full).join(sample).loc[range(8)].reset_index()
result[["full_n", "sample_n"]] = result[["full_n", "sample_n"]].astype(int)
result.to_csv(root / "results_full" / "eight_class_mapping_verified.csv", index=False)
print("Verified 37,264 unique pairs and one mechanism/action per event ID.")
print(result.to_string(index=False))
