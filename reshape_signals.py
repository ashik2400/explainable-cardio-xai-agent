"""
Reshapes long-format PTB-XL signal CSVs (one row per timestep) into
per-record .npy files (shape (1000, 12) at 100Hz).

Expects data/raw/{split}_signal.csv for split in ['train', 'valid', 'test'],
each with columns: ecg_id, channel-0 ... channel-11.

Usage:
    python src/data/reshape_signals.py
"""

import os
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed/signals"
CHUNK_SIZE = 500_000  # 500 records * 1000 rows/record; must be a multiple of 1000

SPLITS = ["train", "valid", "test"]


def reshape_split(split: str) -> None:
    out_path = os.path.join(OUT_DIR, split)
    os.makedirs(out_path, exist_ok=True)

    csv_path = os.path.join(RAW_DIR, f"{split}_signal.csv")
    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, dtype={"ecg_id": "int32"})

    n_records = 0
    for chunk in reader:
        for ecg_id, group in chunk.groupby("ecg_id"):
            arr = group.drop(columns="ecg_id").to_numpy(dtype="float32")
            assert arr.shape == (1000, 12), f"Unexpected shape {arr.shape} for ecg_id {ecg_id}"
            import numpy as np
            np.save(os.path.join(out_path, f"{ecg_id}.npy"), arr)
            n_records += 1

    print(f"[{split}] reshaped {n_records} records -> {out_path}")


def verify_split(split: str, meta_filename: str) -> None:
    meta = pd.read_csv(os.path.join(RAW_DIR, meta_filename))
    n_files = len(os.listdir(os.path.join(OUT_DIR, split)))
    status = "OK" if n_files == len(meta) else "MISMATCH"
    print(f"[{split}] meta rows: {len(meta)}, npy files: {n_files} -> {status}")


if __name__ == "__main__":
    for split in SPLITS:
        reshape_split(split)

    for split in SPLITS:
        verify_split(split, f"{split}_meta.csv")
