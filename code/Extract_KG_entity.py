#!/usr/bin/env python3
# Purpose: Extract unique KG entities and node indices from prune_kg.csv for downstream entity matching.

import argparse
from pathlib import Path
import pandas as pd


def default_paths():
    root = Path(__file__).resolve().parents[1]
    return (
        root / "data" / "prune_kg.csv",
        root / "data" / "Entity_Matching" / "prune_filtered_entities.txt",
    )


def main() -> None:
    default_kg, default_out = default_paths()

    ap = argparse.ArgumentParser(description="Extract unique KG entities and indices from prune_kg.csv")
    ap.add_argument("--kg_csv", default=str(default_kg), help="Path to prune_kg.csv")
    ap.add_argument("--out_txt", default=str(default_out), help="Output text file: <entity>: <index>")
    args = ap.parse_args()

    kg_csv = Path(args.kg_csv)
    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(kg_csv, usecols=["x_name", "x_index", "y_name", "y_index"], low_memory=False)

    x_pairs = df[["x_name", "x_index"]].dropna().drop_duplicates()
    y_pairs = df[["y_name", "y_index"]].dropna().drop_duplicates()
    y_pairs = y_pairs.rename(columns={"y_name": "x_name", "y_index": "x_index"})

    merged = pd.concat([x_pairs, y_pairs], ignore_index=True)

    entities = {}
    for _, row in merged.iterrows():
        name = str(row["x_name"]).strip()
        try:
            idx = int(float(row["x_index"]))
        except Exception:
            continue
        if name:
            entities[name] = idx

    with out_txt.open("w", encoding="utf-8") as f:
        for name, idx in entities.items():
            f.write(f"{name}: {idx}\n")

    print(f"Wrote {len(entities)} entities to {out_txt}")


if __name__ == "__main__":
    main()
