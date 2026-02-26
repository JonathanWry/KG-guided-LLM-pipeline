#!/usr/bin/env python3
# Purpose: Extract positively matched query names from refined mapping files.

import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="Extract names with index > 0 from refined mapping file")
    ap.add_argument("--input_path", default=str(root / "data" / "Entity_Matching" / "existing_nodes_manually_refined.txt"))
    ap.add_argument("--output_path", default=str(root / "data" / "Entity_Matching" / "matched_nodes.txt"))
    args = ap.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            name, values = line.split(":", 1)
            values = values.strip().strip("[]")
            items = [v.strip() for v in values.split(",")]

            if len(items) < 2:
                continue

            try:
                idx = int(float(items[-1]))
            except Exception:
                continue

            if idx > 0:
                matches.append(name.strip())

    with output_path.open("w", encoding="utf-8") as out:
        for m in matches:
            out.write(m + "\n")

    print(f"Wrote {len(matches)} matches to {output_path}")


if __name__ == "__main__":
    main()
