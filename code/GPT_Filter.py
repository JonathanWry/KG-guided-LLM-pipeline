#!/usr/bin/env python3
# Purpose: Post-filter top-k entity matches with an LLM and resume from checkpoint.

import argparse
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


DEFAULT_SYSTEM_PROMPT = (
    "You're an expert in the medical domain. You receive one query and candidate entity matches. "
    "Return the best exact/closest match as '<query>: [<match_name>, <match_index>]'. "
    "If no valid match exists, return '<query>: [NONE, -1]'."
)


def load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def save_checkpoint(path: Path, line_number: int) -> None:
    path.write_text(str(line_number), encoding="utf-8")


def extract_between_markers(text: str) -> Optional[str]:
    t = text.strip()
    lower = t.lower()
    start = lower.find("<start>")
    end = lower.find("<end>")
    if start != -1 and end != -1 and end > start:
        return t[start + len("<start>") : end].strip()
    return t if t else None


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="LLM-based filter for entity_match output")
    ap.add_argument("--input_file", default=str(root / "data" / "Entity_Matching" / "existing_nodes_topN.txt"))
    ap.add_argument("--output_file", default=str(root / "data" / "Entity_Matching" / "existing_nodes_gpt_match.txt"))
    ap.add_argument("--checkpoint_file", default=str(root / "data" / "Entity_Matching" / "checkpoint.txt"))
    ap.add_argument("--model", default="gpt-4.1-nano")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running GPT_Filter.py")

    client = OpenAI(api_key=api_key)

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    checkpoint_file = Path(args.checkpoint_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    start_line = load_checkpoint(checkpoint_file)
    lines = input_file.read_text(encoding="utf-8").splitlines()

    mode = "a" if output_file.exists() and start_line > 0 else "w"
    with output_file.open(mode, encoding="utf-8") as out:
        for i, line in tqdm(enumerate(lines[start_line:], start=start_line), desc="Filtering lines", unit="line"):
            user_prompt = line.strip()
            if not user_prompt:
                out.write("NONE\n")
                save_checkpoint(checkpoint_file, i + 1)
                continue

            try:
                response = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": args.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = response.output_text.strip()
                extracted = extract_between_markers(text) or "NONE"
            except Exception as e:
                extracted = f"NONE  # error: {e}"

            out.write(extracted + "\n")
            out.flush()
            save_checkpoint(checkpoint_file, i + 1)

    print(f"Completed filtering. Output: {output_file}")


if __name__ == "__main__":
    main()
