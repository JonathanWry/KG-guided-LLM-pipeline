# Purpose: Filter COT JSONL by parse/label consistency and export Alpaca-format JSON datasets.
import json
import re
import argparse
from typing import Any, Dict, Iterable, List, Optional


# =========================
# Parsing helpers
# =========================

def extract_conclusion(text: str) -> Optional[str]:
    """
    Strict parser: ONLY look at the last non-empty line.
    Accepts Yes/No/True/False (any casing), and strips common markdown wrappers
    (**bold**, `code`, quotes), and trailing punctuation.
    Returns "Yes" / "No" / None.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Get the last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]

    # Strip common markdown/quote wrappers at both ends
    # leading wrappers (>, spaces, md/code marks, quotes, brackets, dashes)
    last = re.sub(r'^[>\s`*_~"\'()\-–—]+', '', last)
    # trailing wrappers/punctuation
    last = re.sub(r'[\s`*_~"\'()\-–—.!?]+$', '', last)

    # Find a yes/no/true/false token on this final line
    m = re.search(r'\b(yes|no|true|false)\b', last, flags=re.IGNORECASE)
    if not m:
        return None

    tok = m.group(1).lower()
    if tok in ("yes", "true"):
        return "Yes"
    if tok in ("no", "false"):
        return "No"
    return None

def normalize_label(v: Any) -> Optional[int]:
    """Convert label to int 0/1 or return None if invalid."""
    try:
        if isinstance(v, str):
            return int(v.strip())
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
    except Exception:
        pass
    return None


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Read JSONL file line by line into dicts."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# =========================
# Cleaning helpers
# =========================

# Leading markdown header patterns like "### Input:" / "### Output:"
_INPUT_HDR_RE = re.compile(r'^\s*#{0,6}\s*input\s*:?\s*\n+', flags=re.IGNORECASE)
_OUTPUT_HDR_RE = re.compile(r'^\s*#{0,6}\s*output\s*:?\s*\n+', flags=re.IGNORECASE)

def _trim_edge_blank_lines(s: str) -> str:
    """Trim leading/trailing blank lines but keep interior formatting."""
    if not isinstance(s, str):
        return ""
    # Remove leading/trailing whitespace-only lines
    s = re.sub(r'^\s*\n+', '', s)
    s = re.sub(r'\n+\s*$', '', s)
    return s

def _normalize_indent_start(s: str) -> str:
    """
    Remove any leftover leading spaces/tabs at the very start of the string.
    Does not alter indentation of subsequent lines.
    """
    if not isinstance(s, str):
        return ""
    return s.lstrip(" \t").rstrip("\r")

def clean_field(s: str, strip_input_header: bool = False, strip_output_header: bool = False) -> str:
    """Apply full cleaning pipeline for a field."""
    if not isinstance(s, str):
        return ""
    # 1) Trim leading/trailing blank lines first
    s = _trim_edge_blank_lines(s)
    # 2) Remove leading header if requested
    if strip_input_header:
        s = _INPUT_HDR_RE.sub("", s)
    if strip_output_header:
        s = _OUTPUT_HDR_RE.sub("", s)
    # 3) Trim edges again (in case header removal exposed new blank lines)
    s = _trim_edge_blank_lines(s)
    # 4) Normalize starting indentation (leading spaces/tabs at very beginning)
    s = _normalize_indent_start(s)
    return s


# =========================
# Core pipeline
# =========================

def filter_dataset_to_alpaca(
    in_jsonl: str,
    out_alpaca_json: str,
    out_bad_json: str,
    out_mismatch_json: str,
) -> Dict[str, Any]:
    """
    - Reads CoT samples from `in_jsonl`
    - Removes:
        * lines where "Yes/No" conclusion cannot be parsed
        * lines where parsed Yes/No doesn't match label
    - Writes:
        * out_alpaca_json: Alpaca format JSON array
        * out_bad_json: JSON array of lines where conclusion can't be parsed
        * out_mismatch_json: JSON array of lines where conclusion != label
    - Returns statistics.
    """
    answer_map = {"Yes": 1, "No": 0}

    total = 0
    kept = 0
    bad = 0
    mism = 0

    alpaca_rows: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []
    mismatch_rows: List[Dict[str, Any]] = []

    for rec in iter_jsonl(in_jsonl):
        total += 1

        output = rec.get("output", "")
        instr = rec.get("instruction", "")
        inp = rec.get("input", "")
        label_raw = rec.get("label", None)

        label = normalize_label(label_raw)
        concl = extract_conclusion(output)

        # Can't parse or invalid label -> bad bucket
        if concl is None or label not in (0, 1):
            bad += 1
            bad_rows.append(rec)
            continue

        pred = answer_map.get(concl)
        if pred != label:
            mism += 1
            mismatch_rows.append({
                **rec,
                "_parsed_conclusion": concl,
                "label": label,
            })
            continue

        # Clean fields
        instr_clean = clean_field(instr)  # remove leading \n/space; trim blank lines
        instr_clean = instr_clean + "\nReason and return the final answer derived from your reasoning, with **Yes** or **No** printed and specified in the end."
        inp_clean = clean_field(inp, strip_input_header=True)  # also strip leading "### Input:"
        out_clean = clean_field(output, strip_output_header=True)  # also strip leading "### Output:"

        # Build Alpaca record (no system/history/label/patient_id)
        kept += 1
        alpaca_rows.append({
            "instruction": instr_clean,
            "input": inp_clean if inp_clean.strip() else "",
            "output": out_clean if isinstance(out_clean, str) else "",
        })

    # Write outputs as JSON arrays
    with open(out_alpaca_json, "w", encoding="utf-8") as f_alp:
        json.dump(alpaca_rows, f_alp, ensure_ascii=False, indent=2)

    with open(out_bad_json, "w", encoding="utf-8") as f_bad:
        json.dump(bad_rows, f_bad, ensure_ascii=False, indent=2)

    with open(out_mismatch_json, "w", encoding="utf-8") as f_m:
        json.dump(mismatch_rows, f_m, ensure_ascii=False, indent=2)

    return {
        "total": total,
        "kept": kept,
        "removed_cant_parse_or_invalid_label": bad,
        "removed_mismatch": mism,
        "out_alpaca_json": out_alpaca_json,
        "out_bad_json": out_bad_json,
        "out_mismatch_json": out_mismatch_json,
    }


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Filter CoT dataset and export to Alpaca format (JSON arrays only)")
    ap.add_argument("--in_jsonl", required=True, help="Path to original JSONL dataset")
    ap.add_argument("--out_alpaca_json", required=True, help="Alpaca dataset (JSON array)")
    ap.add_argument("--out_bad_json", default="bad_lines.json", help="Lines with unparsed conclusion/invalid label (JSON array)")
    ap.add_argument("--out_mismatch_json", default="mismatches.json", help="Lines with mismatched labels (JSON array)")
    args = ap.parse_args()

    stats = filter_dataset_to_alpaca(
        in_jsonl=args.in_jsonl,
        out_alpaca_json=args.out_alpaca_json,
        out_bad_json=args.out_bad_json,
        out_mismatch_json=args.out_mismatch_json,
    )

    print("\n=== Done ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


# python filter_and_format_alpaca.py \
#   --in_jsonl your_path_to_repo/cot_dataset/cot_dataset_with_paths.jsonl \
#   --out_alpaca_json your_path_to_repo/cot_dataset/cot_dataset_1000_patient_with_paths.json \
#   --out_bad_json your_path_to_repo/cot_dataset/cot_dataset_1000_patient_with_paths_parse_error.json \
#   --out_mismatch_json your_path_to_repo/cot_dataset/cot_dataset_1000_patient_with_paths_mismatch.json
