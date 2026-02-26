# Purpose: Detect and score conclusion-vs-label conflicts in generated reasoning outputs.
import json
import re
from typing import Iterable, Dict, Any, Tuple, List, Optional
from sklearn.metrics import accuracy_score, roc_auc_score
import math
import os

# =========================
# Strict last-line parser
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


def _to_int_label(label) -> Optional[int]:
    try:
        if isinstance(label, str):
            return int(label.strip())
        if isinstance(label, bool):
            return int(label)
        if isinstance(label, (int, float)):
            return int(label)
    except Exception:
        pass
    return None


def count_mismatches(records: Iterable[Dict[str, Any]]):
    answer_map = {"Yes": 1, "No": 0}
    total = 0
    parsed_ok = 0
    missing = 0
    mismatches = []

    # per-label stats
    label_stats = {
        0: {"total": 0, "mismatch": 0},
        1: {"total": 0, "mismatch": 0},
    }
    failed_lines: List[Tuple[int, str]] = []  # (index, output_text)

    for idx, rec in enumerate(records):
        total += 1
        output = rec.get("output", "")
        label = _to_int_label(rec.get("label", None))

        concl = extract_conclusion(output)
        if concl is None or label not in (0, 1):
            missing += 1
            failed_lines.append((idx, str(output)))   # store failed line
            continue

        parsed_ok += 1
        pred = 1 if concl == "Yes" else 0

        # update label totals
        label_stats[label]["total"] += 1

        if pred != label:
            mismatches.append({
                "index": idx,
                "patient_id": rec.get("patient_id"),
                "disease": rec.get("disease"),
                "conclusion": concl,
                "label": label
            })
            label_stats[label]["mismatch"] += 1

    return total, parsed_ok, missing, mismatches, label_stats, failed_lines


# =========================
# Per-disease metrics (accuracy & AUC) + macro AUC
# =========================
def compute_per_disease_metrics(records: Iterable[Dict[str, Any]]):
    """
    Build per-disease accuracy/AUC using strict last-line parsing.
    prob is set to float(pred) (0/1) because we only have hard labels.
    AUC is computed only when both classes are present and all probs exist.
    Returns (rows, macro_auc) where rows is a list of dicts:
      {"disease", "n", "n_valid", "accuracy", "auc"}
    """
    by_disease: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        disease = rec.get("disease", "<unknown>")
        gold = _to_int_label(rec.get("label", None))
        concl = extract_conclusion(rec.get("output", ""))
        pred = None if concl is None else (1 if concl == "Yes" else 0)
        prob = None if pred is None else float(pred)
        by_disease.setdefault(disease, []).append(
            {"gold": gold, "pred": pred, "prob": prob}
        )

    rows = []
    aucs = []
    for disease, items in sorted(by_disease.items()):
        # keep rows where we parsed a prediction AND have a valid gold label
        valid = [x for x in items if x["pred"] is not None and x["gold"] in (0, 1)]
        if not valid:
            acc = float("nan")
            auc = float("nan")
        else:
            y_true = [x["gold"] for x in valid]
            y_pred = [x["pred"] for x in valid]
            acc = accuracy_score(y_true, y_pred)

            # AUC only if both classes present and we have a prob for each
            probs = [x["prob"] for x in valid]
            if len(set(y_true)) > 1 and len(probs) == len(y_true):
                try:
                    auc = roc_auc_score(y_true, probs)
                except Exception:
                    auc = float("nan")
            else:
                auc = float("nan")

        rows.append({
            "disease": disease,
            "n": len(items),
            "n_valid": len(valid),
            "accuracy": acc,
            "auc": auc
        })
        if not math.isnan(auc):
            aucs.append(auc)

    macro_auc = float(sum(aucs) / len(aucs)) if aucs else float("nan")
    return rows, macro_auc


def save_per_disease_csv(rows: List[Dict[str, Any]], out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("disease,n,n_valid,accuracy,auc\n")
        for r in rows:
            acc = "" if math.isnan(r["accuracy"]) else f"{r['accuracy']:.6f}"
            auc = "" if math.isnan(r["auc"]) else f"{r['auc']:.6f}"
            f.write(f"{r['disease']},{r['n']},{r['n_valid']},{acc},{auc}\n")


# ---------- Main entry ----------
def examine_conflict(path,
                     failed_out="your_path_to_output/failed_lines.jsonl",
                     per_disease_csv=None):
    # 1) Read JSONL
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # 2) Count mismatches / parse failures
    total, parsed_ok, missing, mismatches, label_stats, failed_lines = count_mismatches(data)

    print(f"Total samples: {total}")
    print(f"Parsed conclusions: {parsed_ok}")
    print(f"Missing/invalid conclusion or label: {missing}")
    print(f"Overall mismatches: {len(mismatches)}")

    for lbl in (0, 1):
        total_lbl = label_stats[lbl]["total"]
        mismatch_lbl = label_stats[lbl]["mismatch"]
        if total_lbl > 0:
            ratio = mismatch_lbl / total_lbl * 100
            print(f"Label={lbl}: total={total_lbl}, mismatches={mismatch_lbl}, mismatch ratio={ratio:.2f}%")
        else:
            print(f"Label={lbl}: total=0")

    # show a few mismatches
    for m in mismatches[:5]:
        print(m)

    # 3) Save parse-failure lines
    if failed_lines:
        print(f"\n⚠️ Failed to parse lines: {len(failed_lines)} (showing up to 5)")
        for idx, text in failed_lines[:5]:
            print(f"- Index {idx}: {text[:200]}...")
        with open(failed_out, "w", encoding="utf-8") as fout:
            for idx, text in failed_lines:
                fout.write(json.dumps({"index": idx, "output": text}, ensure_ascii=False) + "\n")
        print(f"\nAll failed lines saved to: {failed_out}")

    # 4) Compute per-disease metrics + macro AUC
    rows, macro_auc = compute_per_disease_metrics(data)
    print("\n=== Eval (per disease) ===")
    for r in rows:
        acc_str = "nan" if math.isnan(r["accuracy"]) else f"{r['accuracy']:.3f}"
        auc_str = "nan" if math.isnan(r["auc"]) else f"{r['auc']:.6f}"
        print(f"{r['disease']}: n={r['n']}, valid={r['n_valid']}, acc={acc_str}, auc={auc_str}")

    print(f"\nMacro AUC (valid diseases only): {'nan' if math.isnan(macro_auc) else f'{macro_auc:.6f}'}")

    # Optional: save CSV
    if per_disease_csv:
        save_per_disease_csv(rows, per_disease_csv)
        print(f"\nSaved per-disease metrics CSV to: {per_disease_csv}")


if __name__ == "__main__":
    path = "your_path_to_output/cot_dataset_with_paths.jsonl"
    # Optional: output path for per-disease metrics CSV
    examine_conflict(path, per_disease_csv="your_path_to_output/per_disease_metrics.csv")
