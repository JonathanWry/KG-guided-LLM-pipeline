# Purpose: Compute multilabel metrics from per-sample generation JSONL outputs.
import argparse, json, sys, numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score

def safe_auc(y, s):
    try: return roc_auc_score(y, s)
    except ValueError: return np.nan

def safe_aupr(y, s):
    try: return average_precision_score(y, s)
    except ValueError: return np.nan

def safe_bacc(y, yhat):
    try: return balanced_accuracy_score(y, yhat)
    except ValueError: return np.nan

def weighted_nanmean(values, weights):
    """Compute weighted mean ignoring NaNs in values (and their weights)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & (weights > 0)
    if not np.any(mask):
        return float('nan')
    return float(np.average(values[mask], weights=weights[mask]))

def load_allowed_diseases(filter_map_path, mode):
    """
    Returns a set of allowed disease names based on the filter map and mode.

    mode:
      - 'drop_neg_only': drop only diseases with index < 0 if they appear in map.
                         Others (not in map) are kept.
      - 'keep_nonneg_only': keep only diseases that appear in map with index >= 0.
                            All others are dropped.
    """
    if mode in ("none",):
        return None
        
    if not filter_map_path:
        return None  # no filtering

    with open(filter_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    in_map_nonneg = {d for d, arr in m.items()
                     if isinstance(arr, list) and len(arr) >= 2 and isinstance(arr[1], (int, float)) and arr[1] >= 0}
    in_map_neg = {d for d, arr in m.items()
                  if isinstance(arr, list) and len(arr) >= 2 and isinstance(arr[1], (int, float)) and arr[1] < 0}

    if mode == "drop_neg_only":
        # Allow everything except the explicit negatives
        return {"__MODE__": mode, "nonneg": in_map_nonneg, "neg": in_map_neg}
    elif mode == "keep_nonneg_only":
        # Allow only explicit non-neg listed diseases
        return {"__MODE__": mode, "nonneg": in_map_nonneg, "neg": in_map_neg}
    else:
        raise ValueError(f"Unknown filter_mode: {mode}")

def apply_filter_to_disease_list(all_diseases, filter_info):
    """
    Reduce the disease set according to filter_info returned by load_allowed_diseases.
    """
    if filter_info is None:
        return sorted(all_diseases), set()  # no filtering, no dropped

    mode = filter_info["__MODE__"]
    nonneg, neg = filter_info["nonneg"], filter_info["neg"]

    if mode == "drop_neg_only":
        # Remove those explicitly marked as index < 0; keep everything else
        kept = sorted([d for d in all_diseases if d not in neg])
        dropped = set(all_diseases) - set(kept)
        return kept, dropped

    if mode == "keep_nonneg_only":
        # Keep only those explicitly marked as index >= 0
        kept = sorted([d for d in all_diseases if d in nonneg])
        dropped = set(all_diseases) - set(kept)
        return kept, dropped

    # Fallback (shouldn't hit)
    return sorted(all_diseases), set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default="per_sample_generations.jsonl",
                    help="Path to per-sample JSONL.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for positive prediction.")
    ap.add_argument("--filter_map", type=str, default=None,
                    help="Path to disease filter JSON (disease -> [name, index]).")
    ap.add_argument(
        "--filter_mode",
        type=str,
        default="drop_neg_only",
        choices=["drop_neg_only", "keep_nonneg_only", "none"],
        help=(
            "drop_neg_only: drop diseases with index < 0 if in map; keep others. "
            "keep_nonneg_only: keep only diseases with index >= 0 if in map; drop others. "
            "none: ignore the filter map entirely."
        ),
    )
    args = ap.parse_args()

    # --- Load JSONL ---
    rows = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print("No rows found in JSONL. Exiting.")
        return

    # --- Patients & diseases found in the JSONL ---
    patients = sorted({r["patient_id"] for r in rows})
    diseases_all = sorted({r["disease"] for r in rows})

    # --- Optional filtering by disease mapping ---
    filter_info = load_allowed_diseases(args.filter_map, args.filter_mode)
    diseases, dropped = apply_filter_to_disease_list(diseases_all, filter_info)

    print(f"#patients: {len(patients)}")
    print(f"#diseases in JSONL: {len(diseases_all)}")
    if filter_info is None:
        print("Filtering: OFF")
    else:
        print(f"Filtering: ON ({args.filter_mode})")
        print(f"  kept diseases: {len(diseases)}")
        print(f"  dropped diseases: {len(dropped)}")

    # --- Build matrices (only for kept diseases) ---
    pid2i = {p: i for i, p in enumerate(patients)}
    dis2j = {d: j for j, d in enumerate(diseases)}

    Y_true  = np.full((len(patients), len(diseases)), np.nan, dtype=float)
    Y_score = np.full_like(Y_true, np.nan)

    skipped_rows = 0
    for r in rows:
        d = r["disease"]
        if d not in dis2j:   # filtered out
            skipped_rows += 1
            continue
        i, j = pid2i[r["patient_id"]], dis2j[d]
        Y_true[i, j]  = float(r["gold"])
        Y_score[i, j] = float(r["score_p_yes"])

    if skipped_rows > 0:
        print(f"Rows skipped due to filtering: {skipped_rows}")

    if Y_true.size == 0 or np.all(np.isnan(Y_true)):
        print("No data left after filtering; exiting.")
        return

    # --- Threshold & mask ---
    thr = args.threshold
    mask = ~np.isnan(Y_true) & ~np.isnan(Y_score)
    y_true = Y_true[mask].astype(int)
    y_pred = (Y_score[mask] > thr).astype(int)

    # --- Micro metrics ---
    micro_acc = (y_true == y_pred).mean()
    # For micro F1 over flattened decisions, this is equivalent to micro-F1 on a binary vector
    micro_f1  = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # --- Per-label metrics + supports ---
    per_f1, per_auc, per_aupr = [], [], []
    per_acc, per_bacc, per_support = [], [], []

    for j in range(Y_true.shape[1]):
        col_mask = ~np.isnan(Y_true[:, j]) & ~np.isnan(Y_score[:, j])
        n_j = int(np.sum(col_mask))
        if n_j == 0:
            per_f1.append(np.nan)
            per_auc.append(np.nan)
            per_aupr.append(np.nan)
            per_acc.append(np.nan)
            per_bacc.append(np.nan)
            per_support.append(0)
            continue

        yt = Y_true[col_mask, j].astype(int)
        ys = Y_score[col_mask, j].astype(float)
        yp = (ys > thr).astype(int)

        per_f1.append(f1_score(yt, yp, average='binary', zero_division=0))
        per_auc.append(safe_auc(yt, ys))
        per_aupr.append(safe_aupr(yt, ys))
        per_acc.append((yt == yp).mean())
        per_bacc.append(safe_bacc(yt, yp))
        per_support.append(n_j)

    # --- Weighted macro metrics (by per-label support) ---
    macro_f1   = float(np.nanmean(per_f1))
    macro_auc  = float(np.nanmean(per_auc))
    macro_aupr = float(np.nanmean(per_aupr))
    macro_acc  = float(np.nanmean(per_acc))
    macro_bacc = float(np.nanmean(per_bacc))

    print(f"micro_acc   : {micro_acc:.6f}")
    print(f"micro_f1    : {micro_f1:.6f}")
    print(f"macro_acc_w : {macro_acc:.6f}")
    print(f"macro_bacc_w: {macro_bacc:.6f}")
    print(f"macro_f1_w  : {macro_f1:.6f}")
    print(f"macro_auc_w : {macro_auc:.6f}")
    print(f"macro_aupr_w: {macro_aupr:.6f}")

if __name__ == "__main__":
    main()


# python compute_metrics_from_jsonl.py --jsonl your_path_to_repo/results/evaluation/<run_name>/per_sample_generations.jsonl --filter_map your_path_to_repo/data/mimic/disease-filter_updated.json
