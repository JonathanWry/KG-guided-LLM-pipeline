#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Purpose: Generate COT datasets from graph/CSV inputs with checkpointing and optional KG context.

"""
construct_COT_dataset_new.py

End-to-end COT dataset generator with two modes:

1) CSV mode (--csv_mode):
   - Reads a patients CSV with columns: patient_id(optional), features_present, and 0/1 disease columns.
   - Samples patients FIRST (deterministic), then creates (patient, disease) pairs.
   - Uses relevance.txt (JSON-ish) + filtered_path_mappings.json for context.

2) GRAPH mode (--graph_mode):
   - Reads:
        * edge-labels file (rows=patients, columns=disease edges) in braced format: "{ 1,0,1,... }"
        * edge_text.json mapping edge index -> disease name
        * disease_filter.json mapping disease name -> [alias_or_null, numeric_id]; keep only numeric_id >= 0
        * hyperedges.txt in braced format: "{ 1,1026,5,... }" listing present feature node IDs per patient
        * node_text.json mapping node id -> feature text
   - Samples patients FIRST (deterministic), then builds worklist of allowed (patient, disease-edge) pairs.
   - Uses relevance.txt (JSON-ish) + filtered_path_mappings.json for context.

Common:
   - Streams outputs to JSONL with checkpointing + resumable progress.
   - Calls Azure OpenAI Chat Completions via environment variables (no keys hard-coded).
   - Writes final records containing prompt context and model outputs.

Usage (GRAPH mode example):
---------------------------
python construct_COT_dataset_new.py \
  --graph_mode \
  --hyperedges_txt your_path_to_repo/data/mimic/hyperedges-mimic3_truncated.txt \
  --node_text_json your_path_to_repo/data/mimic/node_text.json \
  --edge_labels_txt your_path_to_repo/data/mimic/edge-labels-mimic3_updated_truncated.txt \
  --edge_text_json your_path_to_repo/data/mimic/edge_text.json \
  --disease_filter_json your_path_to_repo/data/mimic/disease-filter_updated.json \
  --relevance_txt your_path_to_repo/data/relevence.txt \
  --paths_json your_path_to_repo/data/filtered_path_mappings.json \
  --output_jsonl your_path_to_repo/cot_dataset/cot_dataset_with_paths.jsonl \
  --progress_file your_path_to_repo/results/progress/cot_dataset_with_paths.progress \
  --checkpoint_every 10 \
  --sample_patients 100 \
  --sample_seed 42 \
  --model gpt-4o

Notes:
- Ensure AZURE_OPENAI_API_KEY is set in your environment.
- Please revoke any keys that were previously committed to logs/scripts.
"""

import os
import re
import json
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple, Set, Iterable
from sklearn.metrics import accuracy_score, roc_auc_score
import math

import pandas as pd
from tqdm import tqdm

# External util available in your environment
from examine_conflict import examine_conflict

# =========================
# Azure OpenAI client
# =========================
from openai import AzureOpenAI

AZURE_OPENAI_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    "https://your_azure_openai_endpoint",
)
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

_client = None


def get_azure_client():
    global _client
    if _client is None:
        if not AZURE_OPENAI_API_KEY:
            raise RuntimeError("Set AZURE_OPENAI_API_KEY (or OPENAI_API_KEY) for Azure OpenAI access.")
        if AZURE_OPENAI_ENDPOINT == "https://your_azure_openai_endpoint":
            raise RuntimeError("Set AZURE_OPENAI_ENDPOINT for Azure OpenAI access.")
        _client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
    return _client

# =========================
# System prompt used for the COT run
# =========================
SYSTEM_PROMPT = """
Given a disease to be verified, a list of diagnostic_features_present measured at the index visit (time t), a subset of {potentially_relevant_exist_features} of confirmed positive features, a list of {potentially_relevant_absent_features} of confirmed negatives, a partial set of preliminary reasoning paths, and a provided judgement label referring to the NEXT visit (time t+1), your task is to reason step by step as if you are independently determining whether the disease will be present at the next visit with **Yes** or **No** without prior knowledge of the given answer.

IMPORTANT TEMPORAL SETTING:
- Interpret all provided features as observations at time t (the previous/last visit).
- The target label is the presence/absence of the disease at time t+1 (the next visit).
- Prefer prognostic/persistence evidence (chronicity, structural disease, repeatedly documented diagnoses, objective abnormalities likely to persist) over transient/acute findings.
- Do NOT assume that a disease present at time t persists to time t+1 unless there is explicit evidence of chronicity, recurrence risk, or ongoing pathophysiology.
- *Explicitly check whether each key feature or diagnosis at time t is acute/transient or chronic/persistent. Down-weight acute findings such as AMI, stroke, GI bleed, or sepsis unless explicit recurrence or chronic sequelae are documented.*
- **Medication caveat.** Do **not** infer disease presence from treatment context alone (e.g., vasopressors, inpatient insulin) unless paired with a simple persistence anchor (e.g., diagnosis code, outpatient regimen, or longitudinal labs).
- **Acute default.** For acute/episodic diseases (AMI, shock, GI bleed, sepsis, stroke), predict **No** for t+1 unless there is explicit “ongoing/recurrent/unresolved” evidence **(e.g., recent transfusion plus source-localizing/therapeutic endoscopy or angiography)**.
- **Context ≠ diagnosis.** Do not treat comorbidities or typical co-medications as proof of the target disease (e.g., **CHF or nitrates ≠ CAD**) without direct evidence.

Instructions:
1. Explore the question and first filter out irrelevant partial reasoning paths provided for disease verification, leaving only useful paths based on your expertise.
2. Incorporate the remaining paths and potentially_relevant_features naturally as if you discovered them yourself, then supplement with your new reasoning path from provided features to form complete reasoning paths based on your expertise.  Treat potentially_relevant_present as confirmed present (1) and potentially_relevant_absent as confirmed absent (0) at time t.
3. Analyze and evaluate the reasoning paths with explicit attention to TEMPORAL VALIDITY, expanding on the most relevant ones together with explicit diagnostic evidence to construct a logical, well-supported explanation. Diagnostic evidence at time t only supports t+1 if it is chronic/persistent. Prefer objective findings (imaging, labs, pathognomonic signs) and recorded diagnoses with known persistence over context-only signals. Distinguish between **prognostic risk** (may increase chance of disease) and **confirmed presence** (objective diagnosis or persistent abnormality). Use risk features only as minor contextual modifiers, never as decisive proof.
4. Base your decision ONLY on confirmed, definitive evidence (1/0). Avoid using assumptions, indirect risk factors, or therapy context as critical evidence. Do NOT equate risk factor with a confirmed diagnosis.
5. If there are resolution cues (e.g., “unspecified hemorrhage” without source, single-episode events, acute procedures that typically resolve), down-weight them for predicting t+1. Always check for **resolution indicators** (e.g., ""treated,"" ""status post,"" ""resolved,"" ""acute episode"") and treat such findings as low-weight for t+1 prediction.
6. Do not mention the existence of predefined reasoning paths or the provided answer in your response.
7. Do not assume the given answer is correct. Instead, determine the answer solely based on your reasoning.
8. If your final conclusion contradicts the given answer, acknowledge potential discrepancies without mentioning provided answer (e.g., ""Wait, there might be something wrong"") and refine your response again accordingly.

**Tiebreaker:** If evidence is balanced or ambiguous about persistence, **No** should be preferred.

### Output:
Finding reasoning paths:
(you ""discover"" potential reasoning paths yourself by using the given paths if useful or generating your own if not.  It should be concise as a list of knowledge paths indexed properly.
Reasoning Process:
(Step-by-step reasoning process, do not assume the given answer is correct and do not mention the existence of answer.}
Conclusion:
(The final answer derived from your reasoning, with **Yes** or **No** printed and specified in the end.)
""".strip()



SYSTEM_PROMPT_WITHOUT_KG = """
Given a disease to be verified, a list of diagnostic_features_present measured at the index visit (time t), a subset of {potentially_relevant_exist_features} of confirmed positive features, a list of {potentially_relevant_absent_features} of confirmed negatives, and a provided judgement label referring to the NEXT visit (time t+1), your task is to reason step by step as if you are independently determining whether the disease will be present at the next visit with **Yes** or **No** without prior knowledge of the given answer.

IMPORTANT TEMPORAL SETTING:
- Interpret all provided features as observations at time t (the previous/last visit).
- The target label is the presence/absence of the disease at time t+1 (the next visit).
- Prefer prognostic/persistence evidence (chronicity, structural disease, repeatedly documented diagnoses, objective abnormalities likely to persist) over transient/acute findings.
- Do NOT assume that a disease present at time t persists to time t+1 unless there is explicit evidence of chronicity, recurrence risk, or ongoing pathophysiology.
- *Explicitly check whether each key feature or diagnosis at time t is acute/transient or chronic/persistent. Down-weight acute findings such as AMI, stroke, GI bleed, or sepsis unless explicit recurrence or chronic sequelae are documented.*
- **Medication caveat.** Do **not** infer disease presence from treatment context alone (e.g., vasopressors, inpatient insulin) unless paired with a simple persistence anchor (e.g., diagnosis code, outpatient regimen, or longitudinal labs).
- **Acute default.** For acute/episodic diseases (AMI, shock, GI bleed, sepsis, stroke), predict **No** for t+1 unless there is explicit “ongoing/recurrent/unresolved” evidence **(e.g., recent transfusion plus source-localizing/therapeutic endoscopy or angiography)**.
- **Context ≠ diagnosis.** Do not treat comorbidities or typical co-medications as proof of the target disease (e.g., **CHF or nitrates ≠ CAD**) without direct evidence.

Instructions:
1. Explore the question based on your expertise.
2. Incorporate potentially_relevant_features naturally as if you discovered them yourself, then supplement with your new reasoning path from provided features to form complete reasoning paths based on your expertise.  Treat potentially_relevant_present as confirmed present (1) and potentially_relevant_absent as confirmed absent (0) at time t.
3. Analyze and evaluate your reasoning paths with explicit attention to TEMPORAL VALIDITY, expanding on the most relevant ones together with explicit diagnostic evidence to construct a logical, well-supported explanation. Diagnostic evidence at time t only supports t+1 if it is chronic/persistent. Prefer objective findings (imaging, labs, pathognomonic signs) and recorded diagnoses with known persistence over context-only signals. Distinguish between **prognostic risk** (may increase chance of disease) and **confirmed presence** (objective diagnosis or persistent abnormality). Use risk features only as minor contextual modifiers, never as decisive proof.
4. Base your decision ONLY on confirmed, definitive evidence (1/0). Avoid using assumptions, indirect risk factors, or therapy context as critical evidence. Do NOT equate risk factor with a confirmed diagnosis.
5. If there are resolution cues (e.g., “unspecified hemorrhage” without source, single-episode events, acute procedures that typically resolve), down-weight them for predicting t+1. Always check for **resolution indicators** (e.g., ""treated,"" ""status post,"" ""resolved,"" ""acute episode"") and treat such findings as low-weight for t+1 prediction.
6. Do not mention the existence of the provided answer in your response.
7. Do not assume the given answer is correct. Instead, determine the answer solely based on your reasoning.
8. If your final conclusion contradicts the given answer, acknowledge potential discrepancies without mentioning provided answer (e.g., ""Wait, there might be something wrong"") and refine your response again accordingly.

**Tiebreaker:** If evidence is balanced or ambiguous about persistence, **No** should be preferred.

### Output:
Finding reasoning paths:
(you ""discover"" potential reasoning paths.
Reasoning Process:
(Step-by-step reasoning process, do not assume the given answer is correct and do not mention the existence of answer.}
Conclusion:
(The final answer derived from your reasoning, with **Yes** or **No** printed and specified in the end.)
""".strip()

# =========================
# Small utilities (I/O, progress, normalization)
# =========================
def _ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def append_jsonl(file_path: str, records: list):
    if not records:
        return
    _ensure_dir(file_path)
    with open(file_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()

def read_progress(progress_path: str) -> int:
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return 0

def write_progress(progress_path: str, next_idx: int):
    _ensure_dir(progress_path)
    tmp = progress_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(next_idx))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, progress_path)

def norm_key(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\b,?\s*genetic\b", "", s)   # drop trailing ", genetic"
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# Relevance / Paths
# =========================
def load_relevance(path: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text.lstrip().startswith("{"):
        text = "{\n" + text + "\n}"
    text = re.sub(r"\}\s*\n\s*\"", "},\n\"", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse {path} as JSON-ish text. Error: {e}\n\nPreview:\n{text[:500]}...")
    norm_index = {norm_key(k): v for k, v in data.items()}
    return data, norm_index

def load_path_mapping(path_json: str) -> Dict[str, List[str]]:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {norm_key(k): v for k, v in data.items()}

def get_paths_for_disease(paths_norm: Dict[str, List[str]], disease: str) -> List[str]:
    return paths_norm.get(norm_key(disease), [])

# =========================
# CSV helpers
# =========================
def disease_columns(df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in exclude_cols]

def features_from_row(row: pd.Series, feats_col: str) -> List[str]:
    raw = row.get(feats_col, "")
    if pd.isna(raw) or raw is None:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]

# =========================
# Braced-format loaders for GRAPH mode
# =========================
def _parse_braced_ints_line(line: str) -> List[int]:
    return [int(x) for x in re.findall(r"[0-9]+", line)]

def load_edge_labels_braced(path: str) -> List[List[int]]:
    rows: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                rows.append([])
            else:
                rows.append(_parse_braced_ints_line(line))
    return rows

def load_hyperedges_braced(path: str) -> List[List[int]]:
    return load_edge_labels_braced(path)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# Disease filter (GRAPH mode)
# =========================
def load_disease_filter(path: str) -> Dict[str, List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_allowed_edge_indices(edge_text: Dict[str, Any], disease_filter: Dict[str, List[Any]]) -> Set[int]:
    allowed: Set[int] = set()
    for k, name in edge_text.items():
        try:
            eidx = int(k)
        except Exception:
            continue
        entry = disease_filter.get(name)
        if not entry or len(entry) < 2:
            continue
        num_id = entry[1]
        try:
            if isinstance(num_id, (int, float)) and int(num_id) >= 0:
                allowed.add(eidx)
        except Exception:
            pass
    return allowed

# =========================
# Relevance helpers
# =========================
def get_relevant_entities(relevance_norm: Dict[str, dict], disease: str) -> List[str]:
    block = relevance_norm.get(norm_key(disease), {})
    names = []
    for _, v in sorted(block.items(), key=lambda kv: kv[0]):
        n = v.get("name")
        if n:
            names.append(n)
    return names

def partition_potential_relevance(patient_features: List[str], relevant_entities: List[str]) -> Tuple[List[str], List[str]]:
    present, absent = [], []
    for ent in relevant_entities:
        ent_l = ent.lower()
        found = any((ent_l == f.lower()) or (ent_l in f.lower()) for f in patient_features)
        (present if found else absent).append(ent)
    return present, absent

def extract_conclusion(text: str):
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

    
    
def _append_eval_record(eval_records: List[dict], disease_name: str, gold: int, model_text: str):
    """
    Parse the model output and add an evaluation record.
    - Yes/No on last line -> pred/prob = 1/0
    - True/False on last line -> pred/prob = 1/0
    - If neither is found, record pred=None (excluded from metrics)
    """
    pred = None
    prob = None

    yn = parse_final_yes_no(model_text)
    if yn in ("Yes", "No"):
        pred = 1 if yn == "Yes" else 0
        prob = float(pred)
    else:
        tf = parse_final_true_false(model_text)
        if tf in (0, 1):
            pred = int(tf)
            prob = float(pred)

    eval_records.append({
        "disease": disease_name,
        "gold": int(gold),
        "pred": pred,          # may be None if unparsable
        "prob": prob,          # may be None if unparsable
    })


def _summarize_and_save_eval(eval_records: List[dict], out_dir: str, tag: str):
    """
    Compute per-disease accuracy/AUC (AUC only if both classes present and probs exist),
    then print and save:
      - <tag>_by_disease.csv
      - <tag>_summary.txt
    Macro AUC is the mean of valid per-disease AUCs.
    """
    # Group by disease
    by_dis = {}
    for r in eval_records:
        by_dis.setdefault(r["disease"], []).append(r)

    rows = []
    aucs = []
    for d, items in sorted(by_dis.items()):
        # keep only records with a parsed prediction
        valid = [r for r in items if r["pred"] is not None]
        if not valid:
            acc = float("nan")
            auc = float("nan")
        else:
            y_true = [r["gold"] for r in valid]
            y_pred = [r["pred"] for r in valid]
            acc = accuracy_score(y_true, y_pred)

            # probs for AUC
            probs = [r["prob"] for r in valid if r["prob"] is not None]
            if len(set(y_true)) > 1 and len(probs) == len(y_true):
                try:
                    auc = roc_auc_score(y_true, probs)
                except Exception:
                    auc = float("nan")
            else:
                auc = float("nan")

        rows.append({"disease": d, "n": len(items), "n_valid": len(valid), "accuracy": acc, "auc": auc})
        if not math.isnan(auc):
            aucs.append(auc)

    # Macro AUC over diseases with a valid AUC
    macro_auc = float(sum(aucs) / len(aucs)) if aucs else float("nan")

    # Print a compact summary
    print("\n=== Eval (per disease) ===")
    for r in rows:
        print(f"{r['disease']}: n={r['n']}, valid={r['n_valid']}, acc={r['accuracy']:.3f}, auc={r['auc'] if not math.isnan(r['auc']) else 'nan'}")
    print(f"\nMacro AUC (valid diseases only): {macro_auc if not math.isnan(macro_auc) else 'nan'}")

    # Save CSV + summary
    os.makedirs(out_dir or ".", exist_ok=True)
    by_dis_csv = os.path.join(out_dir, f"{tag}_by_disease.csv")
    with open(by_dis_csv, "w", encoding="utf-8") as f:
        f.write("disease,n,n_valid,accuracy,auc\n")
        for r in rows:
            acc = "" if math.isnan(r["accuracy"]) else f"{r['accuracy']:.6f}"
            auc = "" if math.isnan(r["auc"]) else f"{r['auc']:.6f}"
            f.write(f"{r['disease']},{r['n']},{r['n_valid']},{acc},{auc}\n")

    summary_txt = os.path.join(out_dir, f"{tag}_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Macro AUC: {'' if math.isnan(macro_auc) else f'{macro_auc:.6f}'}\n")
        f.write("Per-disease rows in CSV: " + os.path.basename(by_dis_csv) + "\n")

    print(f"\nSaved per-disease metrics to: {by_dis_csv}")
    print(f"Saved summary to:             {summary_txt}")

# =========================
# Prompt building for the GENERATION (unchanged)
# =========================
def build_setting3_user_input(
    disease: str,
    present_features: List[str],
    pot_rel_present: List[str],
    pot_rel_absent: List[str],
    answer_yes_no: str,
    paths: Optional[List[str]] = None,
    include_kg: bool = True,
) -> str:
    def j(xs: List[str]) -> str:
        return "; ".join(xs) if xs else "(none)"
    base_sections = [
        "### Input:\n",
        f"Disease to be verified: {disease}\n\n",
        f"diagnostic_features_present: {j(present_features)}\n\n",
        f"potentially_relevant_present_features: {j(pot_rel_present)}\n\n",
        f"potentially_relevant_absent_features: {j(pot_rel_absent)}\n\n",
    ]
    if include_kg:
        paths_block = "\n".join(f"- {p}" for p in (paths or [])) if paths else "(none)"
        base_sections.append(f"Reasoning Paths:\n{paths_block}\n\n")
    base_sections.append(f"Answer: {answer_yes_no}\n")
    return "".join(base_sections)

def _last_nonempty_line(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s:
            return s
    return ""

def _clean_token(s: str) -> str:
    # strip very common wrappers/punct (bold/italics/code/quotes/dashes/periods)
    return s.strip().strip("*`_~'\"()[]{}<>-–—.!?:;").lower()

def parse_final_yes_no(text: str):
    """
    Return 'Yes' or 'No' based ONLY on the last non-empty line.
    Accepts minimal wrappers like **Yes**, `no`, Yes., etc.
    """
    last = _last_nonempty_line(text)
    tok = _clean_token(last)
    if tok == "yes":
        return "Yes"
    if tok == "no":
        return "No"
    return None

def parse_final_true_false(text: str):
    """
    Return 1 for True, 0 for False based ONLY on the last non-empty line.
    Accepts minimal wrappers like **True**, `false`, False., etc.
    """
    last = _last_nonempty_line(text)
    tok = _clean_token(last)
    if tok == "true":
        return 1
    if tok == "false":
        return 0
    return None

def call_openai_responses(
    model: str,
    system_prompt: str,
    user_input: str,
    max_output_tokens: int = 2048,
) -> str:
    try:
        client = get_azure_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_input},
            ],
            max_tokens=max_output_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[OpenAI Chat Error] {e}", flush=True)
        return ""

# =========================
# NEW: dataset-facing formatting that you asked for
# =========================
def format_instruction(disease: str) -> str:
    """
    Format the instruction for GPT based on the given disease.
    """
    instruction = f"""
    Given a disease to be verified, a list of diagnostic_features_present measured at the index visit (time t), a subset of potentially_relevant_exist_features of confirmed positive features, a list of potentially_relevant_absent_features of confirmed negative, your task is to reason step by step as if you are independently determining whether the disease will be present at the next visit with **Yes** or **No**.
    """
    return instruction.strip()

def _join(xs: List[str]) -> str:
    return ", ".join(xs) if xs else "(none)"

def format_user_query(
    patient_record: List[str],
    disease_name: str,
    pot_rel_present: List[str],
    pot_rel_absent: List[str],
    paths: Optional[List[str]],
    include_kg: bool = True,
) -> str:
    """
    Format the user's query to verify if a patient has a particular disease.
    Consistent with COT generation: includes the same potential relevant features
    (present/absent) that were supplied to the generator.
    """
    paths_block = "\n".join(f"- {p}" for p in (paths or [])) if paths else "(none)"
    sections = [
        "### Input:\n",
        f"Disease to be verified: {disease_name}\n\n",
        f"diagnostic_features_present:  {_join(patient_record)}\n\n",
        f"potentially_relevant_present_features: {_join(pot_rel_present)}\n\n",
        f"potentially_relevant_absent_features: {_join(pot_rel_absent)}\n\n",
    ]
    if include_kg:
        paths_block = "\n".join(f"- {p}" for p in (paths or [])) if paths else "(none)"
        sections.append(f"Reasoning Paths:\n{paths_block}\n\n")
    return "".join(sections)

# =========================
# Sampling & worklists
# =========================
def _pid_sort_key(pid: Any):
    if isinstance(pid, str) and pid.isdigit():
        return (0, int(pid))
    return (1, str(pid))

def sample_ids(ids: Iterable[Any], k: int, seed: int) -> List[Any]:
    ids_sorted = sorted(list(ids), key=_pid_sort_key)
    rng = random.Random(seed)
    return rng.sample(ids_sorted, min(k, len(ids_sorted)))

def collect_all_patient_ids_from_graph(hyperedges: List[List[int]], edge_labels: List[List[int]]) -> List[int]:
    n_pat = min(len(hyperedges), len(edge_labels))
    return list(range(n_pat))

def build_worklist_csv(df: pd.DataFrame, diseases: List[str], allow_patient_ids: Optional[Set[Any]] = None) -> List[Tuple[Any, str]]:
    wl: List[Tuple[Any, str]] = []
    for _, row in df.iterrows():
        pid = row["patient_id"]
        if allow_patient_ids is not None and pid not in allow_patient_ids:
            continue
        for d in diseases:
            wl.append((pid, d))
    return wl

def build_worklist_graph(
    *,
    sampled_patients: Set[int],
    allowed_edge_indices: Set[int],
    edge_labels: List[List[int]],
    edge_text: Dict[str, Any],
) -> List[Tuple[int, int, str]]:
    worklist: List[Tuple[int, int, str]] = []
    for pid in sorted(sampled_patients):
        if pid < 0 or pid >= len(edge_labels):
            continue
        row = edge_labels[pid]
        row_len = len(row)
        if row_len == 0:
            continue
        for eidx in sorted(allowed_edge_indices):
            if eidx >= row_len:
                continue
            name = edge_text.get(str(eidx)) or edge_text.get(eidx)
            if not name:
                continue
            worklist.append((pid, eidx, name))
    worklist.sort(key=lambda t: (t[0], norm_key(t[2])))
    return worklist

# =========================
# NEW: compute end index for first N patients from a start index
# =========================
def end_index_for_first_n_patients_in_worklist(
    worklist: List[Tuple[Any, ...]],
    start_idx: int,
    n_patients: int,
    pid_pos: int = 0
) -> int:
    """
    Returns the exclusive end index so that the slice [start_idx:end_idx)
    contains pairs for exactly the next 'n_patients' unique patient IDs,
    or until the worklist ends.
    """
    if n_patients is None or n_patients <= 0:
        return len(worklist)
    seen: Set[Any] = set()
    end = start_idx
    while end < len(worklist):
        pid = worklist[end][pid_pos]
        if pid not in seen:
            if len(seen) == n_patients:
                break
            seen.add(pid)
        end += 1
    return end

# =========================
# Main processing (CSV mode)
# =========================
def process_setting3_csv(
    *,
    patients_csv: str,
    relevance_txt: str,
    paths_json: str,
    features_col: str,
    exclude_cols: List[str],
    model: str,
    output_jsonl: str,
    progress_file: str,
    checkpoint_every: int = 20,
    sample_patients: Optional[int] = None,
    sample_seed: int = 42,
    view_cot: bool = False,
    early_stop: Optional[int] = None,            # legacy (pairs)
    early_stop_patients: Optional[int] = None,   # NEW (patients)
    eval_records: List[dict] = [],
    no_kg_context: bool = False,
):
    df = pd.read_csv(patients_csv)
    if "patient_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "patient_id"})

    _, relevance_norm = load_relevance(relevance_txt)
    paths_norm = load_path_mapping(paths_json)

    diseases = disease_columns(df, exclude_cols)
    assert diseases, f"No disease columns found; check exclude_cols={exclude_cols}"

    # Sample patients FIRST (deterministic)
    allow_ids: Optional[Set[Any]] = None
    if sample_patients is not None and sample_patients > 0:
        sampled = sample_ids(df["patient_id"].tolist(), k=sample_patients, seed=sample_seed)
        allow_ids = set(sampled)

    worklist = build_worklist_csv(df, diseases, allow_patient_ids=allow_ids)
    total = len(worklist)

    start_idx = read_progress(progress_file)
    start_idx = max(0, min(start_idx, total))

    if early_stop_patients is not None:
        end_idx = end_index_for_first_n_patients_in_worklist(worklist, start_idx, early_stop_patients, pid_pos=0)
    else:
        end_idx = max(start_idx, min(early_stop, total)) if early_stop is not None else total

    pbar = tqdm(total=total, initial=start_idx, desc="Setting-3 (CSV)", ncols=100)
    buffer: List[dict] = []
    next_idx = start_idx

    try:
        for idx in range(start_idx, end_idx):
            pid, disease_name = worklist[idx]
            row = df[df["patient_id"] == pid].iloc[0]
            pid_int = int(pid)

            feats = features_from_row(row, features_col)
            label = int(row[disease_name])  # 0/1
            answer = "Yes" if label == 1 else "No"

            ents = get_relevant_entities(relevance_norm, disease_name)
            pot_present, pot_absent = partition_potential_relevance(feats, ents)
            paths = get_paths_for_disease(paths_norm, disease_name)

            # ==== prompts for generation ====
            gen_user_input = build_setting3_user_input(
                disease=disease_name,
                present_features=feats,
                pot_rel_present=pot_present,
                pot_rel_absent=pot_absent,
                answer_yes_no=answer,
                paths=None if no_kg_context else paths,
                include_kg=not no_kg_context,
            )
            cot = call_openai_responses(
                model=model,
                system_prompt=SYSTEM_PROMPT_WITHOUT_KG if no_kg_context else SYSTEM_PROMPT,
                user_input=gen_user_input,
                max_output_tokens=2048,
            )
            _append_eval_record(eval_records, disease_name=disease_name, gold=label, model_text=cot)

            # ==== dataset-facing fields per your spec ====
            instruction = format_instruction(disease_name)
            user_input_for_row = format_user_query(
                patient_record=feats,
                disease_name=disease_name,
                pot_rel_present=pot_present,
                pot_rel_absent=pot_absent,
                paths=None if no_kg_context else paths,
                include_kg=not no_kg_context,
            )

            row_obj = {
                "instruction": instruction,
                "input": user_input_for_row,
                "output": cot,
                "label": label,
                "disease": disease_name,
                "patient_id": pid_int,
            }
            buffer.append(row_obj)

            next_idx = idx + 1
            pbar.update(1)

            if view_cot:
                print(f"\n--- disease={disease_name} | patient={pid_int} ---\n{gen_user_input}\n\n{cot}\n{'='*60}")

            if len(buffer) >= checkpoint_every:
                append_jsonl(output_jsonl, buffer)
                write_progress(progress_file, next_idx)
                buffer.clear()
                pbar.set_postfix_str(f"flushed@{next_idx}")

        if buffer:
            append_jsonl(output_jsonl, buffer)
            write_progress(progress_file, next_idx)
            buffer.clear()

    finally:
        pbar.close()
        examine_conflict(path=output_jsonl)

# =========================
# Main processing (GRAPH mode)
# =========================
def process_setting3_graph(
    *,
    hyperedges_txt: str,
    node_text_json: str,
    edge_labels_txt: str,
    edge_text_json: str,
    disease_filter_json: str,
    relevance_txt: str,
    paths_json: str,
    model: str,
    output_jsonl: str,
    progress_file: str,
    checkpoint_every: int = 20,
    sample_patients: Optional[int] = None,
    sample_seed: int = 42,
    view_cot: bool = False,
    early_stop: Optional[int] = None,            # legacy (pairs)
    early_stop_patients: Optional[int] = None,   # NEW (patients)
    max_absent: int = 50,
    absent_shuffle_seed: int = 123,
    eval_records: List[dict] = [],
    no_kg_context: bool = False,
):
    # Load artifacts
    hyperedges = load_hyperedges_braced(hyperedges_txt)
    edge_labels = load_edge_labels_braced(edge_labels_txt)
    node_text = load_json(node_text_json)
    edge_text = load_json(edge_text_json)
    disease_filter = load_disease_filter(disease_filter_json)

    # Sanity alignment
    n_pat = min(len(hyperedges), len(edge_labels))
    if n_pat == 0:
        raise ValueError("No patients found across hyperedges / edge-labels.")

    # Allowed disease edge indices
    allowed_edge_indices: Set[int] = build_allowed_edge_indices(edge_text, disease_filter)
    if not allowed_edge_indices:
        raise ValueError("No allowed diseases found (disease_filter produced an empty set).")

    # Relevance & paths
    _, relevance_norm = load_relevance(relevance_txt)
    paths_norm = load_path_mapping(paths_json)

    # Sample patients FIRST
    all_patients = collect_all_patient_ids_from_graph(hyperedges, edge_labels)
    if sample_patients is not None and sample_patients > 0:
        sampled_list = sample_ids(all_patients, k=sample_patients, seed=sample_seed)
    else:
        sampled_list = all_patients
    sampled_patients: Set[int] = set(sampled_list)

    # Build worklist (patient, edge_idx, disease_name)
    worklist = build_worklist_graph(
        sampled_patients=sampled_patients,
        allowed_edge_indices=allowed_edge_indices,
        edge_labels=edge_labels,
        edge_text=edge_text,
    )
    total = len(worklist)
    if total == 0:
        raise ValueError("Worklist is empty after applying allowed-edge filter; check disease_filter & edge_text indices.")

    start_idx = read_progress(progress_file)
    start_idx = max(0, min(start_idx, total))

    if early_stop_patients is not None:
        end_idx = end_index_for_first_n_patients_in_worklist(worklist, start_idx, early_stop_patients, pid_pos=0)
    else:
        end_idx = max(start_idx, min(early_stop, total)) if early_stop is not None else total

    pbar = tqdm(total=total, initial=start_idx, desc="Setting-3 (GRAPH)", ncols=100)
    buffer: List[dict] = []
    next_idx = start_idx

    try:
        for idx in range(start_idx, end_idx):
            pid, eidx, disease_name = worklist[idx]
            pid_int = int(pid)

            # Present features via hyperedges -> node_text
            present_ids = hyperedges[pid] if pid < len(hyperedges) else []
            present_features = []
            for nid in present_ids:
                name = node_text.get(str(nid))
                if name:
                    present_features.append(name)

            # Optional absent features (kept for possible future use; not placed in row "input")
            feat_set = set(int(x) for x in present_ids)
            absent_features = []
            for k, name in node_text.items():
                try:
                    k_int = int(k)
                except Exception:
                    continue
                if k_int not in feat_set and name:
                    absent_features.append(name)
            if max_absent is not None and max_absent > 0 and len(absent_features) > max_absent:
                rng_abs = random.Random(absent_shuffle_seed + pid)
                rng_abs.shuffle(absent_features)
                absent_features = absent_features[:max_absent]

            # Label for this (patient, disease-edge)
            row = edge_labels[pid]
            if eidx >= len(row):
                continue
            label = int(row[eidx])  # 0/1
            answer = "**True**" if label == 1 else "**False**"

            # Relevance-based partition (by disease name)
            ents = get_relevant_entities(relevance_norm, disease_name)
            pot_present, pot_absent = partition_potential_relevance(present_features, ents)

            # Paths for this disease
            paths = get_paths_for_disease(paths_norm, disease_name)

            # ==== prompts for generation ====
            gen_user_input = build_setting3_user_input(
                disease=disease_name,
                present_features=present_features,
                pot_rel_present=pot_present,
                pot_rel_absent=pot_absent,
                answer_yes_no=answer,
                paths=None if no_kg_context else paths,
                include_kg=not no_kg_context,
            )
            cot = call_openai_responses(
                model=model,
                system_prompt=SYSTEM_PROMPT_WITHOUT_KG if no_kg_context else SYSTEM_PROMPT,
                user_input=gen_user_input,
                max_output_tokens=2048,
            )
            _append_eval_record(eval_records, disease_name=disease_name, gold=label, model_text=cot)

            # ==== dataset-facing fields per your spec ====
            instruction = format_instruction(disease_name)
            user_input_for_row = format_user_query(
                patient_record=present_features,
                disease_name=disease_name,
                pot_rel_present=pot_present,
                pot_rel_absent=pot_absent,
                paths=None if no_kg_context else paths,
                include_kg=not no_kg_context,
            )

            row_obj = {
                "instruction": instruction,
                "input": user_input_for_row,
                "output": cot,
                "label": label,
                "disease": disease_name,
                "patient_id": pid_int,
            }
            buffer.append(row_obj)

            next_idx = idx + 1
            pbar.update(1)

            if view_cot:
                print(f"\n--- disease={disease_name} | patient={pid_int} ---\n{gen_user_input}\n\n{cot}\n{'='*60}")

            if len(buffer) >= checkpoint_every:
                append_jsonl(output_jsonl, buffer)
                write_progress(progress_file, next_idx)
                buffer.clear()
                pbar.set_postfix_str(f"flushed@{next_idx}")

        if buffer:
            append_jsonl(output_jsonl, buffer)
            write_progress(progress_file, next_idx)
            buffer.clear()

    finally:
        pbar.close()
        examine_conflict(path=output_jsonl)

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Generate COT (setting-3) from CSV or GRAPH sources with disease_filter and sampling-first."
    )

    # Common
    ap.add_argument(
        "--model",
        default=AZURE_OPENAI_DEPLOYMENT,
        help="Azure OpenAI deployment/model name (defaults to AZURE_OPENAI_DEPLOYMENT or gpt-4o).",
    )
    ap.add_argument("--output_jsonl", required=True, help="Destination JSONL for generations.")
    ap.add_argument("--progress_file", required=True, help="Progress checkpoint file.")
    ap.add_argument("--checkpoint_every", type=int, default=20)
    ap.add_argument("--sample_patients", type=int, default=None, help="Sample N distinct patients (deterministic).")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--view_cot", action="store_true")

    # Early stop options
    ap.add_argument("--early_stop", type=int, default=None, help="(Legacy) Stop after N pairs.")
    ap.add_argument("--early_stop_patients", type=int, default=None, help="Stop after N unique patients from current resume point.")

    # Relevance/paths (used in both modes)
    ap.add_argument("--relevance_txt", required=True, help="JSON-ish relevance file.")
    ap.add_argument("--paths_json", required=True, help="filtered_path_mappings.json (disease -> list of paths).")
    ap.add_argument("--no_kg_context", action="store_true",
                    help="Disable KG reasoning paths in prompts and use SYSTEM_PROMPT_WITHOUT_KG.")

    # CSV mode
    ap.add_argument("--csv_mode", action="store_true", help="Use CSV workflow.")
    ap.add_argument("--patients_csv", help="CSV with columns: patient_id(optional), features_present, and 0/1 disease columns.")
    ap.add_argument("--features_col", default="features_present")
    ap.add_argument("--exclude_cols", nargs="*", default=["patient_id", "features_present"])

    # GRAPH mode
    ap.add_argument("--graph_mode", action="store_true", help="Use graph/hyperedge workflow.")
    ap.add_argument("--hyperedges_txt", help="hyperedges file (for this repo: hyperedges-mimic3_truncated.txt)")
    ap.add_argument("--node_text_json", help="node_text.json mapping node id -> feature text")
    ap.add_argument("--edge_labels_txt", help="edge-labels file (for this repo: edge-labels-mimic3_updated_truncated.txt)")
    ap.add_argument("--edge_text_json", help="edge_text.json mapping edge id -> disease name")
    ap.add_argument("--disease_filter_json", help="disease_filter.json: disease name -> [alias/null, numeric_id]")
    ap.add_argument("--max_absent", type=int, default=50, help="Max absent features to include after shuffle.")
    ap.add_argument("--absent_shuffle_seed", type=int, default=123, help="Shuffle seed for absent features.")

    args = ap.parse_args()
    eval_records: List[dict] = []

    # Exactly one mode must be chosen
    if args.csv_mode == args.graph_mode:
        raise SystemExit("Choose exactly one: --csv_mode OR --graph_mode")

    if args.csv_mode:
        if not args.patients_csv:
            raise SystemExit("--patients_csv is required in --csv_mode")
        process_setting3_csv(
            patients_csv=args.patients_csv,
            relevance_txt=args.relevance_txt,
            paths_json=args.paths_json,
            features_col=args.features_col,
            exclude_cols=args.exclude_cols,
            model=args.model,
            output_jsonl=args.output_jsonl,
            progress_file=args.progress_file,
            checkpoint_every=args.checkpoint_every,
            sample_patients=args.sample_patients,
            sample_seed=args.sample_seed,
            view_cot=args.view_cot,
            early_stop=args.early_stop,
            early_stop_patients=args.early_stop_patients,
            eval_records=eval_records,   # <-- PASS IT IN
            no_kg_context=args.no_kg_context,
        )
    else:
        needed = [
            args.hyperedges_txt, args.node_text_json,
            args.edge_labels_txt, args.edge_text_json,
            args.disease_filter_json
        ]
        if any(not x for x in needed):
            raise SystemExit(
                "GRAPH mode requires: --hyperedges_txt --node_text_json --edge_labels_txt "
                "--edge_text_json --disease_filter_json"
            )

        process_setting3_graph(
            hyperedges_txt=args.hyperedges_txt,
            node_text_json=args.node_text_json,
            edge_labels_txt=args.edge_labels_txt,
            edge_text_json=args.edge_text_json,
            disease_filter_json=args.disease_filter_json,
            relevance_txt=args.relevance_txt,
            paths_json=args.paths_json,
            model=args.model,
            output_jsonl=args.output_jsonl,
            progress_file=args.progress_file,
            checkpoint_every=args.checkpoint_every,
            sample_patients=args.sample_patients,
            sample_seed=args.sample_seed,
            view_cot=args.view_cot,
            early_stop=args.early_stop,
            early_stop_patients=args.early_stop_patients,
            max_absent=args.max_absent,
            absent_shuffle_seed=args.absent_shuffle_seed,
            eval_records=eval_records,   # <-- PASS IT IN
            no_kg_context=args.no_kg_context,
        )

if __name__ == "__main__":
    main()
