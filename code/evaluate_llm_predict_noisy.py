#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Purpose: Main multilabel evaluation pipeline for noisy/full MIMIC or CRADLE-style inputs.

import argparse, json, os, re, sys, random, csv
import faulthandler

faulthandler.enable()
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
except Exception as e:
    raise RuntimeError("scikit-learn is required for this evaluation block.") from e

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback

# ============================================================
# Small utils
# ============================================================

FEWSHOT_CRADLE = r"""
### Few-shot Example (CRADLE)

Target outcome: Cardiovascular disease within 1 year after type 2 diabetes diagnosis

Patient features present (subset shown):
- Type II diabetes mellitus without complication (disorder)
- Benign essential hypertension / Essential hypertension (disorder)
- CKD stage 3; Chronic renal disease; Disorder of kidney due to diabetes mellitus
- Proteinuria (finding); Albumin (urine microalbumin)
- Long-term current use of aspirin; Beta blocking agents; Calcium channel blockers; DPP-4 inhibitors; Combinations of oral blood glucose lowering drugs
- Routine outpatient encounters and labs (A1c, creatinine, urinalysis, glucose monitoring)
- Age-related osteoporosis; DXA scan
- Liver transplant recipient; Calcineurin inhibitors; Transplant follow-up

## Finding Reasoning Paths
1) T2DM → Diabetic kidney disease → CKD stage 3 → Proteinuria → ↑ baseline CVD risk  
2) Hypertension on maintenance therapy → chronic follow-up pattern, no acute CVD workflow  
3) Antiplatelet prophylaxis present but no post-event pattern (e.g., no DAPT, no revascularization)  
4) Transplant + immunosuppression increases systemic risk but is non-specific for near-term CVD event  
5) Geriatric/osteoporosis signals raise baseline risk, still no direct evidence of an outcome within 1 year

## Analysis
- Evidence provided is **risk factors** and routine management, not an **outcome signal**.  
- No codes/procedures/medication patterns consistent with **MI/stroke/ACS/revascularization** within the prediction horizon.  
- Presence-only setting ⇒ unlisted items are **unknown**, not absent.

## Conclusion
Elevated baseline CVD risk but **no positive evidence of a CVD outcome within 1 year** after T2DM index event.

Final answer: No
""".strip()
def _pid_sort_key(pid: Any):
    if isinstance(pid, str) and pid.isdigit():
        return (0, int(pid))
    return (1, str(pid))

# ---------------- NEW/SHARED LOADERS ----------------
def load_hyperedges(file_path: str) -> List[List[int]]:
    rows: List[List[int]] = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                rows.append([])
                continue
            rows.append([int(i) for i in line.split(",") if i != ""])
    return rows

def load_edge_indicator_matrix(file_path: str) -> List[List[int]]:
    """
    Each line is '0,1,0,...' aligned to disease indices from edge_txt.json.
    Returns for each patient a list of disease ids (column indices) where value==1.
    """
    rows_present_ids: List[List[int]] = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                rows_present_ids.append([])
                continue
            bits = [int(x) for x in line.split(",") if x != ""]
            present = [i for i, b in enumerate(bits) if b == 1]
            rows_present_ids.append(present)
    return rows_present_ids
    
# >>> CHANGED/NEW <<<
def load_id_list_file(file_path: str) -> List[List[int]]:
    """
    Same CSV-line format as hyperedges: each line is 'id1,id2,...' (or empty).
    Used here for edge_labels_txt: per-patient disease node ids that are PRESENT.
    """
    rows: List[List[int]] = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                rows.append([])
                continue
            rows.append([int(i) for i in line.split(",") if i != ""])
    return rows

    
# ============================================================
# Prompt helpers (kept consistent with your pipeline)
# ============================================================

SYSTEM_PROMPT_NO_KG = (
    "Given a disease to be verified, a list of diagnostic_features_present measured at the index visit (time t), "
    "a subset of potentially_relevant_exist_features of confirmed positive features, a list of "
    "potentially_relevant_absent_features of confirmed negative, and no reasoning-path hints, your task is to "
    "reason step by step as if you are independently determining whether the disease will be present at the next "
    "visit with **Yes** or **No**."
)


def format_instruction(
    disease: str,
    *,
    system_prompt_variant: str = "default",
    custom_system_prompt: Optional[str] = None,
) -> str:
    """
    Build the system prompt.

    Priority:
      1) If custom_system_prompt is provided, use it (supports {disease} interpolation).
      2) Else choose a built-in variant by `system_prompt_variant`.
         - "default": original behavior
         - "cradle":  presence-only features filtered by MIMIC names
    """
    if custom_system_prompt:
        try:
            body = custom_system_prompt.format(disease=disease)
        except Exception:
            # if the custom file doesn't have {disease}, just append the disease line
            body = f"{custom_system_prompt}"
        # Ensure the final instruction guidance exists
        final = "Reason and return the final answer derived from your reasoning, with **Yes** or **No** printed and specified in the end."
        return body.strip() + "\n" + final

    if system_prompt_variant.lower() == "cradle":
        base = (
            "Given a disease to be verified within 1 year after index visit, a list of diagnostic_features_present measured at the index visit (time t), a subset of potentially_relevant_exist_features of confirmed positive features, a list of potentially_relevant_absent_features of confirmed negative, your task is to reason step by step as if you are independently determining whether the disease will be present at the next visit with **Yes** or **No**.\n"
        )
        base = base
    elif system_prompt_variant.lower() == "no_kg":
        base = SYSTEM_PROMPT_NO_KG + "\n"
    else:  # "default"
        base = (
            "Given a disease to be verified, a list of diagnostic_features_present measured at the index visit (time t), a subset of potentially_relevant_exist_features of confirmed positive features, a list of potentially_relevant_absent_features of confirmed negative, your task is to reason step by step as if you are independently determining whether the disease will be present at the next visit with **Yes** or **No**.\n"
        )
    return base

def _norm_key(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\b,?\s*genetic\b", "", s)   # match Code B quirk
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_relevance_jsonish(text: str) -> Dict[str, dict]:
    """
    Accepts the raw text of a 'JSON-ish' relevance file (same forgiving behavior as Code B).
    Returns a normalized index: norm_key(disease) -> blob
    """
    t = text.strip()
    if not t.lstrip().startswith("{"):
        t = "{\n" + t + "\n}"
    t = re.sub(r"\}\s*\n\s*\"", "},\n\"", t)
    data = json.loads(t)
    return { _norm_key(k): v for k, v in data.items() }

def get_relevant_entities(relevance_norm: Dict[str, dict], disease: str) -> List[str]:
    block = relevance_norm.get(_norm_key(disease), {})
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


def load_path_mapping(path_json: str) -> Dict[str, List[str]]:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {_norm_key(k): v for k, v in data.items()}


def _paths_for_disease(paths_norm: Optional[Dict[str, List[str]]], disease: str) -> List[str]:
    if not paths_norm:
        return []
    return paths_norm.get(_norm_key(disease), [])

# --- exact input formatter used in Code B ---
def _join(xs: List[str]) -> str:
    return ", ".join(xs) if xs else "(none)"

def format_user_query_alpaca(
    patient_record: List[str],
    disease_name: str,
    pot_rel_present: List[str],
    pot_rel_absent: List[str],
    paths: Optional[List[str]] = None,
    include_paths: bool = True,
) -> str:
    """
    Build the evaluation-time user prompt (mirrors generation) with reasoning paths appended.
    """
    parts = [
        "### Input:\n",
        f"Disease to be verified: {disease_name}\n\n",
        f"diagnostic_features_present:  {_join(patient_record)}\n\n",
        f"potentially_relevant_present_features: {_join(pot_rel_present)}\n\n",
        f"potentially_relevant_absent_features: {_join(pot_rel_absent)}\n\n",
    ]
    if include_paths:
        paths_block = "\n".join(f"- {p}" for p in (paths or [])) if paths else "(none)"
        parts.append(f"Reasoning Paths:\n{paths_block}\n\n")
    return "".join(parts)


def format_user_query_with_presence(
    disease_name: str,
    present_features: List[str],
    absent_features: Optional[List[str]] = None,
    include_absent_in_prompt: bool = False,
) -> str:
    """COT-style prompt formatting."""
    def j(xs: List[str]) -> str:
        return ", ".join(xs) if xs else "(none)"
    base = (
        "### Input:\n"
        f"Patient features present: {j(present_features)}\n"
        f"Disease to be verified: {disease_name}\n"
    )
    if include_absent_in_prompt and absent_features is not None:
        base += f"Patient features absent(truncated): {j(absent_features)}\n"
    return base


def to_patient_present_absent(
    patient_id: int,
    hyperedges: List[List[int]],         # patient -> feature node ids
    feature_node_text: Dict[str, str],   # node_id(str) -> feature name
    max_absent: int = 50,
    absent_shuffle_seed: int = 123,
    *,
    allowed_name_norm_set: Optional[set[str]] = None,  # <- NEW: normalized names from MIMIC node_txt.json
    filter_absent_by_allowed: bool = False,            # <- optional: usually False per your spec
) -> Tuple[List[str], List[str]]:
    """
    Returns (present_features, absent_features) as name lists for the patient.

    If `allowed_name_norm_set` is provided (derived from MIMIC node_txt.json VALUES),
    we KEEP a present feature only if its NAME (normalized) is in that set.
    By default we do NOT filter the absent list (set filter_absent_by_allowed=True if you want that too).
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    feats = hyperedges[patient_id] if patient_id < len(hyperedges) else []
    feat_set = set(int(x) for x in feats if isinstance(x, int))

    present_features: List[str] = []
    for fidx in feats:
        name = feature_node_text.get(str(fidx))
        if not name:
            continue
        if allowed_name_norm_set is not None and _norm(name) not in allowed_name_norm_set:
            continue  # gate by MIMIC names (cradle case)
        present_features.append(name)

    absent_features: List[str] = []
    for k, name in feature_node_text.items():
        try:
            k_int = int(k)
        except Exception:
            continue
        if k_int in feat_set or not name:
            continue
        if filter_absent_by_allowed and allowed_name_norm_set is not None:
            if _norm(name) not in allowed_name_norm_set:
                continue
        absent_features.append(name)

    # optional truncation/shuffle for absent list
    if max_absent is not None and max_absent > 0 and len(absent_features) > max_absent:
        rng_abs = random.Random(absent_shuffle_seed + patient_id)  # stable per patient
        rng_abs.shuffle(absent_features)
        absent_features = absent_features[:max_absent]

    return present_features, absent_features

# ============================================================
# Parsing helpers (extract Yes/No from model output)
# ============================================================

_FINAL_RE = re.compile(r"Final answer:\s*(Yes|No)\b", flags=re.IGNORECASE)
_LAST_YN_RE = re.compile(r"\b(Yes|No)\b", flags=re.IGNORECASE)
_FINAL_TF_RE = re.compile(r"Final answer:\s*(True|False)\b", flags=re.IGNORECASE)
_LAST_TF_RE = re.compile(r"\b(True|False)\b", flags=re.IGNORECASE)

def parse_final_yes_no(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).title()
    matches = _LAST_YN_RE.findall(text)
    if matches:
        return matches[-1].title()
    return None


def _last_nonempty_line(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_final_true_false(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None
    m = _FINAL_TF_RE.search(text)
    if m:
        return 1 if m.group(1).lower() == "true" else 0
    matches = _LAST_TF_RE.findall(text)
    if matches:
        return 1 if matches[-1].lower() == "true" else 0
    return None

# ============================================================
# Chat rendering
# ============================================================

def _augment_instruction(instruction: str) -> str:
    return (instruction or "").strip()

def build_messages(instruction: str, user_text: str) -> List[Dict[str, str]]:
    instruction = _augment_instruction(instruction)
    user_text   = (user_text or "").strip()
    return [
        {"role": "system", "content": instruction},
        {"role": "user",   "content": user_text},
    ]

def _render_gemma(messages: List[Dict[str, str]]) -> str:
    sys_txt = ""
    usr_txt = ""
    for m in messages:
        if m["role"] == "system":
            sys_txt += (m["content"] or "").strip()
        elif m["role"] == "user":
            usr_txt += (m["content"] or "").strip()
    merged_user = (sys_txt + ("\n\n" if sys_txt and usr_txt else "") + usr_txt).strip()
    return (
        f"<start_of_turn>user\n{merged_user}\n<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def _render_fallback_plain(messages: List[Dict[str, str]]) -> str:
    sys_txt = ""
    usr_txt = ""
    for m in messages:
        if m["role"] == "system":
            sys_txt += (m["content"] or "").strip()
        elif m["role"] == "user":
            usr_txt += (m["content"] or "").strip()
    return (sys_txt + "\n\n" + usr_txt + "\n").strip()

def _looks_like_gemma(model_id_or_name: str, tokenizer) -> bool:
    mid = (model_id_or_name or "").lower()
    if "gemma" in mid:
        return True
    tmpl = getattr(tokenizer, "chat_template", None)
    return isinstance(tmpl, str) and "start_of_turn" in tmpl

def render_prompt(
    tokenizer,
    model_id_or_name: str,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True
) -> str:
    if _looks_like_gemma(model_id_or_name, tokenizer):
        return _render_gemma(messages)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception as e:
        msg = str(e).lower()
        if "system role not supported" in msg or "chat template" in msg or "jinja" in msg:
            if _looks_like_gemma(model_id_or_name, tokenizer):
                return _render_gemma(messages)
            return _render_fallback_plain(messages) + "\nFinal answer:"
        return _render_fallback_plain(messages) + "\nFinal answer:"

# ============================================================
# Deterministic generation
# ============================================================

# helper to find subsequence in list
def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """
    Return the first index where `needle` appears inside `haystack`, or -1 if absent.
    """
    if not needle or len(needle) > len(haystack):
        return -1
    last = len(haystack) - len(needle)
    for idx in range(last + 1):
        if haystack[idx:idx + len(needle)] == needle:
            return idx
    return -1


@torch.inference_mode()
def generate_reason_and_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[Optional[int], float, str]:
    """
    Returns (pred_label, p_yes, generated_text)
      - pred_label: 1 for Yes, 0 for No, None if unparsed
      - p_yes: 1.0 if Yes, 0.0 if No, 0.5 if None
      - generated_text: decoded assistant text
    """
    model_id_or_name = getattr(model, "name_or_path", "") or ""
    prompt_text = render_prompt(tokenizer, model_id_or_name, messages, add_generation_prompt=True)

    enc = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    do_sample = (temperature or 0.0) > 0.0
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = gen_out.sequences
    scores = gen_out.scores or []
    gen_cont = sequences[:, input_ids.shape[1]:]
    text = tokenizer.decode(gen_cont[0], skip_special_tokens=True).strip()

    decision_kind: Optional[str] = None
    decision_word: Optional[str] = None

    yn = parse_final_yes_no(text)
    if yn in ("Yes", "No"):
        decision_kind = "yesno"
        decision_word = yn
    else:
        m = re.findall(r"\b(Yes|No)\b", text, flags=re.IGNORECASE)
        if m:
            yn = m[-1].title()
            decision_kind = "yesno"
            decision_word = yn
        else:
            tf = parse_final_true_false(text)
            if tf in (0, 1):
                decision_kind = "truefalse"
                decision_word = "True" if tf == 1 else "False"
                yn = "Yes" if tf == 1 else "No"

    p_yes = 0.5
    pred = None

    if yn is None or decision_word is None:
        return None, 0.5, text

    pred = 1 if yn == "Yes" else 0

    # Determine candidate decision tokens and positive/negative ids
    decision_ids_primary = tokenizer.encode(decision_word, add_special_tokens=False)
    decision_ids_alt = tokenizer.encode(" " + decision_word, add_special_tokens=False)
    decision_token_variants = [
        [tok for tok in decision_ids_primary if tok >= 0],
        [tok for tok in decision_ids_alt if tok >= 0],
    ]
    decision_token_variants = [seq for seq in decision_token_variants if seq]

    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False) or tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False) or tokenizer.encode("No", add_special_tokens=False)
    true_ids = tokenizer.encode(" True", add_special_tokens=False) or tokenizer.encode("True", add_special_tokens=False)
    false_ids = tokenizer.encode(" False", add_special_tokens=False) or tokenizer.encode("False", add_special_tokens=False)

    if decision_kind == "truefalse":
        positive_ids = true_ids
        negative_ids = false_ids
    else:
        positive_ids = yes_ids
        negative_ids = no_ids

    if not decision_token_variants:
        if yn == "Yes" and positive_ids:
            decision_token_variants = [positive_ids]
        elif yn == "No" and negative_ids:
            decision_token_variants = [negative_ids]

    # Locate final line tokens to anchor the verdict
    last_line = _last_nonempty_line(text)
    line_tokens = tokenizer.encode(last_line, add_special_tokens=False)
    gen_token_list = gen_cont[0].tolist()

    decision_position = None
    if scores and gen_token_list and line_tokens:
        span_start = None
        for idx in range(len(gen_token_list) - len(line_tokens), -1, -1):
            if gen_token_list[idx:idx + len(line_tokens)] == line_tokens:
                span_start = idx
                break

        if span_start is not None:
            for candidate in decision_token_variants:
                offset = _find_subsequence(line_tokens, candidate)
                if offset >= 0:
                    decision_position = span_start + offset
                    break

    # If we still failed to locate the answer token, fall back to last occurrence in generated tokens
    if decision_position is None and decision_token_variants:
        target_seq = decision_token_variants[0]
        for idx in range(len(gen_token_list) - len(target_seq), -1, -1):
            if gen_token_list[idx:idx + len(target_seq)] == target_seq:
                decision_position = idx
                break

    if decision_position is not None and scores:
        token_idx = decision_position
        if token_idx < len(scores):
            logits = scores[token_idx][0]
            probs = torch.softmax(logits, dim=-1)

            pos_id = positive_ids[0] if positive_ids else None
            neg_id = negative_ids[0] if negative_ids else None

            if pos_id is not None and neg_id is not None:
                pos_prob = probs[pos_id].item()
                neg_prob = probs[neg_id].item()
                norm = pos_prob + neg_prob
                if norm > 0:
                    p_yes = pos_prob / norm
                else:
                    p_yes = pos_prob
            else:
                p_yes = probs[pos_id].item() if pos_id is not None else 0.5

    return pred, p_yes, text

# ============================================================
# Metrics: Block-2 style (macro over labels, micro flattened)
# ============================================================

def _safe_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan

def _safe_aupr(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except ValueError:
        return np.nan

def eval_multilabel_block2_style(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    mode: str = "eval",
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    y_true, y_score: (num_patients, num_labels)
    MACRO: mean over labels; MICRO: over all elements.
    Adds macro balanced accuracy (macro BACC).
    """
    assert y_true.shape == y_score.shape
    num_labels = y_true.shape[1]
    y_pred = (y_score > threshold).astype(int)

    per_label_acc, per_label_auc, per_label_aupr, per_label_f1, per_label_bacc = [], [], [], [], []
    per_label_stats: List[Dict[str, Any]] = []
    for i in range(num_labels):
        yi = y_true[:, i]
        pi = y_pred[:, i]
        si = y_score[:, i]
        label_name = None
        if label_names and i < len(label_names):
            label_name = label_names[i]

        per_label_acc.append((pi == yi).mean())
        per_label_f1.append(f1_score(yi, pi, average='binary', zero_division=0))
        # safe auc/aupr (handle single-class labels)
        try:
            per_label_auc.append(roc_auc_score(yi, si))
        except ValueError:
            per_label_auc.append(np.nan)
        try:
            per_label_aupr.append(average_precision_score(yi, si))
        except ValueError:
            per_label_aupr.append(np.nan)
        # balanced accuracy can also fail if only one class present; treat as NaN
        try:
            per_label_bacc.append(balanced_accuracy_score(yi, pi))
        except ValueError:
            per_label_bacc.append(np.nan)
        per_label_stats.append({
            "label_index": i,
            "label_name": label_name if label_name is not None else f"label_{i}",
            "support_total": int(len(yi)),
            "support_positive": int(yi.sum()),
            "support_negative": int(len(yi) - yi.sum()),
            "acc": float(per_label_acc[-1]),
            "bacc": float(per_label_bacc[-1]) if not np.isnan(per_label_bacc[-1]) else float('nan'),
            "auc": float(per_label_auc[-1]) if not np.isnan(per_label_auc[-1]) else float('nan'),
            "aupr": float(per_label_aupr[-1]) if not np.isnan(per_label_aupr[-1]) else float('nan'),
            "f1": float(per_label_f1[-1]),
        })

    macro = dict(
        acc = float(np.nanmean(per_label_acc)),
        bacc= float(np.nanmean(per_label_bacc)),  # << NEW
        auc = float(np.nanmean(per_label_auc)),
        aupr= float(np.nanmean(per_label_aupr)),
        f1  = float(np.nanmean(per_label_f1)),
    )

    micro_acc = (y_pred == y_true).mean()
    micro_f1  = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='micro', zero_division=0)
    # micro AUC/AUPR over flattened vectors
    try:
        micro_auc = roc_auc_score(y_true.reshape(-1), y_score.reshape(-1))
    except ValueError:
        micro_auc = float('nan')
    try:
        micro_aupr= average_precision_score(y_true.reshape(-1), y_score.reshape(-1))
    except ValueError:
        micro_aupr = float('nan')

    micro = dict(acc=float(micro_acc), auc=float(micro_auc), aupr=float(micro_aupr), f1=float(micro_f1))

    print(f'[{mode.upper()}] MICRO  -> ACC:{micro["acc"]:.6f}, AUC:{micro["auc"]:.6f}, AUPR:{micro["aupr"]:.6f}, F1:{micro["f1"]:.6f}')
    print(f'[{mode.upper()}] MACRO  -> BACC:{macro["bacc"]:.6f}, ACC:{macro["acc"]:.6f}, AUC:{macro["auc"]:.6f}, AUPR:{macro["aupr"]:.6f}, F1:{macro["f1"]:.6f}')

    return {"micro": micro, "macro": macro, "per_label": per_label_stats}


def write_per_disease_tables(stats: List[Dict[str, Any]], out_dir: str, prefix: str) -> None:
    if not stats:
        return
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{prefix}.json")
    csv_path = os.path.join(out_dir, f"{prefix}.csv")

    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(stats, f_json, ensure_ascii=False, indent=2)

    fieldnames = [
        "label_index",
        "label_name",
        "support_total",
        "support_positive",
        "support_negative",
        "acc",
        "bacc",
        "auc",
        "aupr",
        "f1",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow({k: row.get(k) for k in fieldnames})

# ============================================================
# Build dataset rows (prompts) in deterministic disease order
# ============================================================

# ------------- NEW BUILDER (NO PKL NEEDED) -------------
def build_rows_from_txt_labels(
    sampled_patients: List[int],
    feature_node_text: Dict[str, str],          # features (node_text.json)
    disease_node_text: Optional[Dict[str, str]],# UNUSED; kept for signature compatibility
    hyperedges: List[List[int]],
    edge_labels: List[List[int]],
    *,
    diseases_in_order: List[str],                 # final label order
    disease_name_to_index: Dict[str, int],        # name -> column index
    include_absent_in_prompt: bool = False,
    max_absent: int = 50,
    absent_shuffle_seed: int = 123,
    allowed_name_norm_set: Optional[set[str]] = None,
    filter_absent_by_allowed: bool = False,
    system_prompt_variant: str = "default",
    custom_system_prompt: Optional[str] = None,
    relevance_norm: Optional[Dict[str, dict]] = None,
    paths_norm: Optional[Dict[str, List[str]]] = None,
    include_paths: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    for pid in sampled_patients:
        present_features, absent_features = to_patient_present_absent(
            pid,
            hyperedges,
            feature_node_text,
            max_absent=max_absent,
            absent_shuffle_seed=absent_shuffle_seed,
            allowed_name_norm_set=allowed_name_norm_set,
            filter_absent_by_allowed=filter_absent_by_allowed,
        )

        present_disease_ids = set(edge_labels[pid]) if pid < len(edge_labels) else set()

        for disease in diseases_in_order:
            did = disease_name_to_index[disease]
            gold = 1 if did in present_disease_ids else 0

            instruction = format_instruction(
                disease,
                system_prompt_variant=system_prompt_variant,
                custom_system_prompt=custom_system_prompt,
            )
            pot_rel_present, pot_rel_absent = [], []
            if relevance_norm is not None:
                ents = get_relevant_entities(relevance_norm, disease)
                pot_rel_present, pot_rel_absent = partition_potential_relevance(present_features, ents)

            reasoning_paths = _paths_for_disease(paths_norm if include_paths else None, disease)

            # ---- NEW: use the Code-B style user input formatter ----
            user_input = format_user_query_alpaca(
                patient_record=present_features,
                disease_name=disease,
                pot_rel_present=pot_rel_present,
                pot_rel_absent=pot_rel_absent,
                paths=reasoning_paths,
                include_paths=include_paths,
            )
            rows.append({
                "patient_id": pid,
                "disease": disease,
                "instruction": instruction,
                "input": user_input,
                "label": gold,
                "present_features": present_features,
                "absent_features": absent_features,
            })
    return rows, diseases_in_order

# ============================================================
# Main evaluation pipeline
#   1) Sample unseen patient IDs (deterministic)
#   2) Build prompts per (patient × disease) with fixed disease order
#   3) Run model → parse Yes/No → per-patient multilabel tensors
#   4) Evaluate with Block-2 style metrics
# ============================================================


@torch.inference_mode()
def evaluate_multilabel_with_sampling(
    model_dir: str,
    unseen_ids_json: str,
    node_text_json: str,        # features (node_text.json)
    edge_txt_json: str,         # diseases (edge_txt.json)
    hyperedges_txt: str,
    edge_labels_txt: str,
    out_dir: str,
    total_patients: int,
    ratio: float,
    seed: int,
    disease_whitelist_json: Optional[str] = None,
    device: str = "cuda",
    max_eval_rows: Optional[int] = None,
    verbose: bool = False,
    log_file: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    threshold: float = 0.5,
    include_absent_in_prompt: bool = False,
    max_absent: int = 50,
    absent_shuffle_seed: int = 123,
    mimic_name_json: Optional[str] = None,      # name-level gating (MIMIC)
    filter_absent_by_allowed: bool = False,
    dataset: Optional[str] = None,              # <- NEW: e.g., "cradle", "mimic", "mimic_trunc"
    system_prompt_file: Optional[str] = None,   # <- NEW: full override file
    relevance_txt: Optional[str] = None,
    paths_json: Optional[str] = None,
    no_kg_context: bool = False,
) -> Dict[str, Dict[str, float]]:

    os.makedirs(out_dir, exist_ok=True)

    # 1) sample patients
    with open(unseen_ids_json, "r") as f:
        unseen_ids = json.load(f)
    if not isinstance(unseen_ids, list) or not unseen_ids:
        raise ValueError("unseen_patient_ids.json must be a non-empty JSON list")

    k_target = int(total_patients * ratio)
    k = min(k_target, len(unseen_ids))
    if k <= 0:
        raise ValueError(f"Non-positive sample size computed: {k}")

    rng = random.Random(seed)
    unseen_sorted = sorted(unseen_ids, key=_pid_sort_key)
    sampled = [int(x) for x in rng.sample(unseen_sorted, k)]

    # 2) load sources
    with open(node_text_json, "r") as f:
        feature_node_text = json.load(f)   # features
    with open(edge_txt_json, "r") as f:
        disease_node_text = json.load(f)   # diseases

    hyperedges  = load_id_list_file(hyperedges_txt)            # features per patient (IDs list)
    edge_labels = load_edge_indicator_matrix(edge_labels_txt)  # diseases per patient (indices where 1)

    relevance_norm = None
    if relevance_txt is not None:
        with open(relevance_txt, "r", encoding="utf-8") as f:
            relevance_norm = load_relevance_jsonish(f.read())

    paths_norm = None
    if paths_json is not None:
        paths_norm = load_path_mapping(paths_json)
    # 2b) Build allowed NAME set if provided (MIMIC names)
    allowed_name_norm_set: Optional[set[str]] = None
    if mimic_name_json:
        with open(mimic_name_json, "r") as f:
            mimic_map = json.load(f)  # id -> name
        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", str(s)).strip().lower()
        allowed_name_norm_set = {_norm(v) for v in mimic_map.values() if isinstance(v, str) and v.strip()}

    # 2c) Load optional custom system prompt from file
    custom_system_prompt: Optional[str] = None
    if system_prompt_file:
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            custom_system_prompt = f.read()

    # 3) column order from edge_txt.json
    pairs = sorted(((int(k), v) for k, v in disease_node_text.items()), key=lambda x: x[0])
    all_diseases_in_index_order = [name for (_idx, name) in pairs]
    name_to_index_full = {name: idx for (idx, name) in pairs}
    num_labels_from_json = len(pairs)

    # Verify width matches the number of diseases
    with open(edge_labels_txt, "r") as _f:
        first_line = next((ln.strip() for ln in _f if ln.strip()), "")
        width = len([x for x in first_line.split(",") if x != ""])
    assert width == num_labels_from_json, \
        f"edge-labels columns ({width}) != diseases in edge_txt.json ({num_labels_from_json})."

    disease_whitelist = None
    if disease_whitelist_json is not None:
        with open(disease_whitelist_json, "r") as f:
            disease_whitelist = json.load(f)
        if not isinstance(disease_whitelist, list):
            raise ValueError("--disease_whitelist_json must be a JSON array of names")

    # Load canonical map once
    with open(edge_txt_json, "r") as f:
        disease_node_text = json.load(f)  # {"0": "Name0", ...}

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip().lower()

    # Reverse map: canonical name -> column index (int)
    name_to_edge_id = { _norm(v): int(k) for k, v in disease_node_text.items() }

    if disease_whitelist:
        # Keep only names present in whitelist (by normalized equality)
        kept_pairs = []
        missing = []
        for raw_name in disease_whitelist:
            nid = name_to_edge_id.get(_norm(raw_name))
            if nid is not None:
                kept_pairs.append((nid, disease_node_text[str(nid)]))
            else:
                missing.append(raw_name)

        if missing:
            print("[Whitelist] Could not resolve these names in edge_txt.json:", file=sys.stderr)
            for m in missing:
                print("  -", m, file=sys.stderr)

        kept_pairs = sorted(set(kept_pairs), key=lambda x: x[0])
        diseases_in_order = [name for (_idx, name) in kept_pairs]
        disease_name_to_index = {name: idx for (idx, name) in kept_pairs}
    else:
        # Fall back: no whitelist provided -> use full set or your old disease_filter_json logic if desired
        diseases_in_order = all_diseases_in_index_order
        disease_name_to_index = name_to_index_full
    # 4) decide prompt variant
    system_prompt_variant = "cradle" if (dataset and dataset.lower() == "cradle") else "default"
    if no_kg_context:
        system_prompt_variant = "no_kg"

    # 5) build rows
    rows, diseases_in_order = build_rows_from_txt_labels(
        sampled,
        feature_node_text=feature_node_text,
        disease_node_text=None,
        hyperedges=hyperedges,
        edge_labels=edge_labels,
        diseases_in_order=diseases_in_order,
        disease_name_to_index=disease_name_to_index,
        include_absent_in_prompt=include_absent_in_prompt,
        max_absent=max_absent,
        absent_shuffle_seed=absent_shuffle_seed,
        allowed_name_norm_set=allowed_name_norm_set,
        filter_absent_by_allowed=filter_absent_by_allowed,
        system_prompt_variant=system_prompt_variant,
        custom_system_prompt=custom_system_prompt,
        relevance_norm=relevance_norm,   # <-- NEW
        paths_norm=paths_norm,
        include_paths=not no_kg_context,
    )

    if max_eval_rows is not None:
        rows = rows[:max_eval_rows]

    # save snapshots
    with open(os.path.join(out_dir, "sampled_patient_ids.json"), "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "prompts_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "disease_order.json"), "w", encoding="utf-8") as f:
        json.dump(diseases_in_order, f, ensure_ascii=False, indent=2)

    print(f"Unseen patients available: {len(unseen_ids)}")
    print(f"Requested sample: floor({total_patients} * {ratio}) = {k_target}  ->  using k={k}")
    print(f"Sampled distinct patients: {len(set(sampled))}")
    print(f"Built rows (patient × diseases): {len(rows)}")
    print(f"Disease order (first 5): {diseases_in_order[:5]} (total {len(diseases_in_order)})")

    # 6) load model
    print(f"\n==> Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    preferred_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=preferred_dtype,
        device_map="auto",
    )
    if device == "cpu":
        model.to("cpu")

    # 7) run & collect scores
    per_sample_records: List[Dict[str, Any]] = []
    iterator = tqdm(rows, desc="Generate→parse Yes/No") if verbose else rows

    lf = None
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        lf = open(log_file, "a", encoding="utf-8", buffering=1)

    try:
        for rec in iterator:
            messages = build_messages(rec["instruction"], rec["input"])
            pred, p_yes, gen_text = generate_reason_and_answer(
                model, tokenizer, messages,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            parsed_ok = pred is not None
            score = float(p_yes) if parsed_ok else 0.5
            pred_label = int(pred) if parsed_ok else None
            out = {
                "patient_id": rec["patient_id"],
                "disease": rec["disease"],
                "gold": int(rec["label"]),
                "score_p_yes": score,
                "pred_label": pred_label,
                "parsed": bool(parsed_ok),
                "present_features": rec.get("present_features", []),
                "generated_text": gen_text,
            }
            per_sample_records.append(out)
            if lf:
                lf.write(json.dumps(out, ensure_ascii=False) + "\n"); lf.flush()
    finally:
        if lf: lf.close()

    with open(os.path.join(out_dir, "per_sample_generations.jsonl"), "w", encoding="utf-8") as f:
        for r in per_sample_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 8) assemble matrices
    disease2col = {d: i for i, d in enumerate(diseases_in_order)}
    num_labels = len(diseases_in_order)
    by_pid_gold: Dict[int, np.ndarray] = {}
    by_pid_score_prob: Dict[int, np.ndarray] = {}
    by_pid_score_hard: Dict[int, np.ndarray] = {}

    for r in per_sample_records:
        pid = int(r["patient_id"])
        col = disease2col[r["disease"]]
        if pid not in by_pid_gold:
            by_pid_gold[pid] = np.full((num_labels,), np.nan, dtype=float)
            by_pid_score_prob[pid] = np.full((num_labels,), np.nan, dtype=float)
            by_pid_score_hard[pid] = np.full((num_labels,), np.nan, dtype=float)
        by_pid_gold[pid][col] = float(r["gold"])
        by_pid_score_prob[pid][col] = float(r["score_p_yes"])
        if r["pred_label"] is not None:
            by_pid_score_hard[pid][col] = float(r["pred_label"])

    Y_true_list_prob, Y_score_list_prob = [], []
    Y_true_list_hard, Y_score_list_hard = [], []
    skipped_prob = 0
    skipped_hard = 0

    for pid, yt in by_pid_gold.items():
        ys_prob = by_pid_score_prob[pid]
        ys_hard = by_pid_score_hard[pid]

        if not (np.isnan(yt).any() or np.isnan(ys_prob).any()):
            Y_true_list_prob.append(yt.astype(int))
            Y_score_list_prob.append(ys_prob.astype(float))
        else:
            skipped_prob += 1

        if not (np.isnan(yt).any() or np.isnan(ys_hard).any()):
            Y_true_list_hard.append(yt.astype(int))
            Y_score_list_hard.append(ys_hard.astype(float))
        else:
            skipped_hard += 1

    if not Y_true_list_prob:
        raise RuntimeError("No complete patients with probability scores to evaluate.")

    Y_true_prob = np.stack(Y_true_list_prob, axis=0)
    Y_score_prob = np.stack(Y_score_list_prob, axis=0)

    metrics_prob = eval_multilabel_block2_style(
        Y_true_prob, Y_score_prob, threshold=threshold, mode="prob", label_names=diseases_in_order
    )

    metrics_hard = None
    if Y_true_list_hard:
        Y_true_hard = np.stack(Y_true_list_hard, axis=0)
        Y_score_hard = np.stack(Y_score_list_hard, axis=0)
        metrics_hard = eval_multilabel_block2_style(
            Y_true_hard, Y_score_hard, threshold=threshold, mode="hard", label_names=diseases_in_order
        )

    payload = {
        "probability": {
            "metrics": metrics_prob,
            "patients": len(Y_true_list_prob),
            "skipped": skipped_prob,
        },
        "hard_label": {
            "metrics": metrics_hard,
            "patients": len(Y_true_list_hard),
            "skipped": skipped_hard,
        },
    }

    per_label_prob = metrics_prob.get("per_label") if metrics_prob else None
    if per_label_prob:
        write_per_disease_tables(per_label_prob, out_dir, "per_disease_metrics_prob")
    if metrics_hard and metrics_hard.get("per_label"):
        write_per_disease_tables(metrics_hard["per_label"], out_dir, "per_disease_metrics_hard")

    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n== Multilabel per-patient evaluation complete ==")
    print("Probability view (macro):"); print(json.dumps(metrics_prob["macro"], indent=2))
    print("Probability view (micro):"); print(json.dumps(metrics_prob["micro"], indent=2))
    if metrics_hard:
        print("Hard-label view (macro):"); print(json.dumps(metrics_hard["macro"], indent=2))
        print("Hard-label view (micro):"); print(json.dumps(metrics_hard["micro"], indent=2))
    if skipped_prob or skipped_hard:
        print(f"Skipped patients -> prob:{skipped_prob}, hard:{skipped_hard}")

    return payload

# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Sample unseen patients → build prompts → evaluate multilabel per patient (macro over labels, micro flattened).")
    # Sampling inputs
    ap.add_argument("--unseen_ids_json", required=True, help="Path to unseen_patient_ids.json (list of ids)")
    ap.add_argument("--total_patients", type=int, required=True, help="Total distinct patients in the corpus (for ratio * total).")
    ap.add_argument("--ratio", type=float, default=0.01, help="Sampling ratio over total patients (e.g., 0.2)")
    ap.add_argument("--seed", type=int, default=42)

    # Data sources
    ap.add_argument("--node_text_json", required=True, help="JSON map: node_id(str) -> text (features)")
    ap.add_argument("--hyperedges_txt", required=True, help="CSV lines of node ids per patient index")
    ap.add_argument("--edge_labels_txt", required=True,
                    help="TXT with 0/1 per disease column per patient (same width as edge_txt_json).")
    ap.add_argument("--edge_txt_json", required=True,
                    help="JSON mapping disease_id(str) -> disease name.")
    ap.add_argument("--disease_whitelist_json")

    # Model & eval
    ap.add_argument("--model_dir", required=True, help="HF path to model (e.g., Llama-3.1-8B-Instruct).")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs.")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_eval_rows", type=int, help="Evaluate only first N (patient × disease) rows.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_file", type=str)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)

    # Prompt formatting opts
    ap.add_argument("--max_absent", type=int, default=50,
                    help="Maximum number of absent features to include per patient after shuffling (COT-style).")
    ap.add_argument("--absent_shuffle_seed", type=int, default=123,
                    help="Seed used to shuffle absent features before truncation (COT-style).")
    ap.add_argument("--include_absent_in_prompt", action="store_true",
                   help="If set, append 'Patient features absent(truncated): ...' to the prompt.")

    # NEW: name-level gating
    ap.add_argument(
        "--mimic_name_json",
        help="Path to canonical MIMIC names JSON (use mimic/node_txt.json). "
             "If provided (e.g., for cradle), ONLY present features whose NAME "
             "appears here (case/space-normalized) will be included in the prompt.",
    )
    ap.add_argument(
        "--filter_absent_by_allowed",
        action="store_true",
        help="Also filter the 'absent' list by the same MIMIC name set (off by default)."
    )
    ap.add_argument("--dataset", choices=["mimic", "mimic_trunc", "cradle"], help="Dataset name to select a built-in system prompt variant.")
    ap.add_argument("--system_prompt_file", help="Path to a text file to fully override the system prompt (supports {disease} interpolation).")
    ap.add_argument("--relevance_txt", help="JSON-ish relevance file (same format used in Code B).")
    ap.add_argument("--paths_json", help="filtered_path_mappings.json (disease -> list of reasoning paths).")
    ap.add_argument("--no_kg_context", action="store_true",
                    help="Disable KG reasoning paths in prompts (omit 'Reasoning Paths' block).")

    args = ap.parse_args()

    evaluate_multilabel_with_sampling(
        model_dir=args.model_dir,
        unseen_ids_json=args.unseen_ids_json,
        node_text_json=args.node_text_json,
        edge_txt_json=args.edge_txt_json,
        hyperedges_txt=args.hyperedges_txt,
        edge_labels_txt=args.edge_labels_txt,
        out_dir=args.out_dir,
        total_patients=args.total_patients,
        ratio=args.ratio,
        seed=args.seed,
        disease_whitelist_json=args.disease_whitelist_json,
        device=args.device,
        max_eval_rows=args.max_eval_rows,
        verbose=args.verbose,
        log_file=args.log_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        threshold=args.threshold,
        include_absent_in_prompt=args.include_absent_in_prompt,
        max_absent=args.max_absent,
        absent_shuffle_seed=args.absent_shuffle_seed,
        mimic_name_json=args.mimic_name_json,
        filter_absent_by_allowed=args.filter_absent_by_allowed,
        dataset=args.dataset,
        system_prompt_file=args.system_prompt_file,
        relevance_txt=args.relevance_txt,   # <-- pass it
        paths_json=args.paths_json,
        no_kg_context=args.no_kg_context,
    )
 
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    main()
