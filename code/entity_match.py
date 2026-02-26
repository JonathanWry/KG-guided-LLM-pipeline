#!/usr/bin/env python3
# Purpose: Match MIMIC feature names to KG entities using BioBERT similarity and export top-k candidates.

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_entities(entities_file: Path) -> Tuple[Dict[str, int], List[str]]:
    entities_dict: Dict[str, int] = {}
    entities_list: List[str] = []

    with entities_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, idx = line.rsplit(":", 1)
            name = name.strip()
            try:
                idx_i = int(float(idx.strip()))
            except Exception:
                continue
            entities_dict[name] = idx_i
            entities_list.append(name)

    return entities_dict, entities_list


def query_variants(q: str) -> List[str]:
    vals = [q]
    if "," in q:
        vals.append(q.split(",", 1)[0].strip())
    if " and " in q.lower():
        parts = q.split("and", 1)
        vals.append(parts[0].strip())

    out = []
    seen = set()
    for x in vals:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


@torch.no_grad()
def encode_texts(tokenizer, model, texts: List[str], device: str, batch_size: int = 64):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch = texts[i : i + batch_size]
        toks = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        out = model(**toks).last_hidden_state[:, 0, :]
        out = F.normalize(out, dim=1)
        all_embs.append(out.cpu())
    return torch.cat(all_embs, dim=0)


def load_or_build_embeddings(cache_file: Path, tokenizer, model, texts: List[str], device: str):
    if cache_file.exists():
        return torch.load(cache_file, map_location="cpu")
    embs = encode_texts(tokenizer, model, texts, device)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embs, cache_file)
    return embs


def main() -> None:
    root = repo_root()

    ap = argparse.ArgumentParser(description="BioBERT top-k entity matching for MIMIC node_text entries")
    ap.add_argument("--query_json", default=str(root / "data" / "mimic" / "node_text.json"))
    ap.add_argument("--entities_file", default=str(root / "data" / "Entity_Matching" / "prune_filtered_entities.txt"))
    ap.add_argument("--out_file", default=str(root / "data" / "Entity_Matching" / "existing_nodes_topN.txt"))
    ap.add_argument("--entity_emb_cache", default=str(root / "data" / "Entity_Matching" / "entity_embeddings.pt"))
    ap.add_argument("--model_name_or_path", default="dmis-lab/biobert-base-cased-v1.2")
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    query_json = Path(args.query_json)
    entities_file = Path(args.entities_file)
    out_file = Path(args.out_file)
    emb_cache = Path(args.entity_emb_cache)

    with query_json.open("r", encoding="utf-8") as f:
        query_names = list(json.load(f).values())

    entities_dict, entities_list = load_entities(entities_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(device).eval()

    entity_embs = load_or_build_embeddings(emb_cache, tokenizer, model, entities_list, device).to(device)
    entity_embs = F.normalize(entity_embs, dim=1)

    exact_lc = {k.lower(): k for k in entities_list}
    output_rows = []

    for q in tqdm(query_names, desc="Matching query names", unit="query"):
        triples = []
        rank_counter = 1

        for qv in query_variants(str(q)):
            exact_name = exact_lc.get(qv.lower())
            if exact_name:
                triples.append(
                    {
                        "rank": rank_counter,
                        "entity_name": exact_name,
                        "similarity": 1.0,
                        "entity_index": entities_dict[exact_name],
                        "query_variant": qv,
                    }
                )
                rank_counter += 1
                continue

            q_emb = encode_texts(tokenizer, model, [qv], device, batch_size=1).to(device)
            sims = torch.matmul(entity_embs, q_emb[0])
            k = min(args.top_k, sims.numel())
            top_scores, top_idx = torch.topk(sims, k=k)

            for i in range(k):
                ent_name = entities_list[int(top_idx[i])]
                triples.append(
                    {
                        "rank": rank_counter,
                        "entity_name": ent_name,
                        "similarity": round(float(top_scores[i]), 6),
                        "entity_index": entities_dict[ent_name],
                        "query_variant": qv,
                    }
                )
                rank_counter += 1

        output_rows.append({"query_name": q, "query_triples": triples})

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_rows)} query matches to {out_file}")


if __name__ == "__main__":
    main()
