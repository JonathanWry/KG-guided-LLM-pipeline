#!/usr/bin/env python3
# Purpose: Generate per-disease KG reasoning paths from relevance entities and export path mappings JSON.

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_relevance_jsonish(path: Path) -> Dict:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw.startswith("{"):
        raw = "{\n" + raw + "\n}"
    raw = re.sub(r"}\s*(?=\"[^\"]+\"\s*:)", "},\n", raw)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return json.loads(raw)


def build_graph(csv_path: Path) -> Tuple[nx.Graph, pd.DataFrame, List[str], Dict[str, str]]:
    df = pd.read_csv(csv_path, dtype=str)
    g = nx.Graph()
    for _, row in df.iterrows():
        g.add_edge(row["x_name"], row["y_name"])
    nodes = list(g.nodes)
    lower_to_node = {n.lower(): n for n in nodes}
    return g, df, nodes, lower_to_node


def nodes_starting_with(nodes: List[str], prefix: str) -> List[str]:
    p = prefix.lower()
    out = []
    for n in nodes:
        nl = n.lower()
        if nl.startswith(p + " ") or nl.startswith(p + ","):
            out.append(n)
    return out


@torch.no_grad()
def encode_texts(tokenizer, model, texts: List[str], device: str, batch_size: int = 64):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding node batches", unit="batch"):
        batch = texts[i : i + batch_size]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        emb = model(**tok).last_hidden_state.mean(dim=1)
        emb = F.normalize(emb, dim=1)
        out.append(emb.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_one(tokenizer, model, text: str, device: str):
    tok = tokenizer([text], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    emb = model(**tok).last_hidden_state.mean(dim=1)
    return F.normalize(emb, dim=1)


def edge_relation(df: pd.DataFrame, src: str, tgt: str) -> str:
    rows = df[
        ((df["x_name"] == src) & (df["y_name"] == tgt))
        | ((df["x_name"] == tgt) & (df["y_name"] == src))
    ]
    if not rows.empty and "relation" in rows.columns:
        return str(rows.iloc[0]["relation"])
    return "unknown_relation"


def format_path_with_relations(df: pd.DataFrame, path_nodes: List[str]) -> str:
    if len(path_nodes) <= 1:
        return " -> ".join(path_nodes)

    chunks = []
    for i in range(len(path_nodes) - 1):
        src, tgt = path_nodes[i], path_nodes[i + 1]
        rel = edge_relation(df, src, tgt)
        if i == 0:
            chunks.append(f"{src} (node) -> {rel} (edge) -> {tgt} (node)")
        else:
            chunks.append(f"{rel} (edge) -> {tgt} (node)")
    return " -> ".join([chunks[0]] + chunks[1:])


def safe_name(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


def main() -> None:
    root = repo_root()

    ap = argparse.ArgumentParser(description="Generate disease path text files + raw path mapping JSON from relevance entities")
    ap.add_argument("--relevance_txt", default=str(root / "data" / "relevence.txt"))
    ap.add_argument("--graph_csv", default=str(root / "data" / "prune_kg.csv"))
    ap.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--out_dir", default=str(root / "data" / "disease_paths"))
    ap.add_argument("--out_mapping_json", default=str(root / "data" / "path_mappings_raw.json"))
    args = ap.parse_args()

    relevance_path = Path(args.relevance_txt)
    graph_csv = Path(args.graph_csv)
    out_dir = Path(args.out_dir)
    out_mapping_json = Path(args.out_mapping_json)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_mapping_json.parent.mkdir(parents=True, exist_ok=True)

    print("Loading relevance...")
    relevance = load_relevance_jsonish(relevance_path)

    print("Loading graph...")
    graph, edge_df, graph_nodes, lower_to_node = build_graph(graph_csv)
    if graph.number_of_nodes() == 0:
        raise RuntimeError("Graph is empty. Check prune_kg.csv x_name/y_name columns.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    model = AutoModel.from_pretrained(args.encoder_model).to(device).eval()

    print("Precomputing graph-node embeddings...")
    node_embs = encode_texts(tokenizer, model, graph_nodes, device).to(device)

    def resolve_node(name: str):
        if name in graph:
            return name, "exact", []
        lc = name.lower()
        if lc in lower_to_node:
            return lower_to_node[lc], "exact(case-insensitive)", []

        q = encode_one(tokenizer, model, name, device)
        sims = torch.matmul(node_embs, q[0])
        k = min(args.top_k, sims.numel())
        top_scores, top_idx = torch.topk(sims, k=k)
        candidates = []
        for i in range(k):
            score = float(top_scores[i])
            if score >= args.tau:
                candidates.append((graph_nodes[int(top_idx[i])], score))

        if candidates:
            return candidates[0][0], f"similar(score={candidates[0][1]:.3f})", candidates
        return None, "not_found", []

    def expand_source_candidates(term: str):
        if term in graph:
            return [(term, "exact")]
        lc = term.lower()
        if lc in lower_to_node:
            return [(lower_to_node[lc], "exact(case-insensitive)")]

        out = []
        q = encode_one(tokenizer, model, term, device)
        sims = torch.matmul(node_embs, q[0])
        k = min(args.top_k, sims.numel())
        top_scores, top_idx = torch.topk(sims, k=k)
        for i in range(k):
            score = float(top_scores[i])
            if score >= args.tau:
                out.append((graph_nodes[int(top_idx[i])], f"similar(score={score:.3f})"))

        for n in nodes_starting_with(graph_nodes, term):
            out.append((n, "prefix_match"))

        uniq = []
        seen = set()
        for node, why in out:
            if node not in seen:
                seen.add(node)
                uniq.append((node, why))
        return uniq

    mapping: Dict[str, List[str]] = {}
    generated = 0

    for disease, block in tqdm(relevance.items(), desc="Diseases", unit="disease"):
        target, target_status, _ = resolve_node(disease)
        if target is None:
            continue

        lines = [f"Disease: {disease}", f"Resolved target: {target} ({target_status})", ""]
        disease_paths: List[str] = []

        keys = sorted(
            block.keys(),
            key=lambda k: int(re.findall(r"\d+", k)[0]) if re.findall(r"\d+", k) else 10**9,
        )

        for k in keys:
            ent = block[k]
            src_term = str(ent.get("name", "")).strip()
            reason = str(ent.get("reason", "")).strip()
            if not src_term:
                continue

            lines.append(f"Entity: {src_term}")
            if reason:
                lines.append(f"Reason: {reason}")

            cands = expand_source_candidates(src_term)
            if not cands:
                lines.append("Path: no candidates found")
                lines.append("")
                continue

            for cand, why in cands:
                try:
                    path = nx.shortest_path(graph, source=cand, target=target)
                except nx.NetworkXNoPath:
                    lines.append(f"Path ({why}): no path between {cand} and {target}")
                    continue

                pretty = format_path_with_relations(edge_df, path)
                lines.append(f"Path ({why}): {pretty}")
                disease_paths.append(pretty)

            lines.append("")

        unique_paths = list(dict.fromkeys(disease_paths))
        mapping[disease] = unique_paths

        out_txt = out_dir / f"{safe_name(disease)}_paths.txt"
        out_txt.write_text("\n".join(lines), encoding="utf-8")
        generated += 1

    out_mapping_json.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated per-disease path files: {generated} in {out_dir}")
    print(f"Wrote raw mapping JSON: {out_mapping_json}")


if __name__ == "__main__":
    main()
