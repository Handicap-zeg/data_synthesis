from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

from lib.common import load_config, load_df, normalize_text, save_df

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_text(text or "")))


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / max(1, self.n_docs)) if self.n_docs else 1.0
        if self.avgdl <= 0:
            self.avgdl = 1.0

        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)

        for i, toks in enumerate(docs):
            c = Counter(toks)
            self.tf.append(c)
            for t in c.keys():
                self.df[t] += 1
                self.postings[t].append(i)

        self.idf: dict[str, float] = {}
        n = max(1, self.n_docs)
        for t, dft in self.df.items():
            self.idf[t] = float(math.log(1.0 + (n - dft + 0.5) / (dft + 0.5)))

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        q_count = Counter(query_tokens)
        for t, qtf in q_count.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings.get(t, []):
                tf = self.tf[i].get(t, 0)
                if tf <= 0:
                    continue
                dl = float(self.doc_len[i])
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(1e-8, self.avgdl))
                val = idf * (tf * (self.k1 + 1.0)) / max(1e-8, denom)
                scores[i] += float(qtf) * float(val)
        return scores


def _build_subnode_bm25_sim(subnodes: pd.DataFrame) -> dict[tuple[str, str], float]:
    ids: list[str] = []
    docs: list[list[str]] = []
    for r in subnodes.itertuples(index=False):
        sid = str(r.subnode_id)
        text = str(getattr(r, 'msc_full', '')) + ' ' + str(getattr(r, 'concept_cluster', ''))
        toks = TOKEN_RE.findall(normalize_text(text))
        if not toks:
            toks = ['math']
        ids.append(sid)
        docs.append(toks)

    if not ids:
        return {}

    bm25 = _BM25(docs=docs, k1=1.5, b=0.75)
    all_scores: list[list[float]] = []
    self_scores: list[float] = []
    for i, q in enumerate(docs):
        sc = bm25.score(q)
        all_scores.append(sc)
        self_scores.append(max(1e-8, float(sc[i])))

    out: dict[tuple[str, str], float] = {}
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = float(all_scores[i][j]) / self_scores[i]
            s2 = float(all_scores[j][i]) / self_scores[j]
            sim = 0.5 * (s1 + s2)
            if sim < 0.0:
                sim = 0.0
            if sim > 1.0:
                sim = 1.0
            key = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
            out[key] = float(sim)
    return out


def _prepare_subnodes(subnodes: pd.DataFrame) -> pd.DataFrame:
    out = subnodes.copy()
    for c in ["subnode_id", "parent_node_id", "msc_full", "domain", "concept_cluster", "freq"]:
        if c not in out.columns:
            out[c] = "" if c != "freq" else 0
    out["subnode_id"] = out["subnode_id"].astype(str)
    out["parent_node_id"] = out["parent_node_id"].astype(str)
    out["msc_full"] = out["msc_full"].astype(str)
    out["domain"] = out["domain"].astype(str)
    out["concept_cluster"] = out["concept_cluster"].astype(str)
    out["freq"] = pd.to_numeric(out["freq"], errors="coerce").fillna(0.0).astype(float)
    return out


def _build_subnode_qsets(seeds_sub: pd.DataFrame) -> dict[str, set[str]]:
    seeds = seeds_sub[["subnode_id", "qid"]].copy()
    seeds["subnode_id"] = seeds["subnode_id"].astype(str)
    seeds["qid"] = seeds["qid"].astype(str)
    grp = seeds.groupby("subnode_id")["qid"].apply(lambda s: set(s.tolist()))
    return {str(k): set(v) for k, v in grp.to_dict().items()}


def _level_to_y(level: object) -> float | None:
    s = str(level).strip().lower() if level is not None else str()
    if s:
        for ch in s:
            if ch in set('12345'):
                lv = int(ch)
                return (lv - 1) / 4.0
    try:
        lv = int(level)  # type: ignore[arg-type]
        if 1 <= lv <= 5:
            return (lv - 1) / 4.0
    except Exception:
        return None
    return None


def _build_qid_y(graph_dir: Path) -> dict[str, float]:
    seeds_main = load_df(str(graph_dir / 'seeds_index.parquet')).copy()
    if 'qid' not in seeds_main.columns or 'level' not in seeds_main.columns:
        return {}

    seeds_main['qid'] = seeds_main['qid'].astype(str)
    y_eff = pd.to_numeric(seeds_main['level'].map(_level_to_y), errors='coerce')
    out = pd.DataFrame({'qid': seeds_main['qid'], 'y': y_eff}).dropna(subset=['y'])
    out = out.drop_duplicates(subset=['qid'], keep='first')
    return dict(zip(out['qid'].tolist(), out['y'].astype(float).tolist()))


def _build_subnode_difficulty(seeds_sub: pd.DataFrame, qid_y: dict[str, float]) -> dict[str, float]:
    if not qid_y:
        return {}
    s = seeds_sub[["subnode_id", "qid"]].copy()
    s["subnode_id"] = s["subnode_id"].astype(str)
    s["qid"] = s["qid"].astype(str)
    s["y"] = s["qid"].map(qid_y)
    s = s.dropna(subset=["y"])
    if s.empty:
        return {}
    g = s.groupby("subnode_id")["y"].mean()
    return {str(k): float(v) for k, v in g.to_dict().items()}


def _enforce_dag_per_parent(df_edges: pd.DataFrame) -> pd.DataFrame:
    if df_edges.empty:
        return df_edges.copy()

    out_parts: list[pd.DataFrame] = []
    for parent, sub in df_edges.groupby("parent_node_id", dropna=False):
        g = nx.DiGraph()
        for r in sub.itertuples(index=False):
            g.add_edge(str(r.src), str(r.dst), conf=float(r.conf))

        while True:
            try:
                cyc = nx.find_cycle(g, orientation="original")
            except nx.NetworkXNoCycle:
                break
            cyc_edges = [(u, v) for (u, v, _) in cyc]
            e_min = min(cyc_edges, key=lambda e: float(g[e[0]][e[1]].get("conf", 0.0)))
            g.remove_edge(*e_min)

        rows = []
        for u, v, d in g.edges(data=True):
            rows.append({"parent_node_id": str(parent), "src": u, "dst": v, "conf": float(d.get("conf", 0.0))})
        out_parts.append(pd.DataFrame(rows))

    if not out_parts:
        return pd.DataFrame(columns=["parent_node_id", "src", "dst", "conf"])

    merged = pd.concat(out_parts, ignore_index=True)
    if set(["src", "dst", "parent_node_id"]).issubset(df_edges.columns):
        meta = df_edges.drop_duplicates(subset=["parent_node_id", "src", "dst"])
        merged = merged.merge(meta, on=["parent_node_id", "src", "dst"], how="left", suffixes=("", "_m"))
        if "conf_m" in merged.columns:
            merged["conf"] = merged["conf"].fillna(merged["conf_m"])
            merged = merged.drop(columns=["conf_m"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph_dir = Path(cfg["paths"]["graph_dir"])

    pre_cfg = cfg.get("sub_pre_edge", {})
    min_cover = float(pre_cfg.get("min_cover", 0.35))
    min_sim = float(pre_cfg.get("min_sim_concept", 0.20))
    min_pair_cooccur = int(pre_cfg.get("min_pair_cooccur", 1))
    min_difficulty_gap = float(pre_cfg.get("min_difficulty_gap", 0.05))
    min_logfreq_gap = float(pre_cfg.get("min_logfreq_gap", 0.50))
    max_out_per_node = int(pre_cfg.get("max_out_per_node", 3))
    reverse_keep_ratio = float(pre_cfg.get("reverse_keep_ratio", 1.15))
    enforce_dag = bool(pre_cfg.get("enforce_dag", True))

    subnodes = _prepare_subnodes(load_df(str(graph_dir / "subnodes.parquet")))
    seeds_sub = load_df(str(graph_dir / "seeds_sub_index.parquet"))

    qsets = _build_subnode_qsets(seeds_sub)
    qid_y = _build_qid_y(graph_dir)
    d_map = _build_subnode_difficulty(seeds_sub, qid_y)

    bm25_sim_map = _build_subnode_bm25_sim(subnodes)

    by_parent: dict[str, list[str]] = defaultdict(list)
    for r in subnodes.itertuples(index=False):
        by_parent[str(r.parent_node_id)].append(str(r.subnode_id))

    rows: list[dict] = []

    def _eval_dir(src: str, dst: str) -> dict:
        qa = qsets.get(src, set())
        qb = qsets.get(dst, set())
        co = len(qa & qb)

        key = (src, dst) if src < dst else (dst, src)
        sim = float(bm25_sim_map.get(key, 0.0))

        ds = d_map.get(src)
        dd = d_map.get(dst)
        dgap = None if (ds is None or dd is None) else float(dd - ds)

        pass_first = (co >= min_pair_cooccur) or (sim >= min_sim)
        pass_diff = bool(dgap is not None and dgap >= min_difficulty_gap)
        ok = bool(pass_first and pass_diff)

        co_score = min(1.0, float(co) / 3.0)
        d_score = 0.0 if dgap is None else min(1.0, max(0.0, dgap) / max(1e-8, min_difficulty_gap))
        conf = 0.35 * co_score + 0.35 * sim + 0.30 * d_score

        return {
            "ok": ok,
            "pair_cooccur": int(co),
            "sim_concept": float(sim),
            "difficulty_gap": (float(dgap) if dgap is not None else None),
            "conf": float(conf),
        }


    for parent, ids in by_parent.items():
        if len(ids) < 2:
            continue
        for a, b in combinations(sorted(ids), 2):
            ab = _eval_dir(a, b)
            ba = _eval_dir(b, a)

            pick = None
            if ab["ok"] and not ba["ok"]:
                pick = (a, b, ab)
            elif ba["ok"] and not ab["ok"]:
                pick = (b, a, ba)
            elif ab["ok"] and ba["ok"]:
                c1 = float(ab["conf"])
                c2 = float(ba["conf"])
                hi = max(c1, c2)
                lo = max(1e-8, min(c1, c2))
                if hi / lo >= max(1.0, reverse_keep_ratio):
                    pick = (a, b, ab) if c1 >= c2 else (b, a, ba)

            if pick is None:
                continue

            src, dst, feat = pick
            rows.append(
                {
                    "parent_node_id": str(parent),
                    "src": str(src),
                    "dst": str(dst),
                    "conf": float(feat["conf"]),
                    "pair_cooccur": int(feat["pair_cooccur"]),
                    "sim_concept": float(feat["sim_concept"]),
                    "difficulty_gap": feat["difficulty_gap"],
                }
            )

    if not rows:
        out = pd.DataFrame(columns=["parent_node_id", "src", "dst", "conf", "pair_cooccur", "sim_concept", "difficulty_gap"])
        save_df(out, str(graph_dir / "edges_pre_sub.parquet"))
        print("[WARN] no sub_pre edges generated")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["conf", "sim_concept", "pair_cooccur"], ascending=False)
    if max_out_per_node > 0:
        df = df.groupby("src", as_index=False, group_keys=False).head(max_out_per_node)

    df = df.drop_duplicates(subset=["parent_node_id", "src", "dst"], keep="first").reset_index(drop=True)

    if enforce_dag and not df.empty:
        before = len(df)
        df = _enforce_dag_per_parent(df)
        print(f"[INFO] sub_pre after DAG enforce: {len(df)} (removed {before - len(df)})")

    df = df.sort_values("conf", ascending=False).reset_index(drop=True)
    save_df(df, str(graph_dir / "edges_pre_sub.parquet"))
    print(f"[OK] sub_pre edges: {len(df)}")


if __name__ == "__main__":
    main()
