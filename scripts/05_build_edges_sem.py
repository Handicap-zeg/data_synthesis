from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
import threading

import pandas as pd

from lib.common import build_llm_client, load_config, load_df, normalize_text, save_df

TOKEN_RE = re.compile(r"[a-z0-9]+")

SYSTEM_PROMPT_MAIN_SEM = (
    "You judge whether two math areas are suitable to combine into one contest/elementary math problem. "
    "Be conservative. Return strict JSON only."
)


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_text(text or "")))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    den = len(a | b)
    if den <= 0:
        return 0.0
    return float(len(a & b) / den)



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
                s = idf * (tf * (self.k1 + 1.0)) / max(1e-8, denom)
                scores[i] += float(qtf) * float(s)
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

def _prepare_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    for c in ["node_id", "msc_full", "msc_desc", "concept", "domain", "freq"]:
        if c not in out.columns:
            out[c] = "" if c != "freq" else 0
    out["node_id"] = out["node_id"].astype(str)
    out["msc_full"] = out["msc_full"].astype(str)
    out["msc_desc"] = out["msc_desc"].astype(str)
    out["concept"] = out["concept"].astype(str)
    out["domain"] = out["domain"].astype(str)
    out["freq"] = pd.to_numeric(out["freq"], errors="coerce").fillna(0.0).astype(float)
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
    s = seeds_sub[["subnode_id", "qid"]].copy()
    s["subnode_id"] = s["subnode_id"].astype(str)
    s["qid"] = s["qid"].astype(str)
    grp = s.groupby("subnode_id")["qid"].apply(lambda x: set(x.tolist()))
    return {str(k): set(v) for k, v in grp.to_dict().items()}


def _level_to_y(level: object) -> float | None:
    t = str(level).strip().lower() if level is not None else str()
    for ch in t:
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


def _build_qid_meta(graph_dir: Path) -> tuple[dict[str, float], dict[str, str]]:
    seeds_main = load_df(str(graph_dir / 'seeds_index.parquet')).copy()
    qid_y: dict[str, float] = {}
    qid_subject: dict[str, str] = {}

    if 'qid' in seeds_main.columns:
        seeds_main['qid'] = seeds_main['qid'].astype(str)

    if 'qid' in seeds_main.columns and 'level' in seeds_main.columns:
        y_eff = pd.to_numeric(seeds_main['level'].map(_level_to_y), errors='coerce')
        t = pd.DataFrame({'qid': seeds_main['qid'], 'y': y_eff})
        t = t.dropna(subset=['y']).drop_duplicates(subset=['qid'], keep='first')
        qid_y = dict(zip(t['qid'].astype(str).tolist(), t['y'].astype(float).tolist()))

    if 'subject' in seeds_main.columns and 'qid' in seeds_main.columns:
        t2 = seeds_main[['qid', 'subject']].copy()
        t2['subject'] = t2['subject'].astype(str)
        t2 = t2.drop_duplicates(subset=['qid'], keep='first')
        qid_subject = dict(zip(t2['qid'].astype(str).tolist(), t2['subject'].tolist()))
    return qid_y, qid_subject



def _build_subnode_subject_sets(seeds_sub: pd.DataFrame, qid_subject: dict[str, str]) -> dict[str, set[str]]:
    if not qid_subject:
        return {}
    s = seeds_sub[["subnode_id", "qid"]].copy()
    s["subnode_id"] = s["subnode_id"].astype(str)
    s["qid"] = s["qid"].astype(str)
    s["subject"] = s["qid"].map(qid_subject)
    s = s.dropna(subset=["subject"])
    if s.empty:
        return {}
    g = s.groupby("subnode_id")["subject"].apply(lambda x: set(map(str, x.tolist())))
    return {str(k): set(v) for k, v in g.to_dict().items()}



def _build_main_sem_prompt(a: dict, b: dict, a_examples: list[str], b_examples: list[str]) -> str:
    ax = " | ".join(a_examples[:3]) if a_examples else ""
    bx = " | ".join(b_examples[:3]) if b_examples else ""
    return f"""
Decide whether two math areas are suitable to combine into one contest/elementary problem.

Return JSON:
{{
  "allow": "yes|no",
  "confidence": 0.0,
  "reason": "short"
}}

Area A:
- code: {a.get('msc_full','')}
- domain: {a.get('domain','')}
- desc: {a.get('msc_desc','')}
- concept: {a.get('concept','')}
- example sub-concepts: {ax}

Area B:
- code: {b.get('msc_full','')}
- domain: {b.get('domain','')}
- desc: {b.get('msc_desc','')}
- concept: {b.get('concept','')}
- example sub-concepts: {bx}

Rules:
- Prefer pairs that can naturally form one coherent high-school/competition-style problem.
- Reject pairs that are disconnected or too forced.
- Be conservative.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph_dir = Path(cfg["paths"]["graph_dir"])

    sub_sem_cfg = cfg.get("sub_sem_edge", {})
    sub_sem_top_n = int(sub_sem_cfg.get("top_k_by_cooccur", 160))

    main_sem_cfg = cfg.get("main_sem_edge", {})
    bridge_min_score = float(main_sem_cfg.get("bridge_min_score", 0.35))
    bridge_min_count = int(main_sem_cfg.get("bridge_min_count", 2))
    bridge_topk_mean_k = int(main_sem_cfg.get("bridge_topk_mean_k", 3))
    bridge_min_topk_mean = float(main_sem_cfg.get("bridge_min_topk_mean", 0.38))
    max_degree_per_main = int(main_sem_cfg.get("max_degree_per_main", 3))

    enable_llm_gate = bool(main_sem_cfg.get("enable_llm_gate", True))
    llm_votes = int(main_sem_cfg.get("llm_votes", 1))
    llm_temperature = float(main_sem_cfg.get("llm_temperature", 0.0))
    llm_min_conf = float(main_sem_cfg.get("llm_min_conf", 0.55))
    llm_workers = int(main_sem_cfg.get("llm_workers", 8))

    llm_src = cfg.get("pre_edge", {})
    llm_base_url = str(main_sem_cfg.get("base_url", llm_src.get("base_url", "")))
    llm_model = str(main_sem_cfg.get("model", llm_src.get("model", "")))
    llm_api_key_env = main_sem_cfg.get("api_key_env", llm_src.get("api_key_env"))
    llm_api_key = main_sem_cfg.get("api_key", llm_src.get("api_key"))

    nodes = _prepare_nodes(load_df(str(graph_dir / "nodes.parquet")))
    subnodes = _prepare_subnodes(load_df(str(graph_dir / "subnodes.parquet")))
    seeds_sub = load_df(str(graph_dir / "seeds_sub_index.parquet"))

    qsets = _build_subnode_qsets(seeds_sub)
    _, qid_subject = _build_qid_meta(graph_dir)
    subj_sets = _build_subnode_subject_sets(seeds_sub, qid_subject)

    bm25_sim_map = _build_subnode_bm25_sim(subnodes)

    pre_path = graph_dir / "edges_pre_sub.parquet"
    if pre_path.exists():
        pre_sub = load_df(str(pre_path)).copy()
    else:
        pre_sub = pd.DataFrame(columns=["src", "dst"])
    pre_undirected: set[tuple[str, str]] = set()
    if not pre_sub.empty:
        pre_sub["src"] = pre_sub["src"].astype(str)
        pre_sub["dst"] = pre_sub["dst"].astype(str)
        for r in pre_sub.itertuples(index=False):
            a = str(r.src)
            b = str(r.dst)
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            pre_undirected.add(key)

    by_parent: dict[str, list[str]] = defaultdict(list)
    for r in subnodes.itertuples(index=False):
        by_parent[str(r.parent_node_id)].append(str(r.subnode_id))

    sub_sem_candidates: list[dict] = []
    for parent, ids in by_parent.items():
        if len(ids) < 2:
            continue
        for a, b in combinations(sorted(ids), 2):
            key = (a, b) if a < b else (b, a)
            if key in pre_undirected:
                continue
            sim = float(bm25_sim_map.get(key, 0.0))
            pair_cooccur = int(len(qsets.get(key[0], set()) & qsets.get(key[1], set())))
            sub_sem_candidates.append(
                {
                    "parent_node_id": str(parent),
                    "src": key[0],
                    "dst": key[1],
                    "pair_cooccur": pair_cooccur,
                    "sim_concept": float(sim),
                    "conf": float(pair_cooccur),
                }
            )

    sub_sem_candidates = sorted(
        sub_sem_candidates,
        key=lambda r: (int(r["pair_cooccur"]), float(r.get("sim_concept", 0.0))),
        reverse=True,
    )

    if sub_sem_candidates:
        keep_n = max(1, min(int(sub_sem_top_n), len(sub_sem_candidates)))
        sub_sem_rows = sub_sem_candidates[:keep_n]
    else:
        sub_sem_rows = []

    sub_sem_df = pd.DataFrame(sub_sem_rows)
    if sub_sem_df.empty:
        sub_sem_df = pd.DataFrame(columns=["parent_node_id", "src", "dst", "pair_cooccur", "sim_concept", "conf"] )
    save_df(sub_sem_df, str(graph_dir / "edges_sem_sub.parquet"))
    print(f"[INFO] sub sem filter: no-pre only, top_n={sub_sem_top_n}, rank=pair_cooccur")
    print(f"[OK] sub sem edges: {len(sub_sem_df)}")

    node_info = {
        str(r.node_id): {
            "node_id": str(r.node_id),
            "msc_full": str(r.msc_full),
            "msc_desc": str(r.msc_desc),
            "concept": str(r.concept),
            "domain": str(r.domain),
            "freq": float(r.freq),
        }
        for r in nodes.itertuples(index=False)
    }

    top_sub_text: dict[str, list[str]] = defaultdict(list)
    sub_sorted = subnodes.sort_values(["parent_node_id", "freq"], ascending=[True, False])
    for r in sub_sorted.itertuples(index=False):
        p = str(r.parent_node_id)
        if len(top_sub_text[p]) >= 3:
            continue
        txt = str(r.concept_cluster).strip()
        if txt:
            top_sub_text[p].append(txt)

    main_candidates: list[dict] = []
    main_ids = sorted(by_parent.keys())
    for ma, mb in combinations(main_ids, 2):
        a_ids = by_parent.get(ma, [])
        b_ids = by_parent.get(mb, [])
        if not a_ids or not b_ids:
            continue

        scores: list[tuple[float, str, str, float, float]] = []
        for sa in a_ids:
            for sb in b_ids:
                sim = float(bm25_sim_map.get((sa, sb) if sa < sb else (sb, sa), 0.0))
                subj_aff = _jaccard(subj_sets.get(sa, set()), subj_sets.get(sb, set()))
                bridge = 0.70 * sim + 0.30 * subj_aff
                if bridge <= 0:
                    continue
                scores.append((float(bridge), str(sa), str(sb), float(sim), float(subj_aff)))

        if not scores:
            continue

        scores.sort(key=lambda x: x[0], reverse=True)
        k = int(sum(1 for s in scores if s[0] >= bridge_min_score))
        topk = scores[: max(1, bridge_topk_mean_k)]
        b_top = float(sum(x[0] for x in topk) / len(topk))

        if k < bridge_min_count or b_top < bridge_min_topk_mean:
            continue

        main_candidates.append(
            {
                "src": str(ma),
                "dst": str(mb),
                "bridge_count": int(k),
                "bridge_topk_mean": float(b_top),
                "bridge_best_score": float(scores[0][0]),
                "example_pairs": [
                    {
                        "sub_a": str(x[1]),
                        "sub_b": str(x[2]),
                        "bridge": float(x[0]),
                        "sim_concept": float(x[3]),
                        "seed_subject_affinity": float(x[4]),
                    }
                    for x in scores[:3]
                ],
            }
        )

    thread_local = threading.local()

    def _get_llm():
        if not enable_llm_gate:
            return None
        if not hasattr(thread_local, "llm"):
            thread_local.llm = build_llm_client(
                base_url=llm_base_url,
                model=llm_model,
                api_key_env=llm_api_key_env,
                api_key=llm_api_key,
            )
        return thread_local.llm

    def _judge_one(cand: dict) -> dict:
        if not enable_llm_gate:
            return {"allow": True, "llm_conf": 1.0, "llm_vote": 1.0}

        a = node_info.get(str(cand["src"]), {})
        b = node_info.get(str(cand["dst"]), {})
        prompt = _build_main_sem_prompt(
            a,
            b,
            top_sub_text.get(str(cand["src"]), []),
            top_sub_text.get(str(cand["dst"]), []),
        )

        allow_votes = 0
        confs: list[float] = []
        llm = _get_llm()
        for _ in range(max(1, llm_votes)):
            try:
                obj = llm.json_completion(
                    system_prompt=SYSTEM_PROMPT_MAIN_SEM,
                    user_prompt=prompt,
                    temperature=llm_temperature,
                )
                a_raw = str(obj.get("allow", "no")).strip().lower()
                allow = a_raw in {"yes", "true", "1", "allow"}
                c = float(obj.get("confidence", 0.0))
            except Exception:
                allow = False
                c = 0.0
            if allow:
                allow_votes += 1
            confs.append(max(0.0, min(1.0, c)))

        vote_conf = allow_votes / max(1, llm_votes)
        conf_mean = sum(confs) / max(1, len(confs))
        llm_conf = 0.6 * vote_conf + 0.4 * conf_mean
        allow = (allow_votes > (llm_votes // 2)) and (llm_conf >= llm_min_conf)
        return {"allow": bool(allow), "llm_conf": float(llm_conf), "llm_vote": float(vote_conf)}

    judged: list[dict] = []
    if main_candidates:
        if enable_llm_gate and llm_workers > 1:
            with ThreadPoolExecutor(max_workers=llm_workers) as ex:
                futs = {ex.submit(_judge_one, c): c for c in main_candidates}
                for fut in as_completed(futs):
                    c = futs[fut]
                    j = fut.result()
                    row = dict(c)
                    row.update(j)
                    judged.append(row)
        else:
            for c in main_candidates:
                j = _judge_one(c)
                row = dict(c)
                row.update(j)
                judged.append(row)

    kept_main: list[dict] = []
    for r in judged:
        if enable_llm_gate and not bool(r.get("allow", False)):
            continue
        conf = 0.7 * float(r.get("bridge_topk_mean", 0.0)) + 0.3 * float(r.get("llm_conf", 1.0))
        x = dict(r)
        x["conf"] = float(conf)
        kept_main.append(x)

    kept_main = sorted(kept_main, key=lambda r: float(r.get("conf", 0.0)), reverse=True)

    deg_main: dict[str, int] = defaultdict(int)
    main_rows: list[dict] = []
    for r in kept_main:
        a = str(r["src"])
        b = str(r["dst"])
        if max_degree_per_main > 0:
            if deg_main[a] >= max_degree_per_main or deg_main[b] >= max_degree_per_main:
                continue
        main_rows.append(r)
        deg_main[a] += 1
        deg_main[b] += 1

    main_sem_df = pd.DataFrame(main_rows)
    if main_sem_df.empty:
        main_sem_df = pd.DataFrame(
            columns=[
                "src",
                "dst",
                "conf",
                "bridge_count",
                "bridge_topk_mean",
                "bridge_best_score",
                "allow",
                "llm_conf",
                "llm_vote",
                "example_pairs",
            ]
        )

    save_df(main_sem_df, str(graph_dir / "edges_sem_main.parquet"))

    print(f"[OK] sub sem edges: {len(sub_sem_df)}")
    print(f"[OK] main sem edges: {len(main_sem_df)}")


if __name__ == "__main__":
    main()
