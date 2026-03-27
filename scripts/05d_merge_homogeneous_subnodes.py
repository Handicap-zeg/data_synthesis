from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _norm_concept(s: Any) -> str:
    return " ".join(str(s or "").lower().split())


def _phrase_set(s: Any) -> set[str]:
    t = "".join(ch.lower() if (ch.isalnum() or ch in "| ") else " " for ch in str(s or ""))
    return {p.strip() for p in t.split("|") if p.strip()}


class UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.p:
            self.p[x] = x
            return x
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _groupby_agg_pre(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["parent_node_id", "src", "dst"]
    agg: dict[str, str] = {}
    for c in df.columns:
        if c in keys:
            continue
        if c in {"conf", "pair_cooccur", "sim_concept"}:
            agg[c] = "max"
        elif c in {"difficulty_gap"}:
            agg[c] = "min"
        else:
            agg[c] = "first"
    return df.groupby(keys, as_index=False).agg(agg)


def _groupby_agg_sem(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["parent_node_id", "src", "dst"]
    agg: dict[str, str] = {}
    for c in df.columns:
        if c in keys:
            continue
        if c in {"conf", "pair_cooccur", "sim_concept"}:
            agg[c] = "max"
        else:
            agg[c] = "first"
    return df.groupby(keys, as_index=False).agg(agg)


def _map_subnode_list(v: Any, sid_map: dict[str, str]) -> np.ndarray:
    out: list[str] = []
    seen: set[str] = set()
    arr = []
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = list(v)
    elif pd.isna(v):
        arr = []
    else:
        arr = [v]
    for x in arr:
        s = sid_map.get(str(x), str(x))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return np.array(out, dtype=object)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/home/dataset-local/usr/lh/zeg/data_synthesis/data/graph")
    ap.add_argument("--sim-th", type=float, default=0.56)
    ap.add_argument("--require-shared-phrase", type=int, default=1)
    args = ap.parse_args()

    graph_dir = Path(args.graph_dir)
    sub_p = graph_dir / "subnodes.parquet"
    ssi_p = graph_dir / "seeds_sub_index.parquet"
    q2sub_p = graph_dir / "q2subnodes.parquet"
    pre_p = graph_dir / "edges_pre_sub.parquet"
    sem_p = graph_dir / "edges_sem_sub.parquet"

    sub = pd.read_parquet(sub_p).copy()
    ssi = pd.read_parquet(ssi_p).copy()
    q2sub = pd.read_parquet(q2sub_p).copy() if q2sub_p.exists() else pd.DataFrame(columns=["qid", "subnode_ids"])
    pre = pd.read_parquet(pre_p).copy() if pre_p.exists() else pd.DataFrame(columns=["parent_node_id", "src", "dst"])
    sem = pd.read_parquet(sem_p).copy() if sem_p.exists() else pd.DataFrame(columns=["parent_node_id", "src", "dst"])

    sub["subnode_id"] = sub["subnode_id"].astype(str)
    sub["parent_node_id"] = sub["parent_node_id"].astype(str)
    sub["concept"] = sub["concept"].astype(str)
    sub["freq"] = pd.to_numeric(sub.get("freq", 1), errors="coerce").fillna(1).astype(int)

    ssi["subnode_id"] = ssi["subnode_id"].astype(str)
    ssi["qid"] = ssi["qid"].astype(str)

    sub["concept_norm"] = sub["concept"].map(_norm_concept)
    sub["phr"] = sub["concept"].map(_phrase_set)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(sub["concept_norm"].tolist())
    S = linear_kernel(X, X)
    np.fill_diagonal(S, 0.0)

    sid = sub["subnode_id"].tolist()
    pid = sub["parent_node_id"].tolist()
    phr = sub["phr"].tolist()
    n = len(sid)

    uf = UF()
    for s in sid:
        uf.find(s)

    pair_rows: list[dict[str, Any]] = []
    th = float(args.sim_th)
    require_shared = int(args.require_shared_phrase) == 1

    for i in range(n):
        js = np.where(S[i] >= th)[0]
        for j in js:
            if j <= i:
                continue
            if pid[i] != pid[j]:
                continue
            if require_shared and len(phr[i] & phr[j]) < 1:
                continue
            a, b = sid[i], sid[j]
            uf.union(a, b)
            pair_rows.append(
                {
                    "sub_a": a,
                    "sub_b": b,
                    "parent_node_id": pid[i],
                    "sim_char_tfidf": float(S[i, j]),
                    "shared_phrase_cnt": int(len(phr[i] & phr[j])),
                }
            )

    groups: dict[str, list[str]] = defaultdict(list)
    for s in sid:
        groups[uf.find(s)].append(s)

    info = sub.set_index("subnode_id").to_dict(orient="index")
    sid_map: dict[str, str] = {}
    merge_rows: list[dict[str, Any]] = []

    for root, members in groups.items():
        members = sorted(set(members))
        if len(members) == 1:
            sid_map[members[0]] = members[0]
            merge_rows.append(
                {
                    "old_subnode_id": members[0],
                    "new_subnode_id": members[0],
                    "cluster_size": 1,
                    "merged": 0,
                    "parent_node_id": info[members[0]]["parent_node_id"],
                }
            )
            continue

        rep = sorted(
            members,
            key=lambda x: (
                int(info[x].get("freq", 0)),
                len(str(info[x].get("concept", ""))),
                -int(str(x).replace("S", "") or 0),
            ),
            reverse=True,
        )[0]

        for m in members:
            sid_map[m] = rep
            merge_rows.append(
                {
                    "old_subnode_id": m,
                    "new_subnode_id": rep,
                    "cluster_size": len(members),
                    "merged": int(m != rep),
                    "parent_node_id": info[m]["parent_node_id"],
                }
            )

    merge_map = pd.DataFrame(merge_rows)

    # Build merged subnodes
    inv: dict[str, list[str]] = defaultdict(list)
    for old, new in sid_map.items():
        inv[new].append(old)

    new_sub_rows: list[dict[str, Any]] = []
    for rep, members in inv.items():
        rows = [info[m] for m in members]
        new_sub_rows.append(
            {
                "subnode_id": rep,
                "parent_node_id": rows[0]["parent_node_id"],
                "concept": str(info[rep]["concept"]),
                "freq": int(sum(int(r.get("freq", 0)) for r in rows)),
            }
        )

    sub_new = pd.DataFrame(new_sub_rows).sort_values(["parent_node_id", "subnode_id"]).reset_index(drop=True)

    # Remap seeds_sub_index
    ssi_new = ssi.copy()
    ssi_new["subnode_id"] = ssi_new["subnode_id"].map(lambda x: sid_map.get(str(x), str(x)))
    ssi_new = ssi_new.drop_duplicates(subset=["subnode_id", "qid"], keep="first").reset_index(drop=True)

    # Remap q2subnodes
    q2sub_new = q2sub.copy()
    if not q2sub_new.empty and "subnode_ids" in q2sub_new.columns:
        q2sub_new["subnode_ids"] = q2sub_new["subnode_ids"].map(lambda v: _map_subnode_list(v, sid_map))

    # Parent lookup
    parent_of = {r["subnode_id"]: r["parent_node_id"] for r in new_sub_rows}

    # Remap pre edges
    pre_new = pre.copy()
    if not pre_new.empty:
        pre_new["src"] = pre_new["src"].astype(str).map(lambda x: sid_map.get(x, x))
        pre_new["dst"] = pre_new["dst"].astype(str).map(lambda x: sid_map.get(x, x))
        pre_new = pre_new[pre_new["src"] != pre_new["dst"]].copy()
        pre_new["parent_node_id"] = pre_new["src"].map(lambda s: parent_of.get(s, ""))
        pre_new = pre_new[(pre_new["parent_node_id"] != "") & (pre_new["src"].map(parent_of.get) == pre_new["dst"].map(parent_of.get))]
        pre_new = _groupby_agg_pre(pre_new)

    # Remap sem edges
    sem_new = sem.copy()
    if not sem_new.empty:
        sem_new["src"] = sem_new["src"].astype(str).map(lambda x: sid_map.get(x, x))
        sem_new["dst"] = sem_new["dst"].astype(str).map(lambda x: sid_map.get(x, x))
        sem_new = sem_new[sem_new["src"] != sem_new["dst"]].copy()
        # undirected canonicalization
        s = sem_new[["src", "dst"]].min(axis=1)
        d = sem_new[["src", "dst"]].max(axis=1)
        sem_new["src"] = s
        sem_new["dst"] = d
        sem_new["parent_node_id"] = sem_new["src"].map(lambda x: parent_of.get(x, ""))
        sem_new = sem_new[(sem_new["parent_node_id"] != "") & (sem_new["src"].map(parent_of.get) == sem_new["dst"].map(parent_of.get))]
        sem_new = _groupby_agg_sem(sem_new)

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = graph_dir / f"backup_subnode_merge_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in [sub_p, ssi_p, q2sub_p, pre_p, sem_p]:
        if p.exists():
            shutil.copy2(p, backup_dir / p.name)

    # Save
    sub_new.to_parquet(sub_p, index=False)
    ssi_new.to_parquet(ssi_p, index=False)
    if not q2sub_new.empty:
        q2sub_new.to_parquet(q2sub_p, index=False)
    if not pre_new.empty:
        pre_new.to_parquet(pre_p, index=False)
    if not sem_new.empty:
        sem_new.to_parquet(sem_p, index=False)

    pd.DataFrame(pair_rows).to_parquet(graph_dir / "subnode_merge_pairs.parquet", index=False)
    merge_map.to_parquet(graph_dir / "subnode_merge_map.parquet", index=False)
    merge_map.to_csv(graph_dir / "subnode_merge_map.csv", index=False)

    merged_nodes = int(merge_map["merged"].sum())
    clusters = int((merge_map["cluster_size"] > 1).sum())

    print(f"[OK] sim_th={th:.2f}, require_shared_phrase={int(require_shared)}")
    print(f"[OK] subnodes: {len(sub)} -> {len(sub_new)} (merged={merged_nodes})")
    print(f"[OK] clusters(with size>1 member-rows): {clusters}")
    print(f"[OK] seeds_sub_index: {len(ssi)} -> {len(ssi_new)}")
    print(f"[OK] q2subnodes: {len(q2sub)} -> {len(q2sub_new)}")
    print(f"[OK] pre_sub: {len(pre)} -> {len(pre_new)}")
    print(f"[OK] sem_sub: {len(sem)} -> {len(sem_new)}")
    print(f"[OK] backup: {backup_dir}")
    print(f"[OK] merge_map: {graph_dir / 'subnode_merge_map.parquet'}")


if __name__ == "__main__":
    main()
