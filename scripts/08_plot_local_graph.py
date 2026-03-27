from __future__ import annotations

import argparse
import math
import re
import textwrap
from collections import deque
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_COLOR = "#4c78a8"
SUBNODE_COLOR = "#d8e2ef"
MSC_PLACEHOLDER_RE = re.compile(r"^\s*msc\s*\d{2}[A-Za-z]\d{2}\s*$", re.IGNORECASE)


def clean_label_text(s: object) -> str:
    t = str(s or "").strip()
    return re.sub(r"\s+", " ", t)


def _theme_text(row: pd.Series) -> str:
    description = clean_label_text(row.get("description", ""))
    concept = clean_label_text(row.get("concept", ""))
    desc = clean_label_text(row.get("msc_desc", ""))

    if description:
        return description

    if concept:
        parts = [clean_label_text(x) for x in re.split(r"[|;/,]", concept) if clean_label_text(x)]
        if parts:
            return " | ".join(parts[:3])
        return concept

    if desc and not MSC_PLACEHOLDER_RE.match(desc):
        return desc

    return "topic cluster"


def node_label_text(row: pd.Series) -> str:
    code3 = clean_label_text(row.get("msc_code3", ""))
    main = _theme_text(row)
    txt = (f"{code3} {main}" if code3 else main).strip()
    wrapped = "\n".join(textwrap.wrap(txt, width=24, break_long_words=False, break_on_hyphens=False))
    return wrapped


def subnode_label_text(row: pd.Series) -> str:
    cluster = clean_label_text(row.get("concept", ""))
    if not cluster:
        cluster = clean_label_text(row.get("concept_cluster", ""))
    if not cluster:
        cluster = f"subnode {clean_label_text(row.get('subnode_id', ''))}"
    wrapped = "\n".join(textwrap.wrap(cluster, width=22, break_long_words=False, break_on_hyphens=False))
    return wrapped


def lighten_hex(color: str, ratio: float = 0.35) -> str:
    c = str(color).lstrip("#")
    if len(c) != 6:
        return color
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    rr = int((1.0 - ratio) * r + ratio * 255)
    gg = int((1.0 - ratio) * g + ratio * 255)
    bb = int((1.0 - ratio) * b + ratio * 255)
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def top_edges(df: pd.DataFrame, max_n: int, min_conf: float) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "conf" in work.columns:
        work["conf"] = pd.to_numeric(work["conf"], errors="coerce").fillna(0.0).astype(float)
        work = work[work["conf"] >= float(min_conf)].copy()
        work = work.sort_values("conf", ascending=False)
    if max_n > 0:
        work = work.head(int(max_n))
    return work


def scale_widths(values: pd.Series | np.ndarray, min_w: float, max_w: float, use_log1p: bool = True) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    if use_log1p:
        arr = np.log1p(arr)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, (min_w + max_w) * 0.5, dtype=float)
    z = (arr - lo) / (hi - lo)
    return (float(min_w) + (float(max_w) - float(min_w)) * z).astype(float)


def pick_center(nodes: pd.DataFrame, sem_main: pd.DataFrame) -> str:
    deg = {nid: 0 for nid in nodes["node_id"].astype(str).tolist()}
    for r in sem_main.itertuples(index=False):
        a, b = str(r.src), str(r.dst)
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    freq = dict(zip(nodes["node_id"].astype(str), pd.to_numeric(nodes["freq"], errors="coerce").fillna(0.0).astype(float)))
    return max(nodes["node_id"].astype(str).tolist(), key=lambda x: (deg.get(x, 0), freq.get(x, 0.0)))


def bfs_local(gu: nx.Graph, center: str, hops: int) -> set[str]:
    vis = {center}
    q = deque([(center, 0)])
    while q:
        u, d = q.popleft()
        if d >= hops:
            continue
        for v in gu.neighbors(u):
            if v not in vis:
                vis.add(v)
                q.append((v, d + 1))
    return vis


def bfs_depths(gu: nx.Graph, center: str, max_hops: int) -> dict[str, int]:
    depths = {center: 0}
    q = deque([center])
    while q:
        u = q.popleft()
        d = depths[u]
        if d >= max_hops:
            continue
        for v in gu.neighbors(u):
            if v not in depths:
                depths[v] = d + 1
                q.append(v)
    return depths


def radial_init_positions(nodes: list[str], depths: dict[str, int], center: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    layers: dict[int, list[str]] = {}
    max_depth = max(depths.values()) if depths else 1
    for n in nodes:
        d = int(depths.get(n, max_depth))
        layers.setdefault(d, []).append(n)

    pos: dict[str, np.ndarray] = {center: np.array([0.0, 0.0], dtype=np.float32)}
    max_layer = max(layers.keys()) if layers else 0

    for d in range(1, max_layer + 1):
        cur = layers.get(d, [])
        if not cur:
            continue
        r = 1.45 * d
        base = rng.uniform(0.0, 2.0 * math.pi)
        for i, nid in enumerate(cur):
            ang = base + 2.0 * math.pi * i / max(1, len(cur))
            jitter = rng.normal(0, 0.06, size=2)
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            pos[nid] = np.array([x, y], dtype=np.float32) + jitter
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graph")
    ap.add_argument("--out-dir", default="data/figures")
    ap.add_argument("--center-node", default=None)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=16)
    ap.add_argument("--layer-mode", choices=["main", "both"], default="both")
    ap.add_argument("--max-subnodes-per-main", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pre-max", type=int, default=700)
    ap.add_argument("--sem-max", type=int, default=500)
    ap.add_argument("--pre-min-conf", type=float, default=0.0)
    ap.add_argument("--sem-min-conf", type=float, default=0.0)
    ap.add_argument("--label-sub-top", type=int, default=0)
    ap.add_argument("--drop-isolates", type=int, default=1)
    ap.add_argument("--only-center-component", type=int, default=1)
    ap.add_argument("--intra-main-only", type=int, default=1)
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    gdir = Path(args.graph_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(gdir / "nodes.parquet")
    if "node_id" not in nodes.columns:
        raise RuntimeError("nodes.parquet missing node_id")
    for col, default in [("msc_code3", ""), ("description", ""), ("concept", ""), ("freq", 1.0)]:
        if col not in nodes.columns:
            nodes[col] = default
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["msc_code3"] = nodes["msc_code3"].astype(str)
    nodes["description"] = nodes["description"].astype(str)
    nodes["concept"] = nodes["concept"].astype(str)
    nodes["freq"] = pd.to_numeric(nodes["freq"], errors="coerce").fillna(1.0).astype(float)

    valid_main = set(nodes["node_id"].tolist())

    sem_main_path = gdir / "edges_sem_main.parquet"
    if sem_main_path.exists():
        sem_main = pd.read_parquet(sem_main_path)
    elif (gdir / "edges_sem.parquet").exists():
        sem_main = pd.read_parquet(gdir / "edges_sem.parquet")
    else:
        sem_main = pd.DataFrame(columns=["src", "dst", "conf"])

    for col in ["src", "dst"]:
        if col not in sem_main.columns:
            sem_main[col] = ""
    if "conf" not in sem_main.columns:
        sem_main["conf"] = 1.0
    sem_main["src"] = sem_main["src"].astype(str)
    sem_main["dst"] = sem_main["dst"].astype(str)
    sem_main["conf"] = pd.to_numeric(sem_main["conf"], errors="coerce").fillna(0.0).astype(float)
    sem_main = sem_main[sem_main["src"].isin(valid_main) & sem_main["dst"].isin(valid_main)].copy()

    gu_main = nx.Graph()
    gu_main.add_nodes_from(valid_main)
    gu_main.add_edges_from(sem_main[["src", "dst"]].itertuples(index=False, name=None))

    subnodes = pd.DataFrame()
    sub_to_parent: dict[str, str] = {}
    if args.layer_mode == "both" and (gdir / "subnodes.parquet").exists():
        subnodes = pd.read_parquet(gdir / "subnodes.parquet")
        for col, default in [("subnode_id", ""), ("parent_node_id", ""), ("concept", ""), ("concept_cluster", ""), ("freq", 1.0)]:
            if col not in subnodes.columns:
                subnodes[col] = default
        subnodes["subnode_id"] = subnodes["subnode_id"].astype(str)
        subnodes["parent_node_id"] = subnodes["parent_node_id"].astype(str)
        subnodes["concept"] = subnodes["concept"].astype(str)
        subnodes["concept_cluster"] = subnodes["concept_cluster"].astype(str)
        subnodes["freq"] = pd.to_numeric(subnodes["freq"], errors="coerce").fillna(1.0).astype(float)
        sub_to_parent = dict(zip(subnodes["subnode_id"].tolist(), subnodes["parent_node_id"].tolist()))

    center_raw = str(args.center_node).strip() if args.center_node is not None else ""
    center_sub: Optional[str] = None
    if center_raw in valid_main:
        center = center_raw
    elif center_raw and center_raw in sub_to_parent:
        center_sub = center_raw
        center = sub_to_parent[center_raw]
    else:
        center = pick_center(nodes, sem_main)

    if int(args.intra_main_only) == 1:
        keep = {center}
    else:
        keep = bfs_local(gu_main, center, int(args.hops)) if center in gu_main else {center}
    if len(keep) > int(args.max_nodes):
        deg = dict(gu_main.degree(keep))
        freq_map = dict(zip(nodes["node_id"], nodes["freq"]))
        depth_map = bfs_depths(gu_main, center, int(args.hops)) if center in gu_main else {center: 0}
        ranked = sorted(
            keep,
            key=lambda x: (
                x != center,
                int(depth_map.get(x, 10**9)),
                -int(deg.get(x, 0)),
                -float(freq_map.get(x, 0.0)),
                x,
            ),
        )
        keep = set(ranked[: int(args.max_nodes)])

    ndf = nodes[nodes["node_id"].isin(keep)].copy()
    sem_main_l = sem_main[sem_main["src"].isin(keep) & sem_main["dst"].isin(keep)].copy()
    sem_main_l = top_edges(sem_main_l, int(args.sem_max), float(args.sem_min_conf))

    gu_local = nx.Graph()
    gu_local.add_nodes_from(ndf["node_id"].tolist())
    gu_local.add_edges_from(sem_main_l[["src", "dst"]].itertuples(index=False, name=None))

    if int(args.drop_isolates) == 1:
        active = {n for n, d in gu_local.degree() if d > 0}
        if center in set(ndf["node_id"].tolist()):
            active.add(center)
        if len(active) >= 1:
            keep2 = set(active)
            ndf = ndf[ndf["node_id"].isin(keep2)].copy()
            sem_main_l = sem_main_l[sem_main_l["src"].isin(keep2) & sem_main_l["dst"].isin(keep2)].copy()
            gu_local = nx.Graph()
            gu_local.add_nodes_from(ndf["node_id"].tolist())
            gu_local.add_edges_from(sem_main_l[["src", "dst"]].itertuples(index=False, name=None))

    if int(args.only_center_component) == 1 and center in gu_local and gu_local.number_of_edges() > 0:
        cset = set(nx.node_connected_component(gu_local, center))
        ndf = ndf[ndf["node_id"].isin(cset)].copy()
        sem_main_l = sem_main_l[sem_main_l["src"].isin(cset) & sem_main_l["dst"].isin(cset)].copy()
        gu_local = nx.Graph()
        gu_local.add_nodes_from(ndf["node_id"].tolist())
        gu_local.add_edges_from(sem_main_l[["src", "dst"]].itertuples(index=False, name=None))

    if len(ndf) == 0:
        raise RuntimeError("No main nodes left to plot; try larger --hops or disable --drop-isolates")

    depths = bfs_depths(gu_local, center, int(args.hops)) if center in gu_local else {center: 0}
    init_pos = radial_init_positions(ndf["node_id"].astype(str).tolist(), depths, center, int(args.seed))
    pos = nx.spring_layout(
        gu_local,
        seed=int(args.seed),
        pos=init_pos if len(init_pos) == len(ndf) else None,
        fixed=[center] if center in gu_local else None,
        k=3.0 / max(1.0, np.sqrt(float(len(ndf)))),
        iterations=420,
        weight=None,
        scale=1.0,
    )

    if center in pos:
        cxy = pos[center].copy()
        for nid in pos:
            pos[nid] = pos[nid] - cxy

    sub_l = pd.DataFrame()
    sub_pos: dict[str, np.ndarray] = {}
    parent_sub_edges: list[tuple[str, str]] = []
    sub_pre_l = pd.DataFrame(columns=["src", "dst", "conf"])
    sub_sem_l = pd.DataFrame(columns=["src", "dst", "conf"])

    if args.layer_mode == "both" and len(subnodes) > 0:
        main_keep = set(ndf["node_id"].astype(str).tolist())
        work = subnodes[subnodes["parent_node_id"].isin(main_keep)].copy()
        if len(work) > 0:
            work = work.sort_values(["parent_node_id", "freq", "subnode_id"], ascending=[True, False, True])
            cap = int(args.max_subnodes_per_main)
            if cap > 0:
                work = work.groupby("parent_node_id", as_index=False, group_keys=False).head(cap)
            if center_sub is not None and center_sub in sub_to_parent:
                p = sub_to_parent[center_sub]
                if p in main_keep and center_sub not in set(work["subnode_id"].tolist()):
                    add = subnodes[subnodes["subnode_id"] == center_sub]
                    if len(add) > 0:
                        work = pd.concat([work, add], ignore_index=True)
            work = work.drop_duplicates(subset=["subnode_id"]).copy()
            sub_l = work

            rng = np.random.default_rng(int(args.seed) + 7919)
            for pid, grp in sub_l.groupby("parent_node_id"):
                pid = str(pid)
                if pid not in pos:
                    continue
                sids = grp.sort_values(["freq", "subnode_id"], ascending=[False, True])["subnode_id"].astype(str).tolist()
                k = len(sids)
                if k <= 0:
                    continue
                parent_xy = pos[pid]
                base_ang = float(rng.uniform(0.0, 2.0 * math.pi))
                radius = 0.18 + 0.02 * min(k, 8)
                for i, sid in enumerate(sids):
                    ang = base_ang + 2.0 * math.pi * i / max(1, k)
                    jitter = rng.normal(0.0, 0.01, size=2)
                    offset = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) * radius
                    sub_pos[sid] = parent_xy + offset + jitter
                    parent_sub_edges.append((pid, sid))

            sid_keep = set(sub_l["subnode_id"].astype(str).tolist())

            pre_sub_path = gdir / "edges_pre_sub.parquet"
            if pre_sub_path.exists():
                sub_pre = pd.read_parquet(pre_sub_path)
                for col in ["src", "dst"]:
                    if col not in sub_pre.columns:
                        sub_pre[col] = ""
                if "conf" not in sub_pre.columns:
                    sub_pre["conf"] = 0.0
                sub_pre["src"] = sub_pre["src"].astype(str)
                sub_pre["dst"] = sub_pre["dst"].astype(str)
                sub_pre["conf"] = pd.to_numeric(sub_pre["conf"], errors="coerce").fillna(0.0).astype(float)
                sub_pre = sub_pre[sub_pre["src"].isin(sid_keep) & sub_pre["dst"].isin(sid_keep)].copy()
                sub_pre_l = top_edges(sub_pre, int(args.pre_max), float(args.pre_min_conf))

            sem_sub_path = gdir / "edges_sem_sub.parquet"
            if sem_sub_path.exists():
                sub_sem = pd.read_parquet(sem_sub_path)
                for col in ["src", "dst"]:
                    if col not in sub_sem.columns:
                        sub_sem[col] = ""
                if "conf" not in sub_sem.columns:
                    if "pair_cooccur" in sub_sem.columns:
                        sub_sem["conf"] = pd.to_numeric(sub_sem["pair_cooccur"], errors="coerce").fillna(0.0)
                    else:
                        sub_sem["conf"] = 0.0
                sub_sem["src"] = sub_sem["src"].astype(str)
                sub_sem["dst"] = sub_sem["dst"].astype(str)
                sub_sem["conf"] = pd.to_numeric(sub_sem["conf"], errors="coerce").fillna(0.0).astype(float)
                sub_sem = sub_sem[sub_sem["src"].isin(sid_keep) & sub_sem["dst"].isin(sid_keep)].copy()
                sub_sem_l = top_edges(sub_sem, int(args.sem_max), float(args.sem_min_conf))

    plt.figure(figsize=(13, 10), dpi=260)
    ax = plt.gca()
    ax.set_facecolor("#f5f7fa")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.20)

    if len(sem_main_l) > 0:
        sem_w = scale_widths(sem_main_l["conf"], min_w=0.7, max_w=2.0, use_log1p=True)
        nx.draw_networkx_edges(
            gu_local,
            pos,
            edgelist=list(sem_main_l[["src", "dst"]].itertuples(index=False, name=None)),
            width=sem_w,
            edge_color="#b9c8d8",
            alpha=0.55,
        )

    f = ndf["freq"].fillna(1).astype(float).to_numpy()
    lf = np.log1p(f)
    lfmin, lfmax = float(lf.min()), float(lf.max())
    norm = (lf - lfmin) / (lfmax - lfmin + 1e-8)
    sizes = 95 + 460 * norm
    colors = [DEFAULT_COLOR for _ in range(len(ndf))]

    nx.draw_networkx_nodes(
        gu_local,
        pos,
        nodelist=ndf["node_id"].tolist(),
        node_size=sizes,
        node_color=colors,
        edgecolors="#0f1720",
        linewidths=0.55,
        alpha=0.96,
    )

    if len(sub_l) > 0 and len(sub_pos) > 0:
        pos_all = dict(pos)
        pos_all.update(sub_pos)

        if len(sub_sem_l) > 0:
            sub_sem_w = scale_widths(sub_sem_l["conf"], min_w=0.35, max_w=1.5, use_log1p=True)
            sub_sem_g = nx.Graph()
            sub_sem_g.add_nodes_from(pos_all.keys())
            sub_sem_draw = list(sub_sem_l[["src", "dst"]].itertuples(index=False, name=None))
            sub_sem_g.add_edges_from(sub_sem_draw)
            nx.draw_networkx_edges(
                sub_sem_g,
                pos_all,
                edgelist=sub_sem_draw,
                width=sub_sem_w,
                edge_color="#c7d2de",
                alpha=0.40,
            )

        if len(sub_pre_l) > 0:
            sub_pre_w = scale_widths(sub_pre_l["conf"], min_w=0.45, max_w=1.3, use_log1p=False)
            sub_pre_g = nx.DiGraph()
            sub_pre_g.add_nodes_from(pos_all.keys())
            sub_pre_draw = list(sub_pre_l[["src", "dst"]].itertuples(index=False, name=None))
            sub_pre_g.add_edges_from(sub_pre_draw)
            nx.draw_networkx_edges(
                sub_pre_g,
                pos_all,
                edgelist=sub_pre_draw,
                width=sub_pre_w,
                edge_color="#29456a",
                alpha=0.80,
                arrows=True,
                arrowsize=9,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.04",
            )

        parent_sub_draw = [(a, b) for (a, b) in parent_sub_edges if a in pos_all and b in pos_all]
        if len(parent_sub_draw) > 0:
            link_g = nx.Graph()
            link_g.add_nodes_from(pos_all.keys())
            link_g.add_edges_from(parent_sub_draw)
            nx.draw_networkx_edges(
                link_g,
                pos_all,
                edgelist=parent_sub_draw,
                width=0.8,
                edge_color="#7a8797",
                alpha=0.60,
                style="dashed",
            )

        sub_df = sub_l[sub_l["subnode_id"].isin(sub_pos.keys())].copy()
        if len(sub_df) > 0:
            sf = sub_df["freq"].fillna(1).astype(float).to_numpy()
            slf = np.log1p(sf)
            slmin, slmax = float(slf.min()), float(slf.max())
            snorm = (slf - slmin) / (slmax - slmin + 1e-8)
            sub_sizes = 42 + 110 * snorm
            sub_colors = [SUBNODE_COLOR for _ in range(len(sub_df))]
            draw_sid = sub_df["subnode_id"].astype(str).tolist()
            draw_g = nx.Graph()
            draw_g.add_nodes_from(pos_all.keys())
            nx.draw_networkx_nodes(
                draw_g,
                pos_all,
                nodelist=draw_sid,
                node_shape="o",
                node_size=sub_sizes,
                node_color=sub_colors,
                edgecolors="#263445",
                linewidths=0.45,
                alpha=0.92,
            )

    # Label policy (main nodes):
    # 1) always center
    # 2) all endpoints of main SEM edges in local graph
    # 3) plus up to 2 direct neighbors of center
    if int(args.intra_main_only) == 1:
        show = [center]
    else:
        sem_endpoints: set[str] = set()
        if len(sem_main_l) > 0:
            sem_endpoints.update(sem_main_l["src"].astype(str).tolist())
            sem_endpoints.update(sem_main_l["dst"].astype(str).tolist())

        neighbor_extra: list[str] = []
        if center in gu_local:
            freq_map = dict(zip(ndf["node_id"].astype(str), ndf["freq"].astype(float)))
            nbrs = [str(x) for x in gu_local.neighbors(center)]
            nbrs = sorted(
                nbrs,
                key=lambda x: (
                    x not in sem_endpoints,
                    -int(gu_local.degree(x)),
                    -float(freq_map.get(x, 0.0)),
                    x,
                ),
            )
            neighbor_extra = nbrs[:2]

        show = [center] + sorted([x for x in sem_endpoints if x != center]) + neighbor_extra
        show = list(dict.fromkeys(show))

    label_map = {}
    for nid in show:
        rr = ndf[ndf["node_id"] == nid]
        if len(rr) > 0:
            label_map[nid] = node_label_text(rr.iloc[0])

    if len(label_map) > 0:
        nx.draw_networkx_labels(
            gu_local,
            pos,
            labels=label_map,
            font_size=7.2,
            font_color="#0b1220",
            font_weight="regular",
        )

    if len(sub_l) > 0 and len(sub_pos) > 0 and int(args.label_sub_top) > 0:
        sub_df = sub_l[sub_l["subnode_id"].isin(sub_pos.keys())].copy()
        sub_df = sub_df.sort_values(["freq", "subnode_id"], ascending=[False, True])
        sub_show = sub_df["subnode_id"].astype(str).head(int(args.label_sub_top)).tolist()
        if center_sub is not None and center_sub in set(sub_df["subnode_id"].astype(str).tolist()) and center_sub not in sub_show:
            sub_show = [center_sub] + sub_show
        sub_show = list(dict.fromkeys(sub_show))

        sub_label_map = {}
        for sid in sub_show:
            r = sub_df[sub_df["subnode_id"] == sid]
            if len(r) > 0:
                sub_label_map[sid] = subnode_label_text(r.iloc[0])
        if len(sub_label_map) > 0:
            sub_label_graph = nx.Graph()
            sub_label_graph.add_nodes_from(sub_label_map.keys())
            nx.draw_networkx_labels(
                sub_label_graph,
                sub_pos,
                labels=sub_label_map,
                font_size=6.2,
                font_color="#1f2b38",
                font_weight="regular",
            )

    n_sub = len(sub_l) if len(sub_l) > 0 else 0
    center_theme = _theme_text(ndf[ndf["node_id"] == center].iloc[0]) if center in set(ndf["node_id"].astype(str)) else center
    if int(args.intra_main_only) == 1:
        title = (
            f"Internal Subgraph of {center}  "
            f"({center_theme}; sub={n_sub}, sub_pre={len(sub_pre_l)}, sub_sem={len(sub_sem_l)})"
        )
    else:
        title = (
            f"Local Knowledge Subgraph around {center}  "
            f"(main={len(ndf)}, sub={n_sub}, main_sem={len(sem_main_l)}, sub_pre={len(sub_pre_l)}, sub_sem={len(sub_sem_l)})"
        )
    plt.title(title, fontsize=13, weight="semibold")

    legend_items = [Line2D([0], [0], color="#b9c8d8", lw=1.2, label="main E_sem")]

    if args.layer_mode == "both":
        legend_items.extend(
            [
                Line2D([0], [0], color="#29456a", lw=1.0, label="sub E_pre"),
                Line2D([0], [0], color="#c7d2de", lw=1.0, label="sub E_sem"),
                Line2D([0], [0], color="#7a8797", lw=1.0, linestyle="--", label="main-sub link"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=SUBNODE_COLOR,
                    markeredgecolor="#263445",
                    markersize=6,
                    label="subnode",
                ),
            ]
        )

    plt.legend(handles=legend_items, loc="upper left", frameon=True, framealpha=0.95, fontsize=8)
    plt.axis("off")

    png = out / f"local_graph_{center}_{args.layer_mode}_clean.png"
    pdf = out / f"local_graph_{center}_{args.layer_mode}_clean.pdf"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"[OK] saved: {png}")
    print(f"[OK] saved: {pdf}")


if __name__ == "__main__":
    main()
