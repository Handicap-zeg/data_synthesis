from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
import re

import pandas as pd
import numpy as np

from lib.common import load_config, load_df, normalize_text, save_df


FULL_CODE_RE = re.compile(r"^\d{2}[A-Za-z]\d{2}$")
FULL_CODE_FIND_RE = re.compile(r"\d{2}[A-Za-z]\d{2}")


def default_full_from_l1(v: str) -> str:
    l1 = str(v or "00").zfill(2)[:2]
    return f"{l1}A05"


def _mode_value_count_first(s: pd.Series, default: str = "") -> str:
    s = s.astype(str)
    if s.empty:
        return default
    vc = s.value_counts(dropna=False)
    if vc.empty:
        return default
    return str(vc.index[0])


def _mode_nonempty_value_count_first(s: pd.Series, default: str = "") -> str:
    s = s.astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    if s.empty:
        return default
    vc = s.value_counts(dropna=False)
    if vc.empty:
        return default
    return str(vc.index[0])


def _parse_msc_codes(value: object, fallback: str) -> list[str]:
    codes: list[str] = []

    is_iterable_codes = isinstance(value, (list, tuple, set))
    if not is_iterable_codes and hasattr(value, "tolist") and not isinstance(value, str):
        try:
            converted = value.tolist()
            if isinstance(converted, (list, tuple, set)):
                value = converted
                is_iterable_codes = True
            else:
                value = converted
        except Exception:  # noqa: BLE001
            pass

    if is_iterable_codes:
        for x in value:  # type: ignore[assignment]
            c = str(x).strip().upper()
            if FULL_CODE_RE.match(c):
                codes.append(c)
    else:
        text = "" if value is None else str(value).strip()
        if text and text.lower() != "nan":
            # Handle python-list strings and raw text robustly.
            if text.startswith("[") and text.endswith("]"):
                try:
                    obj = ast.literal_eval(text)
                    if isinstance(obj, (list, tuple, set)):
                        for x in obj:
                            c = str(x).strip().upper()
                            if FULL_CODE_RE.match(c):
                                codes.append(c)
                except Exception:  # noqa: BLE001
                    pass
            if not codes:
                codes.extend(FULL_CODE_FIND_RE.findall(text.upper()))

    fb = str(fallback or "").strip().upper()
    if FULL_CODE_RE.match(fb) and fb not in codes:
        codes.insert(0, fb)

    out = []
    seen = set()
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _default_domain_from_l1(msc_l1: str) -> str:
    l1 = str(msc_l1 or "00").zfill(2)[:2]
    if l1 in {"11", "12", "20"}:
        return "number_theory"
    if l1 in {"05", "60", "62", "68"}:
        return "combinatorics"
    if l1 in {"26", "30", "34", "35"}:
        return "calculus"
    if l1 in {"51", "52", "53", "54", "55", "57", "58"}:
        return "geometry"
    return "algebra"


def _load_msc_desc_map(cfg: dict) -> dict[str, str]:
    msc_cfg = cfg.get("msc", {})
    catalog_csv = str(msc_cfg.get("catalog_csv", "data/interim/msc2020_codes.csv"))
    p = Path(catalog_csv)
    if not p.is_absolute():
        p = Path(cfg["_abs_project_root"]) / p
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return {}
    if not {"code", "desc"}.issubset(set(df.columns)):
        return {}
    df["code"] = df["code"].astype(str).str.upper().str.strip()
    df["desc"] = df["desc"].astype(str).str.strip()
    df = df[df["code"].str.match(FULL_CODE_RE, na=False)].copy()
    df = df[df["desc"].str.len() > 0].drop_duplicates(subset=["code"])
    return dict(zip(df["code"], df["desc"]))


def apply_node_prune(df: pd.DataFrame, graph_cfg: dict) -> pd.DataFrame:
    out = df.copy()
    min_freq = int(graph_cfg.get("min_node_freq", 1))
    max_nodes = int(graph_cfg.get("max_nodes", 0))

    if min_freq > 1:
        out = out[out["freq"].astype(int) >= min_freq].copy()

    sort_cols = ["freq"]
    ascending = [False]
    for c in ["msc_full", "concept", "domain"]:
        if c in out.columns:
            sort_cols.append(c)
            ascending.append(True)

    out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if max_nodes > 0 and len(out) > max_nodes:
        out = out.head(max_nodes).copy()

    return out.reset_index(drop=True)


def build_msc_concept_graph(
    seed: pd.DataFrame,
    mentions: pd.DataFrame,
    graph_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mentions = mentions.copy()
    mentions["concept"] = mentions["concept"].map(normalize_text)
    mentions = mentions[mentions["concept"].str.len() >= 3].copy()

    if "msc_l1" not in mentions.columns:
        mentions["msc_l1"] = "00"
    mentions["msc_l1"] = mentions["msc_l1"].astype(str).str.zfill(2)

    if "msc_full" not in mentions.columns:
        mentions["msc_full"] = mentions["msc_l1"].map(default_full_from_l1)
    mentions["msc_full"] = mentions["msc_full"].astype(str).str.upper()

    if "msc_desc" not in mentions.columns:
        mentions["msc_desc"] = ""

    key_cols = ["msc_full", "msc_l1", "msc_desc", "concept", "domain"]
    node_stats = mentions.groupby(key_cols, as_index=False).agg(freq=("qid", "nunique"))
    node_stats = apply_node_prune(node_stats, graph_cfg)

    node_stats["node_id"] = [f"N{i:07d}" for i in range(len(node_stats))]
    nodes = node_stats[["node_id", "msc_full", "msc_l1", "msc_desc", "concept", "domain", "freq"]].copy()

    node_map = nodes[["node_id", "msc_full", "concept", "domain"]]
    seed_links = mentions.merge(node_map, on=["msc_full", "concept", "domain"], how="inner")
    seed_links = seed_links[["node_id", "qid"]].drop_duplicates()
    seed_links = seed_links.merge(seed[["qid", "y", "subject", "level"]], on="qid", how="left")

    q2nodes = (
        seed_links.groupby("qid", as_index=False)
        .agg(node_ids=("node_id", lambda s: sorted(set(s))))
        .merge(seed[["qid", "subject", "y"]], on="qid", how="left")
    )

    topic_table = (
        nodes.groupby(["msc_full", "msc_l1", "domain"], as_index=False)
        .agg(node_count=("node_id", "nunique"), total_freq=("freq", "sum"))
        .sort_values(["node_count", "total_freq"], ascending=[False, False])
    )

    return nodes, seed_links, q2nodes, topic_table


def build_msc_code_graph(
    seed: pd.DataFrame,
    mentions: pd.DataFrame,
    concept_sets: pd.DataFrame | None,
    graph_cfg: dict,
    msc_desc_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mentions = mentions.copy()

    if "msc_l1" not in mentions.columns:
        mentions["msc_l1"] = "00"
    mentions["msc_l1"] = mentions["msc_l1"].astype(str).str.zfill(2)

    if "msc_full" not in mentions.columns:
        mentions["msc_full"] = mentions["msc_l1"].map(default_full_from_l1)
    mentions["msc_full"] = mentions["msc_full"].astype(str).str.upper()

    if "msc_desc" not in mentions.columns:
        mentions["msc_desc"] = ""
    mentions["msc_desc"] = mentions["msc_desc"].astype(str)

    mentions = mentions[mentions["msc_full"].str.match(FULL_CODE_RE, na=False)].copy()
    if mentions.empty:
        raise RuntimeError("No valid msc_full rows found in concept_mentions.parquet")

    if concept_sets is not None and {"qid", "msc_codes"}.issubset(set(concept_sets.columns)):
        work = concept_sets[["qid", "msc_codes", "msc_full"]].copy()
        links_rows = []
        for r in work.itertuples(index=False):
            for code in _parse_msc_codes(r.msc_codes, r.msc_full):
                links_rows.append({"qid": str(r.qid), "msc_full": code})
        qid_codes = pd.DataFrame(links_rows).drop_duplicates()
    else:
        qid_codes = mentions[["qid", "msc_full"]].copy().drop_duplicates()
        qid_codes["qid"] = qid_codes["qid"].astype(str)

    qid_codes = qid_codes[qid_codes["msc_full"].astype(str).str.match(FULL_CODE_RE, na=False)].copy()
    if qid_codes.empty:
        raise RuntimeError("No valid qid->msc_full links available for msc_code mode")

    meta = mentions.groupby("msc_full", as_index=False).agg(
        msc_l1=("msc_l1", lambda s: _mode_value_count_first(s, "00")),
        msc_desc=("msc_desc", lambda s: _mode_nonempty_value_count_first(s, "")),
        domain=("domain", lambda s: _mode_value_count_first(s, "")),
    )
    agg = qid_codes.groupby("msc_full", as_index=False).agg(freq=("qid", "nunique")).merge(meta, on="msc_full", how="left")
    agg["msc_l1"] = agg["msc_l1"].fillna("").astype(str).str.zfill(2).str[:2]
    agg["msc_l1"] = agg["msc_l1"].where(agg["msc_l1"].str.match(r"^\d{2}$", na=False), agg["msc_full"].str[:2])
    agg["msc_desc"] = agg["msc_desc"].fillna("").astype(str).str.strip()
    if msc_desc_map:
        miss_desc = agg["msc_desc"].str.len() == 0
        if miss_desc.any():
            agg.loc[miss_desc, "msc_desc"] = agg.loc[miss_desc, "msc_full"].map(msc_desc_map).fillna("")
    agg["domain"] = agg["domain"].fillna("").astype(str)
    miss_dom = agg["domain"].str.len() == 0
    agg.loc[miss_dom, "domain"] = agg.loc[miss_dom, "msc_l1"].map(_default_domain_from_l1)

    agg["concept"] = agg["msc_desc"].map(normalize_text)
    agg["concept"] = agg["concept"].where(agg["concept"].str.len() >= 3, agg["msc_full"].map(lambda x: f"msc {x}"))

    agg = apply_node_prune(agg, graph_cfg)
    agg["node_id"] = [f"N{i:07d}" for i in range(len(agg))]

    nodes = agg[["node_id", "msc_full", "msc_l1", "msc_desc", "concept", "domain", "freq"]].copy()

    node_map = nodes[["node_id", "msc_full"]]
    seed_links = qid_codes.merge(node_map, on="msc_full", how="inner")
    seed_links = seed_links[["node_id", "qid"]].drop_duplicates()
    seed_links = seed_links.merge(seed[["qid", "y", "subject", "level"]], on="qid", how="left")

    q2nodes = (
        seed_links.groupby("qid", as_index=False)
        .agg(node_ids=("node_id", lambda s: sorted(set(s))))
        .merge(seed[["qid", "subject", "y"]], on="qid", how="left")
    )

    topic_table = (
        nodes.groupby(["msc_full", "msc_l1", "domain"], as_index=False)
        .agg(node_count=("node_id", "nunique"), total_freq=("freq", "sum"))
        .sort_values(["node_count", "total_freq"], ascending=[False, False])
    )

    return nodes, seed_links, q2nodes, topic_table


def build_subnodes(
    mentions: pd.DataFrame,
    nodes_main: pd.DataFrame,
    graph_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not bool(graph_cfg.get("use_subnodes", False)):
        raise RuntimeError("subnodes disabled by config")

    max_sub = int(graph_cfg.get("subnode_max_per_msc", 3))
    min_concept_freq = int(graph_cfg.get("subnode_min_concept_freq", 2))
    top_concepts = int(graph_cfg.get("subnode_top_concepts", 80))
    if max_sub < 1:
        raise RuntimeError("subnode_max_per_msc must be >= 1")

    m = mentions.copy()
    m["concept"] = m["concept"].map(normalize_text)
    m = m[m["concept"].str.len() >= 3].copy()

    allowed_msc = set(nodes_main["msc_full"].astype(str).tolist())
    m = m[m["msc_full"].astype(str).isin(allowed_msc)].copy()

    if m.empty:
        raise RuntimeError("No mentions available for subnode clustering")

    # Concept frequency per MSC.
    cf = (
        m.groupby(["msc_full", "concept"], as_index=False)
        .agg(freq=("qid", "nunique"), domain=("domain", "first"))
        .sort_values(["msc_full", "freq"], ascending=[True, False])
    )
    cf = cf[cf["freq"].astype(int) >= min_concept_freq].copy()
    if cf.empty:
        raise RuntimeError("No concepts meet subnode_min_concept_freq")

    parent_map = dict(zip(nodes_main["msc_full"].astype(str), nodes_main["node_id"].astype(str)))
    domain_map = dict(zip(nodes_main["msc_full"].astype(str), nodes_main["domain"].astype(str)))

    subnodes_rows: list[dict] = []
    subnode_map_rows: list[dict] = []
    concept_to_sub: dict[tuple[str, str], str] = {}
    default_sub_for_msc: dict[str, str] = {}

    # Local import to avoid dependency if subnodes are disabled.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    for msc, group in cf.groupby("msc_full"):
        concepts = group["concept"].astype(str).tolist()
        if not concepts:
            continue

        k = min(max_sub, len(concepts))
        if len(concepts) <= max_sub:
            labels = list(range(len(concepts)))
        else:
            # Use top concepts for fitting, assign others by predict.
            top = group.sort_values("freq", ascending=False).head(top_concepts)
            fit_texts = top["concept"].astype(str).tolist()
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            x_fit = vec.fit_transform(fit_texts)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(x_fit)

            x_all = vec.transform(concepts)
            labels = km.predict(x_all).tolist()

        # Build cluster stats.
        cluster_rows: dict[int, list[tuple[str, int]]] = {}
        for concept, label in zip(concepts, labels):
            freq = int(group[group["concept"] == concept]["freq"].iloc[0])
            cluster_rows.setdefault(int(label), []).append((concept, freq))

        # Create subnodes for this MSC.
        for label, items in sorted(cluster_rows.items()):
            items = sorted(items, key=lambda x: (-x[1], x[0]))
            label_text = " | ".join([c for c, _ in items[:3]])
            subnodes_rows.append(
                {
                    "msc_full": msc,
                    "parent_node_id": parent_map.get(msc, ""),
                    "domain": domain_map.get(msc, ""),
                    "concept_cluster": label_text,
                    "cluster_id": int(label),
                    "freq": int(sum(f for _, f in items)),
                }
            )

        # assign subnode ids for this MSC
        # We'll assign after building full list.

    if not subnodes_rows:
        raise RuntimeError("No subnodes built")

    subnodes = pd.DataFrame(subnodes_rows).reset_index(drop=True)
    subnodes["subnode_id"] = [f"S{i:07d}" for i in range(len(subnodes))]

    # Build mapping (msc_full, cluster_id) -> subnode_id
    cluster_to_sub = {
        (str(r.msc_full), int(r.cluster_id)): str(r.subnode_id) for r in subnodes.itertuples(index=False)
    }

    # Compute default subnode per MSC (max freq).
    for msc, g in subnodes.groupby("msc_full"):
        pick = g.sort_values("freq", ascending=False).iloc[0]["subnode_id"]
        default_sub_for_msc[str(msc)] = str(pick)

    # Map each concept to a subnode (using same clustering process)
    # Re-run clustering assignment by using stored cluster_id from cf.
    # This is safe because we used deterministic kmeans with fixed random_state.
    for msc, group in cf.groupby("msc_full"):
        concepts = group["concept"].astype(str).tolist()
        if not concepts:
            continue
        k = min(max_sub, len(concepts))
        if len(concepts) <= max_sub:
            labels = list(range(len(concepts)))
        else:
            top = group.sort_values("freq", ascending=False).head(top_concepts)
            fit_texts = top["concept"].astype(str).tolist()
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            x_fit = vec.fit_transform(fit_texts)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(x_fit)
            x_all = vec.transform(concepts)
            labels = km.predict(x_all).tolist()

        for concept, label in zip(concepts, labels):
            sid = cluster_to_sub.get((str(msc), int(label)))
            if sid:
                concept_to_sub[(str(msc), str(concept))] = sid

    # Build subnode_map and seeds_sub_index.
    subnode_map_rows = [
        {
            "subnode_id": str(r.subnode_id),
            "parent_node_id": str(r.parent_node_id),
            "msc_full": str(r.msc_full),
            "domain": str(r.domain),
            "concept_cluster": str(r.concept_cluster),
            "freq": int(r.freq),
        }
        for r in subnodes.itertuples(index=False)
    ]

    # Map each mention row to subnode_id
    links = []
    for r in m.itertuples(index=False):
        key = (str(r.msc_full), str(r.concept))
        sid = concept_to_sub.get(key) or default_sub_for_msc.get(str(r.msc_full))
        if not sid:
            continue
        links.append({"subnode_id": sid, "qid": str(r.qid)})

    seeds_sub = pd.DataFrame(links).drop_duplicates()
    q2sub = seeds_sub.groupby("qid", as_index=False).agg(subnode_ids=("subnode_id", lambda s: sorted(set(s))))

    subnodes_out = subnodes[["subnode_id", "parent_node_id", "msc_full", "domain", "concept_cluster", "freq"]].copy()
    subnode_map = pd.DataFrame(subnode_map_rows)
    return subnodes_out, seeds_sub, q2sub, subnode_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    graph_cfg = cfg.get("graph", {})
    node_mode = str(graph_cfg.get("node_mode", "msc_concept")).strip().lower()

    interim_dir = Path(cfg["paths"]["interim_dir"])
    seed = load_df(str(interim_dir / "math_seed.parquet"))
    mentions = load_df(str(interim_dir / "concept_mentions.parquet"))
    concept_sets_path = interim_dir / "concept_sets.parquet"
    concept_sets = load_df(str(concept_sets_path)) if concept_sets_path.exists() else None
    msc_desc_map = _load_msc_desc_map(cfg)

    if node_mode == "msc_code":
        nodes, seed_links, q2nodes, topic_table = build_msc_code_graph(
            seed, mentions, concept_sets, graph_cfg, msc_desc_map=msc_desc_map
        )
    else:
        nodes, seed_links, q2nodes, topic_table = build_msc_concept_graph(seed, mentions, graph_cfg)

    out_graph = Path(cfg["paths"]["graph_dir"])
    save_df(nodes, str(out_graph / "nodes.parquet"))
    save_df(seed_links, str(out_graph / "seeds_index.parquet"))
    save_df(q2nodes, str(out_graph / "q2nodes.parquet"))
    save_df(topic_table, str(out_graph / "topics.parquet"))

    if bool(graph_cfg.get("use_subnodes", False)):
        subnodes, seeds_sub, q2sub, subnode_map = build_subnodes(mentions, nodes, graph_cfg)
        save_df(subnodes, str(out_graph / "subnodes.parquet"))
        save_df(seeds_sub, str(out_graph / "seeds_sub_index.parquet"))
        save_df(q2sub, str(out_graph / "q2subnodes.parquet"))
        save_df(subnode_map, str(out_graph / "subnode_map.parquet"))
        print(f"[OK] subnodes={len(subnodes)} seeds_sub={len(seeds_sub)} q2sub={len(q2sub)}")

    print(f"[OK] node_mode={node_mode} nodes={len(nodes)} seed_links={len(seed_links)} q2nodes={len(q2nodes)}")


if __name__ == "__main__":
    main()
