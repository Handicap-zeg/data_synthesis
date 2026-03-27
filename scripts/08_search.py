from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import pandas as pd


Sig = tuple[tuple[str, ...], tuple[str, ...]]


@dataclass
class Chain:
    nodes: list[str]
    edge_types: list[str]
    edge_confs: list[float]
    pre_count: int
    main_sem_used: int
    min_cooccur: float
    sum_cooccur: float
    max_d_sub: float

    @property
    def avg_cooccur(self) -> float:
        if not self.edge_types:
            return float("inf")
        return self.sum_cooccur / max(1, len(self.edge_types))


def _softmax_sample_index(scores: list[float], temperature: float, rng: random.Random) -> int:
    if not scores:
        return 0
    t = max(1e-3, float(temperature))
    mx = max(scores)
    exps = [math.exp((s - mx) / t) for s in scores]
    tot = sum(exps)
    if tot <= 0:
        return int(rng.randrange(len(scores)))
    r = rng.random() * tot
    acc = 0.0
    for i, v in enumerate(exps):
        acc += v
        if acc >= r:
            return i
    return len(scores) - 1


def _chain_coverage_gain(chain: Chain, node_cover_counts: dict[str, int]) -> float:
    unique_nodes = list(dict.fromkeys(chain.nodes))
    if not unique_nodes:
        return 0.0
    return float(sum(1.0 / math.sqrt(1.0 + float(node_cover_counts.get(sid, 0))) for sid in unique_nodes))


def _chain_sample_score(
    chain: Chain,
    sig: Sig,
    chain_usage: dict[Sig, int],
    recent_chain_counts: dict[Sig, int],
    subnode_parent: dict[str, str],
    node_cover_counts: dict[str, int],
) -> float:
    usage = float(chain_usage.get(sig, 0))
    recent = float(recent_chain_counts.get(sig, 0))
    conf_floor = min(chain.edge_confs) if chain.edge_confs else 0.0
    parents = {subnode_parent.get(sid, "") for sid in chain.nodes}
    parents.discard("")
    parent_cov = float(len(parents)) / max(1, len(chain.nodes))
    pre_ratio = float(chain.pre_count) / max(1, len(chain.edge_types))
    main_sem_bonus = 0.5 if chain.main_sem_used else 0.0
    coverage_gain = _chain_coverage_gain(chain, node_cover_counts)
    repeat_penalty = math.log1p(usage + 2.5 * recent)
    return float(coverage_gain + parent_cov + pre_ratio + main_sem_bonus + conf_floor - repeat_penalty)


def _build_adj(edges: pd.DataFrame) -> dict[str, list[tuple[str, float, float, str]]]:
    out: dict[str, list[tuple[str, float, float, str]]] = {}
    for r in edges.itertuples(index=False):
        src = str(r.src)
        dst = str(r.dst)
        conf = float(getattr(r, "conf", 0.5))
        co = float(getattr(r, "pair_cooccur", 0.0))
        parent = str(getattr(r, "parent_node_id", ""))
        out.setdefault(src, []).append((dst, conf, co, parent))
    for k in out:
        out[k].sort(key=lambda x: (x[1], -x[2]), reverse=True)
    return out


def _build_main_sem(edges: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for r in edges.itertuples(index=False):
        src = str(r.src)
        dst = str(r.dst)
        score = float(getattr(r, "bridge_best_score", 0.5))
        out.setdefault(src, []).append((dst, score))
    for k in out:
        out[k].sort(key=lambda x: x[1], reverse=True)
    return out


def _pick_bridge_subnode(
    parent_to_subs: dict[str, list[str]],
    sub_tokens: dict[str, set[str]],
    target_parent: str,
    ref_tokens: set[str],
) -> str | None:
    cands = parent_to_subs.get(target_parent, [])
    if not cands:
        return None
    best = None
    best_score = -1.0
    for sid in cands:
        toks = sub_tokens.get(sid, set())
        if not toks or not ref_tokens:
            score = 0.0
        else:
            inter = len(toks & ref_tokens)
            union = len(toks | ref_tokens)
            score = inter / union if union else 0.0
        if score > best_score:
            best_score = score
            best = sid
    return best


def _append_cover_edge(out: dict[str, list[str]], src: str, dst: str) -> None:
    if not src or not dst or src == dst:
        return
    bucket = out.setdefault(src, [])
    if dst not in bucket:
        bucket.append(dst)


def _invert_adj(adj: dict[str, list[str]]) -> dict[str, list[str]]:
    rev: dict[str, list[str]] = {}
    for src, dsts in adj.items():
        for dst in dsts:
            rev.setdefault(dst, []).append(src)
    return rev


def _build_cover_adjacency(
    adj_pre: dict[str, list[tuple[str, float, float, str]]],
    adj_sem: dict[str, list[tuple[str, float, float, str]]],
    adj_main: dict[str, list[tuple[str, float]]],
    subnodes_by_parent: dict[str, list[str]],
    sub_tokens: dict[str, set[str]],
    subnode_parent: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    forward: dict[str, list[str]] = {}
    for src, items in adj_pre.items():
        for dst, *_ in items:
            _append_cover_edge(forward, str(src), str(dst))
    for src, items in adj_sem.items():
        for dst, *_ in items:
            _append_cover_edge(forward, str(src), str(dst))
    for sid, parent in subnode_parent.items():
        if parent not in adj_main:
            continue
        for tgt_parent, _score in adj_main.get(parent, [])[:3]:
            dst = _pick_bridge_subnode(
                subnodes_by_parent,
                sub_tokens,
                tgt_parent,
                sub_tokens.get(sid, set()),
            )
            if dst:
                _append_cover_edge(forward, str(sid), str(dst))
    return forward, _invert_adj(forward)


def _limited_reverse_dist(reverse_adj: dict[str, list[str]], target_sid: str, max_depth: int) -> dict[str, int]:
    seen = {str(target_sid): 0}
    q = deque([(str(target_sid), 0)])
    while q:
        cur, dist = q.popleft()
        if dist >= max_depth:
            continue
        for prev in reverse_adj.get(cur, []):
            if prev in seen:
                continue
            seen[prev] = dist + 1
            q.append((prev, dist + 1))
    return seen


def _limited_forward_gain(
    forward_adj: dict[str, list[str]],
    start_sid: str,
    max_depth: int,
    node_cover_counts: dict[str, int],
) -> float:
    seen = {str(start_sid): 0}
    q = deque([(str(start_sid), 0)])
    gain = 0.0
    while q:
        cur, dist = q.popleft()
        depth_weight = 1.0 / max(1, dist + 1)
        gain += depth_weight / math.sqrt(1.0 + float(node_cover_counts.get(cur, 0)))
        if dist >= max_depth:
            continue
        for nxt in forward_adj.get(cur, []):
            if nxt in seen:
                continue
            seen[nxt] = dist + 1
            q.append((nxt, dist + 1))
    return float(gain)


def pick_start_for_coverage(
    all_subnode_ids: list[str],
    forward_adj: dict[str, list[str]],
    reverse_adj: dict[str, list[str]],
    node_cover_counts: dict[str, int],
    start_counts: dict[str, int],
    target_len: int,
    rng: random.Random,
    topk: int = 32,
    temperature: float = 0.7,
) -> str:
    if not all_subnode_ids:
        raise RuntimeError("no subnodes available")

    target_scores = [-0.5 * math.log1p(float(node_cover_counts.get(sid, 0))) for sid in all_subnode_ids]
    target_idx = _softmax_sample_index(target_scores, max(temperature, 0.35), rng)
    target_sid = str(all_subnode_ids[target_idx])

    rev_dist = _limited_reverse_dist(reverse_adj, target_sid, max(1, int(target_len) - 1))
    candidates = [sid for sid in rev_dist if sid != target_sid]
    if not candidates:
        candidates = [target_sid]

    scored: list[tuple[str, float]] = []
    for sid in candidates:
        gain = _limited_forward_gain(
            forward_adj=forward_adj,
            start_sid=sid,
            max_depth=max(1, int(target_len) - 1),
            node_cover_counts=node_cover_counts,
        )
        score = gain / math.sqrt(1.0 + float(start_counts.get(sid, 0)))
        scored.append((sid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: max(1, min(len(scored), int(topk)))]
    idx = _softmax_sample_index([x[1] for x in top], temperature, rng)
    return top[idx][0]


def build_chain(
    start_sid: str,
    target_len: int,
    beam_size: int,
    adj_pre: dict[str, list[tuple[str, float, float, str]]],
    adj_sem: dict[str, list[tuple[str, float, float, str]]],
    adj_main: dict[str, list[tuple[str, float]]],
    subnodes_by_parent: dict[str, list[str]],
    sub_tokens: dict[str, set[str]],
    sub_d: dict[str, float],
    subnode_parent: dict[str, str],
) -> Chain | None:
    beam: list[Chain] = [
        Chain([start_sid], [], [], 0, 0, float("inf"), 0.0, sub_d.get(start_sid, 0.5))
    ]
    for _ in range(target_len - 1):
        pool: list[Chain] = []
        for ch in beam:
            last = ch.nodes[-1]
            last_parent = str(subnode_parent.get(last, ""))
            for dst, conf, co, _ in adj_pre.get(last, []):
                if dst in ch.nodes:
                    continue
                pool.append(
                    Chain(
                        ch.nodes + [dst],
                        ch.edge_types + ["pre"],
                        ch.edge_confs + [conf],
                        ch.pre_count + 1,
                        ch.main_sem_used,
                        min(ch.min_cooccur, co),
                        ch.sum_cooccur + co,
                        max(ch.max_d_sub, sub_d.get(dst, 0.5)),
                    )
                )
            for dst, conf, co, _ in adj_sem.get(last, []):
                if dst in ch.nodes:
                    continue
                pool.append(
                    Chain(
                        ch.nodes + [dst],
                        ch.edge_types + ["sem"],
                        ch.edge_confs + [conf],
                        ch.pre_count,
                        ch.main_sem_used,
                        min(ch.min_cooccur, co),
                        ch.sum_cooccur + co,
                        max(ch.max_d_sub, sub_d.get(dst, 0.5)),
                    )
                )
            if ch.main_sem_used == 0 and last_parent in adj_main:
                for tgt_parent, score in adj_main.get(last_parent, [])[:3]:
                    dst = _pick_bridge_subnode(
                        subnodes_by_parent,
                        sub_tokens,
                        tgt_parent,
                        sub_tokens.get(last, set()),
                    )
                    if not dst or dst in ch.nodes:
                        continue
                    pool.append(
                        Chain(
                            ch.nodes + [dst],
                            ch.edge_types + ["main_sem"],
                            ch.edge_confs + [score],
                            ch.pre_count,
                            1,
                            ch.min_cooccur,
                            ch.sum_cooccur,
                            max(ch.max_d_sub, sub_d.get(dst, 0.5)),
                        )
                    )

        if not pool:
            break

        pool.sort(
            key=lambda c: (
                len(c.nodes),
                c.pre_count,
                c.main_sem_used,
                min(c.edge_confs) if c.edge_confs else 0.0,
                -c.avg_cooccur,
            ),
            reverse=True,
        )
        beam = pool[: max(1, int(beam_size))]

    if not beam:
        return None

    for c in beam:
        if len(c.nodes) >= 2:
            return c
    return None


def pick_chain_from_candidates(
    cand_bank: dict[Sig, Chain],
    chain_usage: dict[Sig, int],
    recent_chain_counts: dict[Sig, int],
    subnode_parent: dict[str, str],
    node_cover_counts: dict[str, int],
    rng: random.Random,
    topk: int,
    temperature: float,
) -> tuple[Sig, Chain] | None:
    if not cand_bank:
        return None
    scored: list[tuple[Sig, Chain, float]] = []
    for sig, cand in cand_bank.items():
        s = _chain_sample_score(
            chain=cand,
            sig=sig,
            chain_usage=chain_usage,
            recent_chain_counts=recent_chain_counts,
            subnode_parent=subnode_parent,
            node_cover_counts=node_cover_counts,
        )
        scored.append((sig, cand, s))
    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[: max(1, min(len(scored), int(topk)))]
    idx = _softmax_sample_index([x[2] for x in top], float(temperature), rng)
    sig, chain, _ = top[idx]
    return sig, chain
