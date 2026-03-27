from __future__ import annotations

import random
import re
from typing import Any, Iterable

from lib.common import normalize_text


def _tokenize(text: str) -> set[str]:
    t = normalize_text(text)
    t = re.sub(r"[^a-z0-9\\s]", " ", t)
    toks = [w for w in t.split() if len(w) >= 2]
    return set(toks)


def _level_num(x: Any) -> int:
    m = re.search(r"(\d+)", str(x or ""))
    return int(m.group(1)) if m else 0


def _seed_token_set(text: str) -> set[str]:
    stop = {
        "the",
        "that",
        "with",
        "from",
        "between",
        "prove",
        "find",
        "show",
        "determine",
        "compute",
        "classify",
        "integer",
        "integers",
        "number",
        "numbers",
        "given",
        "such",
        "have",
        "has",
        "are",
    }
    return {t for t in _tokenize(text) if t not in stop}


def _copy_score(problem: str, seed_problem: str) -> float:
    p = _seed_token_set(problem)
    s = _seed_token_set(seed_problem)
    if not p or not s:
        return 0.0
    inter = len(p & s)
    union = len(p | s)
    tok_j = inter / union if union else 0.0
    nums_p = set(re.findall(r"\d+", str(problem or "")))
    nums_s = set(re.findall(r"\d+", str(seed_problem or "")))
    num_j = (len(nums_p & nums_s) / len(nums_p | nums_s)) if (nums_p or nums_s) else 0.0
    return 0.75 * tok_j + 0.25 * num_j


def _max_seed_copy_score(problem: str, seed_records: list[dict[str, Any]]) -> tuple[float, str]:
    best = 0.0
    best_qid = ""
    for r in seed_records:
        sp = str(r.get("problem", ""))
        if not sp:
            continue
        sc = _copy_score(problem, sp)
        if sc > best:
            best = sc
            best_qid = str(r.get("qid", ""))
    return best, best_qid


def _select_anchor_contrast(
    seed_records: list[dict[str, Any]],
    rng: random.Random | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not seed_records:
        return None, None
    anchor = sorted(seed_records, key=lambda r: _level_num(r.get("level", "")), reverse=True)[0]
    if len(seed_records) == 1:
        return anchor, None
    a_prob = str(anchor.get("problem", ""))
    cands: list[tuple[dict[str, Any], float]] = []
    for r in seed_records:
        if str(r.get("qid", "")) == str(anchor.get("qid", "")):
            continue
        sc = _copy_score(a_prob, str(r.get("problem", "")))
        if sc >= 0.98:
            continue
        cands.append((r, sc))
    if not cands:
        return anchor, None
    medium = [x for x in cands if 0.25 <= x[1] <= 0.75]
    pool = medium if medium else cands
    pool = sorted(pool, key=lambda x: x[1])
    if rng is None:
        contrast = pool[len(pool) // 2][0]
    else:
        lo = len(pool) // 4
        hi = max(lo + 1, (3 * len(pool)) // 4)
        contrast = rng.choice(pool[lo:hi])[0] if hi > lo else pool[len(pool) // 2][0]
    return anchor, contrast


def _build_seed_role_pack(
    seed_records: list[dict[str, Any]],
    anchor_seed: dict[str, Any] | None,
    contrast_seed: dict[str, Any] | None,
    max_support: int = 3,
) -> dict[str, Any]:
    used = set()
    if anchor_seed:
        used.add(str(anchor_seed.get("qid", "")))
    if contrast_seed:
        used.add(str(contrast_seed.get("qid", "")))
    support: list[dict[str, Any]] = []
    for r in seed_records:
        qid = str(r.get("qid", ""))
        if not qid or qid in used:
            continue
        support.append(
            {
                "qid": qid,
                "level": str(r.get("level", "")),
                "role": "support",
                "focus": "supply one local transformation or closure step, but do not copy statement form",
            }
        )
        if len(support) >= max(0, int(max_support)):
            break
    anchor_obj = None
    if anchor_seed:
        anchor_obj = {
            "qid": str(anchor_seed.get("qid", "")),
            "level": str(anchor_seed.get("level", "")),
            "role": "anchor",
            "focus": "provide the global scaffold of the problem",
        }
    contrast_obj = None
    if contrast_seed:
        contrast_obj = {
            "qid": str(contrast_seed.get("qid", "")),
            "level": str(contrast_seed.get("level", "")),
            "role": "contrast",
            "focus": "change a key constraint, pivot, or target form",
        }
    return {
        "anchor": anchor_obj,
        "contrast": contrast_obj,
        "support_pool": support,
    }


def _default_mechanism_plan(
    chain_subnodes_info: list[dict[str, Any]],
    edge_types: list[str],
    required_ids: list[str],
) -> dict[str, Any]:
    roles = []
    for info in chain_subnodes_info:
        sid = str(info.get("subnode_id", ""))
        concept = str(info.get("concept_cluster", "")).split("|")[0].strip() or "core concept"
        roles.append(
            {
                "subnode_id": sid,
                "intermediate": f"Create one intermediate relation centered on {concept}",
                "contribution": "must be used in the next reasoning step",
            }
        )
    transitions = []
    for i, e in enumerate(edge_types):
        if i + 1 >= len(required_ids):
            break
        transitions.append(
            {
                "from_subnode": required_ids[i],
                "to_subnode": required_ids[i + 1],
                "edge_type": str(e),
                "transition": "state what new relation is transferred",
            }
        )
    return {
        "objective": "single-objective contest-style problem",
        "subnode_roles": roles,
        "transitions": transitions,
        "mutation_axes": [
            "change target form (value -> classification/proof or reverse)",
            "change at least one key constraint shape",
        ],
    }


def _generate_mechanism_plan(
    llm: Any,
    chain_subnodes_info: list[dict[str, Any]],
    edge_types: list[str],
    required_ids: list[str],
    seed_role_pack: dict[str, Any],
    ordered_seed_text: str,
) -> dict[str, Any]:
    fallback = _default_mechanism_plan(chain_subnodes_info, edge_types, required_ids)
    schema = """{
  "objective": "...",
  "subnode_roles": [{"subnode_id":"S...","intermediate":"...","contribution":"..."}],
  "transitions": [{"from_subnode":"S...","to_subnode":"S...","edge_type":"pre|sem|main_sem","transition":"..."}],
  "mutation_axes": ["..."]
}"""
    prompt = f"""
Build a compact mechanism plan before writing the final problem.

Chain subnodes:
{chain_subnodes_info}
Edge types:
{edge_types}
Required ids:
{required_ids}
Seed role pack:
{seed_role_pack}

Ordered chain seed dossier:
{ordered_seed_text}

Rules:
- read the chain seeds in node order before assigning intermediates
- single objective only
- each required subnode must provide one nontrivial intermediate
- transitions must explain how information moves between adjacent subnodes
- subnode_roles must form a dependency chain: each intermediate should be consumed by a later step
- assign different jobs to anchor / contrast / support seeds rather than repeating one generic role
- if a subnode were removed, the plan should lose a necessary intermediate rather than just a decorative fact
- avoid lexical padding; focus on mathematical operations

Return JSON:
{schema}
""".strip()
    try:
        out = llm.json_completion(
            system_prompt="You are a math mechanism planner. Return strict JSON only.",
            user_prompt=prompt,
            temperature=0.1,
        )
    except Exception:
        return fallback
    if not isinstance(out, dict):
        return fallback
    if not isinstance(out.get("subnode_roles", None), list):
        return fallback
    if not isinstance(out.get("transitions", None), list):
        return fallback
    return out


def _default_blueprint(
    chain_subnodes_info: list[dict[str, Any]],
    edge_types: list[str],
    anchor_seed: dict[str, Any] | None,
    contrast_seed: dict[str, Any] | None,
) -> dict[str, Any]:
    node_roles = []
    for info in chain_subnodes_info:
        sid = str(info.get("subnode_id", ""))
        concept = str(info.get("concept_cluster", ""))
        short_concept = concept.split("|")[0].strip() if concept else "core concept"
        node_roles.append({"subnode_id": sid, "role": f"Use {short_concept} as one necessary step."})
    edge_plan = []
    for i, e in enumerate(edge_types, 1):
        edge_plan.append({"step": i + 1, "edge_type": str(e), "how": f"Realize a {e} transition with explicit reasoning dependency."})
    changes = [
        "Change at least one numeric range/target from anchor seed.",
        "Introduce at least one extra constraint not present verbatim in any seed.",
    ]
    if contrast_seed:
        changes.append("Borrow one method motif from contrast seed but apply to a different objective form.")
    return {
        "objective": "One closed mathematical objective",
        "node_roles": node_roles,
        "edge_plan": edge_plan,
        "seed_changes": changes,
        "anti_copy_note": f"Do not restate seed {str(anchor_seed.get('qid','')) if anchor_seed else ''} with minor wording edits.",
    }


def _generate_chain_blueprint(
    llm: Any,
    chain_subnodes_info: list[dict[str, Any]],
    edge_types: list[str],
    anchor_seed: dict[str, Any] | None,
    contrast_seed: dict[str, Any] | None,
    ordered_seed_text: str,
) -> dict[str, Any]:
    fallback = _default_blueprint(chain_subnodes_info, edge_types, anchor_seed, contrast_seed)
    if not anchor_seed:
        return fallback
    schema = """{
  "objective": "...",
  "node_roles": [{"subnode_id":"S...","role":"..."}],
  "edge_plan": [{"step":2,"edge_type":"pre|sem|main_sem","how":"..."}],
  "seed_changes": ["change1","change2"],
  "anti_copy_note": "..."
}"""
    prompt = f"""
Design a compact generation blueprint from graph structure and seeds.

Chain nodes:
{chain_subnodes_info}
Edge types:
{edge_types}

Anchor seed:
[{anchor_seed.get('qid','')}] {anchor_seed.get('problem','')}

Contrast seed:
{f"[{contrast_seed.get('qid','')}] {contrast_seed.get('problem','')}" if contrast_seed else "(none)"}

Ordered chain seed dossier:
{ordered_seed_text}

Rules:
- read the node-local seeds in chain order and preserve their edge-conditioned dependencies
- Keep exactly one objective.
- Every required subnode must have a functional role.
- Each edge_type must have a reasoning transition.
- seed_changes must contain >=2 structural differences from anchor seed.
- node_roles should describe chained use, not independent fact drops.
- If contrast seed exists, it must change a pivot/constraint/target instead of only adding surface variation.
- The blueprint should make clear why removing one seed or one node would simplify the problem.
- Avoid lexical keyword constraints.
Return JSON:
{schema}
""".strip()
    try:
        obj = llm.json_completion(
            system_prompt="You are a math curriculum designer. Return strict JSON only.",
            user_prompt=prompt,
            temperature=0.1,
        )
    except Exception:
        return fallback
    if not isinstance(obj, dict):
        return fallback
    if not isinstance(obj.get("node_roles", None), list):
        return fallback
    if not isinstance(obj.get("edge_plan", None), list):
        return fallback
    return obj
