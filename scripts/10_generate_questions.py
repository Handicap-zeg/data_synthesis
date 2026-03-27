from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from lib.common import build_llm_client, load_config, load_df, normalize_text, set_seed


def _load_numbered_module(filename: str, module_name: str):
    base = Path(__file__).resolve().parent
    path = base / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_search08 = _load_numbered_module("08_search.py", "_synth08_search")
_mech09 = _load_numbered_module("09_mechanism.py", "_synth09_mechanism")

Chain = _search08.Chain
_build_adj = _search08._build_adj
_build_main_sem = _search08._build_main_sem
_build_cover_adjacency = _search08._build_cover_adjacency
_build_chain = _search08.build_chain
_pick_start_for_coverage = _search08.pick_start_for_coverage
_pick_chain_from_candidates = _search08.pick_chain_from_candidates

_copy_score = _mech09._copy_score
_max_seed_copy_score = _mech09._max_seed_copy_score
_select_anchor_contrast = _mech09._select_anchor_contrast
_build_seed_role_pack = _mech09._build_seed_role_pack
_generate_mechanism_plan = _mech09._generate_mechanism_plan
_generate_chain_blueprint = _mech09._generate_chain_blueprint


def _short(text: str, n: int) -> str:
    s = str(text or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _tokenize(text: str) -> set[str]:
    t = normalize_text(text)
    t = re.sub(r"[^a-z0-9\\s]", " ", t)
    toks = [w for w in t.split() if len(w) >= 2]
    return set(toks)


def _write_status_file(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


class _RollingJsonlWriter:
    def __init__(self, out_path: Path, shard_size: int = 100, start_index: int | None = None) -> None:
        self.shard_size = max(1, int(shard_size))
        self.out_dir = out_path.parent
        stem = str(out_path.stem)
        self.prefix = stem[:-4] if stem.endswith("_tmp") else stem
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index, self.current_count = self._discover_position(start_index=start_index)
        self.touched_shards: set[int] = set()
        self.total_appended = 0

    def _discover_position(self, start_index: int | None = None) -> tuple[int, int]:
        if start_index is not None:
            forced = max(1, int(start_index))
            forced_path = self.out_dir / f"{self.prefix}_{forced:02d}.jsonl"
            return forced, _count_jsonl_rows(forced_path)
        pat = re.compile(rf"^{re.escape(self.prefix)}_(\d+)\.jsonl$")
        found: list[tuple[int, Path]] = []
        for p in self.out_dir.glob(f"{self.prefix}_*.jsonl"):
            m = pat.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        if not found:
            return 1, 0
        found.sort(key=lambda x: x[0])
        last_idx, last_path = found[-1]
        last_count = _count_jsonl_rows(last_path)
        if last_count < self.shard_size:
            return last_idx, last_count
        return last_idx + 1, 0

    def current_path(self) -> Path:
        return self.out_dir / f"{self.prefix}_{self.shard_index:02d}.jsonl"

    def _sync_current_position(self) -> None:
        while True:
            path = self.current_path()
            actual_count = _count_jsonl_rows(path)
            if actual_count != self.current_count:
                self.current_count = actual_count
            if self.current_count < self.shard_size:
                return
            self.shard_index += 1
            self.current_count = 0

    def append_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._sync_current_position()
            path = self.current_path()
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.current_count += 1
            self.total_appended += 1
            self.touched_shards.add(self.shard_index)

    def touched_paths(self) -> list[str]:
        return [str(self.out_dir / f"{self.prefix}_{idx:02d}.jsonl") for idx in sorted(self.touched_shards)]


def _hard_ops_list(concepts: Iterable[str]) -> list[str]:
    base = {
        "case split",
        "contradiction",
        "invariant",
        "extremal",
        "pigeonhole",
        "bijection",
        "double counting",
        "inclusion-exclusion",
        "recurrence",
        "infinite descent",
        "symmetry",
        "bounding",
        "inequality",
        "substitution",
        "parameter",
        "construction",
        "分类讨论",
        "反证",
        "不变量",
        "极值",
        "抽屉原理",
        "双计数",
        "容斥",
        "递推",
        "无穷递降",
        "对称",
        "界",
        "不等式",
        "代换",
        "参数",
        "构造",
    }
    picked = set()
    for c in concepts:
        t = normalize_text(c)
        for op in list(base):
            if op in t:
                picked.add(op)
    if len(picked) < 4:
        picked |= set(list(base)[: max(4, len(picked))])
    return sorted(picked)


TEMPLATES_HARD = [
    ("H_exist", "Existence + construction: prove an object exists under given constraints and give an explicit construction."),
    ("H_class", "Classification + case split: classify all objects satisfying a property; split into at least two nontrivial cases."),
    ("H_ext", "Extremal + contradiction: assume a minimal/maximal counterexample and derive contradiction using an invariant or inequality."),
    ("H_count", "Counting + structure: count/estimate configurations using double counting or inclusion-exclusion; justify the formula."),
    ("H_rec", "Recurrence/process: define a sequence or process and prove a property via recurrence/invariant."),
]

TEMPLATES_MEDIUM = [
    ("M_calc", "Compute a target quantity under constraints; must use at least two given concepts."),
    ("M_solve", "Solve for all solutions of an equation/system with restrictions; justify each step."),
    ("M_count", "Count configurations using complement counting or a clean case split."),
    ("M_geom", "Find a value/ratio using geometric or trigonometric relations from the concepts."),
    ("M_seq", "Analyze a sequence/recurrence; find a term or a closed form using given concepts."),
]


def _pick_template(concepts: Iterable[str], level: str, rng: random.Random) -> tuple[str, str]:
    text = normalize_text(" ".join(str(c) for c in concepts))

    def has(*keys: str) -> bool:
        return any(k in text for k in keys)

    if level == "hard":
        if has("count", "combin", "pigeon", "double counting", "inclusion", "容斥", "双计数", "抽屉"):
            return TEMPLATES_HARD[3]
        if has("sequence", "recurrence", "递推", "数列"):
            return TEMPLATES_HARD[4]
        if has("inequality", "bound", "不等式", "界"):
            return TEMPLATES_HARD[2]
        if has("geometry", "triangle", "angle", "circle", "几何", "三角", "圆"):
            return TEMPLATES_HARD[1]
        return rng.choice(TEMPLATES_HARD)

    if level == "medium":
        if has("count", "combin", "pigeon", "double counting", "inclusion", "容斥", "双计数", "抽屉"):
            return TEMPLATES_MEDIUM[2]
        if has("sequence", "recurrence", "递推", "数列"):
            return TEMPLATES_MEDIUM[4]
        if has("geometry", "triangle", "angle", "circle", "几何", "三角", "圆", "sin", "cos", "tan"):
            return TEMPLATES_MEDIUM[3]
        return rng.choice(TEMPLATES_MEDIUM)

    return ("", "")


def _extract_anchor_phrases(concepts: Iterable[str], max_phrases: int = 8) -> list[str]:
    phrases: list[str] = []
    for c in concepts:
        if not c:
            continue
        parts = re.split(r"[|/;,，]+", str(c))
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) < 3 or len(p) > 40:
                continue
            phrases.append(p)
    # dedupe while keeping order
    seen = set()
    out = []
    for p in phrases:
        key = normalize_text(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= max_phrases:
            break
    return out


def _phrase_in_text(phrase: str, text: str) -> bool:
    if any("\u4e00" <= ch <= "\u9fff" for ch in phrase):
        return phrase in text
    return normalize_text(phrase) in normalize_text(text)


def _count_anchor_hits(problem: str, anchors: list[str]) -> int:
    if not anchors:
        return 0
    return sum(1 for a in anchors if _phrase_in_text(a, problem))


def _word_count(text: str) -> int:
    t = str(text or "").strip()
    if not t:
        return 0
    tokens = [w for w in re.split(r"\\s+", t) if w]
    if len(tokens) >= 2:
        return len(tokens)
    # Chinese fallback
    return len(re.findall(r"[\\u4e00-\\u9fff]", t))


def _fusion_stage_block(is_easy: bool, is_medium: bool, is_hard: bool) -> str:
    if is_hard:
        return """
Stage 1: Fusion sketch
- Assign distinct seed jobs: one scaffold seed, one pivot seed, and optionally one closure/support seed.
- The scaffold seed must determine the global problem form.
- The pivot seed must change a key constraint, target form, or proof bottleneck.
- Each required node must output one intermediate object/relation that is consumed by the next node.
- The chain must behave like state propagation, not a bag of parallel facts.
- For each planned seed/node, think what becomes easier or impossible if it is removed.

Stage 2: Problem writing
- Write one single-objective hard problem whose real difficulty comes from one of:
  coupled global constraints, hidden pivot lemma, reverse construction, extremal/minimal parameter, or nontrivial case split.
- The final problem must still need the scaffold seed, the pivot seed, and the node chain.
- Do not let any planned seed or node degrade into sentence-end decoration.
""".strip()
    if is_medium:
        return """
Stage 1: Fusion sketch
- Use two different seed jobs: one scaffold seed and one pivot seed.
- At least two required nodes must form a dependency chain: node A creates an intermediate, node B uses it.
- If one planned seed is removed, the problem should noticeably simplify.

Stage 2: Problem writing
- Write one single-objective medium problem with one clear nontrivial pivot.
- Allow a shorter chain and lighter proof burden than hard, but keep genuine seed/node fusion.
- Do not collapse the problem into direct formula substitution or one-step arithmetic.
""".strip()
    return """
Stage 1: Fusion sketch
- Use at most two seed ideas and at most two essential ideas in the final solution.
- Keep one clear scaffold; do not overload the problem with extra pivots.
- If a secondary seed or node is used, it should make one visible local change rather than add decorative text.

Stage 2: Problem writing
- Write one single-objective easy problem with short, clean reasoning.
- Keep the surface simple and the solution compact, but still avoid direct copying from any seed.
- Do not introduce fake complexity, stacked constraints, or long case splits.
""".strip()


def _seed_usage_prompt_block(is_easy: bool, is_medium: bool, is_hard: bool) -> str:
    if is_hard:
        return """
- assign distinct seed jobs: scaffold vs pivot vs optional closure
- contribution must name a concrete intermediate and what breaks if this seed is removed
- at least two seeds must be materially necessary for the final problem
""".strip()
    if is_medium:
        return """
- assign two different seed jobs: scaffold and pivot
- contribution must name a concrete intermediate and how the second seed changes the problem
""".strip()
    return """
- keep seed usage light: at most two seed ideas should matter
- contribution should state one visible local use, not generic inspiration
""".strip()


COUNT_METHOD_WORDS = [
    "double counting",
    "inclusion-exclusion",
    "pigeonhole",
    "bijection",
    "双计数",
    "容斥",
    "抽屉",
    "双射",
]

TEMPLATE_RULES_HARD = {
    "H_exist": {
        "require_groups": [
            ["exist", "exists", "存在", "构造", "construct"],
        ],
        "forbid": [],
    },
    "H_class": {
        "require_groups": [
            ["classify", "classification", "分类", "求所有", "all solutions", "all integers", "全部"],
        ],
        "forbid": [],
    },
    "H_ext": {
        "require_groups": [
            ["extremal", "极值", "最小", "最大", "minimal", "maximal"],
            ["contradiction", "反证", "invariant", "不变量", "bounding", "界", "inequality", "不等式"],
        ],
        "forbid": [],
    },
    "H_count": {
        # Keyword-hit gate removed by request.
        "require_groups": [],
        "forbid": [],
    },
    "H_rec": {
        "require_groups": [
            ["recurrence", "recursive", "sequence", "process", "递推", "数列", "递归", "过程"],
        ],
        "forbid": [],
    },
}


def _has_any(text: str, keys: Iterable[str]) -> bool:
    for k in keys:
        if _phrase_in_text(k, text):
            return True
    return False


def _check_template(problem: str, template_id: str) -> None:
    rules = TEMPLATE_RULES_HARD.get(template_id)
    if not rules:
        return
    # Template lexical hints are soft constraints; keep hard mathematical guards below.
    for group in rules.get("require_groups", []):
        _ = _has_any(problem, group)
    for k in rules.get("forbid", []):
        if _phrase_in_text(k, problem):
            raise RuntimeError(f"template forbidden keyword hit: {template_id}")
    if template_id != "H_count":
        _ = _has_any(problem, COUNT_METHOD_WORDS)

    # hard-clarity guards
    if not any(sym in problem for sym in ["=", "<", ">", "\\le", "\\ge", "≤", "≥"]):
        raise RuntimeError("problem missing explicit equation/inequality")
    if _has_any(problem, ["discuss", "explain", "describe", "讨论", "说明"]):
        raise RuntimeError("problem too open-ended for hard template")
    if template_id in ("H_exist", "H_class", "H_ext") and not _has_any(
        problem, ["such that", "满足", "使得"]
    ):
        pass


def _template_prompt_lines(template_id: str) -> list[str]:
    rules = TEMPLATE_RULES_HARD.get(template_id)
    if not rules:
        return []
    lines = []
    req = rules.get("require_groups", [])
    if req:
        for i, group in enumerate(req, 1):
            lines.append(f"- Must mention at least one of group {i}: {group}")
    if template_id != "H_count":
        lines.append(f"- Forbidden method words for this template: {COUNT_METHOD_WORDS}")
    return lines


TASK_WORDS_EN = [
    "prove",
    "show",
    "find",
    "determine",
    "compute",
    "classify",
    "count",
    "evaluate",
    "disprove",
]

TASK_WORDS_CN = [
    "证明",
    "求",
    "求出",
    "计算",
    "确定",
    "分类",
    "计数",
    "判断",
]

MULTI_TASK_CONNECTORS = [
    "additionally",
    "furthermore",
    "moreover",
    "in addition",
    "另外",
    "此外",
    "并且",
    "同时",
]

TASK_VERBS_EN = r"(prove|show|find|determine|compute|classify|count|evaluate|disprove|solve)"
TASK_VERBS_CN = r"(证明|求出|求|计算|确定|分类|计数|判断|解)"

DOMAIN_TOKENS = [
    "integer",
    "positive integer",
    "natural number",
    "real number",
    "sequence",
    "function",
    "polynomial",
    "matrix",
    "triangle",
    "graph",
    "整数",
    "正整数",
    "自然数",
    "实数",
    "序列",
    "函数",
    "多项式",
    "矩阵",
    "三角形",
    "图",
]


def _task_hits(problem: str) -> set[str]:
    p = normalize_text(problem)
    hits = {w for w in TASK_WORDS_EN if w in p}
    hits |= {w for w in TASK_WORDS_CN if w in problem}
    return hits


def _is_multi_task(problem: str) -> bool:
    p_raw = str(problem or "")
    if not p_raw.strip():
        return False
    p = normalize_text(p_raw)
    if p.count("?") >= 2 or p_raw.count("？") >= 2:
        return True
    clauses = re.split(
        r"[;；]|\badditionally\b|\bfurthermore\b|\bmoreover\b|\bin addition\b|另外|此外|并且|同时",
        p_raw,
        flags=re.IGNORECASE,
    )
    task_clause_cnt = 0
    for c in clauses:
        c = c.strip()
        if not c:
            continue
        if _task_hits(c):
            task_clause_cnt += 1
    if task_clause_cnt >= 2:
        return True
    if len(_task_hits(p_raw)) >= 2 and any(k in p for k in MULTI_TASK_CONNECTORS):
        return True
    # Stronger guard: two explicit task verbs connected in one statement.
    if re.search(rf"\b{TASK_VERBS_EN}\b.{{0,220}}\b(and|then)\b.{{0,220}}\b{TASK_VERBS_EN}\b", p):
        return True
    # Chinese joint-task patterns like “求...并计算...” / “证明...并求...”.
    if re.search(rf"{TASK_VERBS_CN}.{{0,120}}(并|并且|且|同时|再).{{0,120}}{TASK_VERBS_CN}", p_raw):
        return True
    # Special hard-fail phrase that often indicates dual objectives.
    if re.search(r"\bfind\b.{0,220}\band\s+compute\b", p):
        return True
    if " and compute " in p and any(f" {v} " in p for v in ["find", "determine", "solve", "prove", "show", "count", "classify", "evaluate"]):
        return True
    return False



def _relation_count(problem: str) -> int:
    p = str(problem or "")
    # Count formal constraints, not plain vertical bars.
    pats = [
        r"\\le",
        r"\\ge",
        r"≤",
        r"≥",
        r"\\equiv",
        r"\\mid",
        r"=",
        r"<",
        r">",
    ]
    return sum(len(re.findall(pt, p)) for pt in pats)


def _has_constraint_system(problem: str) -> bool:
    p = str(problem or "")
    rel_cnt = _relation_count(p)
    cond_markers = [
        "such that",
        "s.t.",
        "where",
        "given",
        "with",
        "满足",
        "使得",
        "其中",
        "已知",
    ]
    has_cond = _has_any(p, cond_markers)
    has_domain = _has_any(p, DOMAIN_TOKENS)
    if rel_cnt >= 2 and has_domain:
        return True
    if rel_cnt >= 1 and has_cond and has_domain:
        return True
    return False


def _check_hard_core(problem: str) -> None:
    if _is_multi_task(problem):
        raise RuntimeError("multi-task objective detected")


def _problem_has_form(problem: str) -> bool:
    p = str(problem or "")
    keys = [
        "证明",
        "求所有",
        "分类",
        "极值",
        "最大",
        "最小",
        "不存在",
        "存在",
        "prove",
        "classify",
        "extremal",
        "contradiction",
        "exist",
    ]
    return any(k in p for k in keys)


def _count_hard_ops(problem: str, hard_ops: list[str]) -> int:
    p = normalize_text(problem)
    return sum(1 for op in hard_ops if normalize_text(op) in p)


def _extract_problem_keywords(problem: str) -> set[str]:
    p = str(problem or "")
    sym = set(re.findall(r"[a-zA-Z](?:_?\\d+)?", p))
    cn_terms = [w for w in re.split(r"[^\\u4e00-\\u9fff]+", p) if len(w) >= 2]
    drop = {
        "证明",
        "求",
        "求解",
        "已知",
        "给定",
        "设",
        "令",
        "若",
        "则",
        "使得",
        "存在",
        "对于",
        "对所有",
        "判断",
        "是否",
        "多少",
        "最小",
        "最大",
        "分类",
        "极值",
    }
    cn_terms = [w for w in cn_terms if w not in drop]
    return set(sym) | set(cn_terms)


def _problem_self_check(problem: str) -> None:
    p = str(problem or "")
    if len(p) < 8:
        raise RuntimeError("problem too short")


def _ensure_outline_align(problem: str, outline: Any) -> None:
    keys = _extract_problem_keywords(problem)
    if not keys:
        return
    for step in outline if isinstance(outline, list) else []:
        s = normalize_text(str(step))
        if not any(normalize_text(k) in s for k in keys):
            raise RuntimeError("solution_outline step not aligned with problem text")

def _ensure_min_steps(outline: Any, min_steps: int) -> None:
    if not isinstance(outline, list) or len(outline) < int(min_steps):
        raise RuntimeError(f"solution_outline too short: {len(outline) if isinstance(outline, list) else 0}")


def _ensure_tags(outline: Any, required_ids: list[str]) -> None:
    if not isinstance(outline, list) or not outline:
        raise RuntimeError("solution_outline missing or not a list")
    text = " ".join(str(s) for s in outline)
    missing = [nid for nid in required_ids if nid not in text]
    if missing:
        raise RuntimeError(f"solution_outline missing node tags: {missing}")


def _missing_tags(outline: Any, required_ids: list[str]) -> list[str]:
    if not isinstance(outline, list) or not outline:
        return list(required_ids)
    text = " ".join(str(s) for s in outline)
    return [nid for nid in required_ids if nid not in text]


def _append_missing_tags(outline: Any, missing: list[str]) -> list[str]:
    steps = list(outline) if isinstance(outline, list) else []
    for nid in missing:
        steps.append(f"[NODE:{nid}] Apply the concept tied to this node.")
    return steps


def _answers_match(a: str, b: str) -> bool:
    na = normalize_text(a or "")
    nb = normalize_text(b or "")
    if not na or not nb:
        return False
    if na == nb:
        return True
    # simple numeric equality
    try:
        fa = float(eval(na))
        fb = float(eval(nb))
        return abs(fa - fb) <= 1e-6
    except Exception:
        return False


def _level_num(x: Any) -> int:
    m = re.search(r"(\d+)", str(x or ""))
    return int(m.group(1)) if m else 0


def _generate_formal_spec(
    llm: Any,
    chain_subnodes_info: list[dict[str, Any]],
    required_ids: list[str],
    seed_text: str,
    template_line: str,
    is_hard: bool,
) -> dict[str, Any]:
    schema = """{
  "objective_count": 1,
  "task_type": "compute|prove|classify|existence|extremal",
  "proof_mode": "none|contradiction|case_split",
  "pivot": "key contradiction point or case split trigger",
  "domain": "short domain statement",
  "variables": ["..."],
  "givens": ["..."],
  "constraints": ["..."],
  "goal": "single explicit goal",
  "target_form": "integer|fraction|set|proof statement",
  "required_nodes": ["S..."],
  "notes": "short note"
}"""
    hard_line = (
        "- Hard spec must include at least 2 explicit mathematical constraints and a nontrivial goal."
        if is_hard
        else "- Medium spec should include at least 1 explicit mathematical constraint."
    )
    user_prompt = f"""
You are building a formal problem spec from graph nodes.

Chain subnodes info:
{chain_subnodes_info}

Seed examples:
{seed_text}

{template_line}

Output JSON schema:
{schema}

Rules:
- objective_count MUST be exactly 1.
- required_nodes must include all: {required_ids}
{hard_line}
- Constraints must be mathematical statements (equation/inequality/divisibility/range), not generic prose.
- Goal must be decidable and closed-form (no open discussion tasks).
- For hard: proof_mode must be contradiction or case_split, and pivot must be concrete.
""".strip()
    spec = llm.json_completion(
        system_prompt="You are a rigorous math spec designer. Return strict JSON only.",
        user_prompt=user_prompt,
        temperature=0.1,
    )
    return spec if isinstance(spec, dict) else {}


def _validate_formal_spec(spec: dict[str, Any], required_ids: list[str], is_hard: bool) -> None:
    if not isinstance(spec, dict):
        raise RuntimeError("spec is not an object")
    if int(spec.get("objective_count", 0)) != 1:
        raise RuntimeError("spec objective_count must be 1")
    goal = str(spec.get("goal", "")).strip()
    if len(goal) < 8:
        raise RuntimeError("spec goal too short")
    constraints = spec.get("constraints", [])
    if not isinstance(constraints, list) or not constraints:
        raise RuntimeError("spec constraints missing")
    min_constraints = 2 if is_hard else 1
    if len(constraints) < min_constraints:
        raise RuntimeError("spec constraints too few")
    c_join = " ".join(str(x) for x in constraints)
    rel_cnt = _relation_count(c_join)
    if rel_cnt < min_constraints:
        c_norm = normalize_text(c_join)
        semantic_formal = any(
            k in c_norm
            for k in [
                "mod",
                "congr",
                "divis",
                "integer",
                "nonzero",
                "parity",
                "coprime",
                "inequal",
                "bound",
                "prime",
                "同余",
                "整除",
                "整数",
                "非零",
                "奇偶",
                "互素",
                "不等式",
                "界",
                "素数",
            ]
        )
        if not semantic_formal and not is_hard:
            raise RuntimeError("spec constraints not mathematical enough")
    req_nodes = spec.get("required_nodes", [])
    if not isinstance(req_nodes, list):
        raise RuntimeError("spec required_nodes invalid")
    req_set = {str(x) for x in req_nodes}
    miss = [x for x in required_ids if x not in req_set]
    if miss:
        raise RuntimeError(f"spec missing required nodes: {miss}")
    if _is_multi_task(goal):
        raise RuntimeError("spec goal appears multi-task")
    if is_hard:
        mode = str(spec.get("proof_mode", "")).strip().lower()
        if mode not in {"contradiction", "case_split"}:
            # Soft for hard: keep formal spec stage, but avoid over-rejecting compact specs.
            spec["proof_mode"] = "none"
        pivot = str(spec.get("pivot", "")).strip()
        if len(pivot) < 10:
            spec["pivot"] = ""
        # Hard goal must not be a near-restatement of given constraints.
        c_text = " ".join(str(x) for x in constraints)
        if normalize_text(goal) in normalize_text(c_text):
            pass


def _check_hard_pivot(problem: str, spec: dict[str, Any]) -> None:
    mode = str(spec.get("proof_mode", "")).strip().lower()
    pivot = str(spec.get("pivot", "")).strip()
    if mode == "contradiction":
        _ = _has_any(problem, ["contradiction", "assume", "反证", "假设"])
    elif mode == "case_split":
        _ = _has_any(problem, ["case", "cases", "分类", "情形"])
    # Ensure pivot is not dropped when rendering final problem statement.
    _ = pivot and _count_anchor_hits(problem, [pivot]) >= 1


def _generate_lemma_plan(
    llm: Any,
    spec: dict[str, Any],
    required_ids: list[str],
    chain_subnodes_info: list[dict[str, Any]],
) -> dict[str, Any]:
    schema = """{
  "lemmas": [
    {"id": "L1", "from_node": "S...", "claim": "...", "purpose": "..."}
  ],
  "final_assembly": "how lemmas combine to solve the goal"
}"""
    user_prompt = f"""
Build a hard-problem lemma plan from the formal spec.

Formal spec:
{spec}

Chain subnodes info:
{chain_subnodes_info}

Output JSON schema:
{schema}

Rules:
- Use at least {max(3, len(required_ids))} lemmas.
- Every required node must appear in at least one lemma: {required_ids}
- Each lemma claim must be mathematically concrete (not motivational text).
- final_assembly must explicitly state how the single goal is concluded.
""".strip()
    plan = llm.json_completion(
        system_prompt="You are a competition-math proof planner. Return strict JSON only.",
        user_prompt=user_prompt,
        temperature=0.1,
    )
    return plan if isinstance(plan, dict) else {}


def _validate_lemma_plan(plan: dict[str, Any], required_ids: list[str], spec: dict[str, Any] | None = None) -> None:
    if not isinstance(plan, dict):
        raise RuntimeError("lemma plan is not an object")
    lemmas = plan.get("lemmas", [])
    if not isinstance(lemmas, list) or len(lemmas) < max(3, len(required_ids)):
        raise RuntimeError("lemma plan too short")
    covered = set()
    novel_claims = 0
    spec_constraints = []
    if isinstance(spec, dict):
        spec_constraints = spec.get("constraints", []) if isinstance(spec.get("constraints", []), list) else []
    c_text = normalize_text(" ".join(str(x) for x in spec_constraints))
    for it in lemmas:
        if not isinstance(it, dict):
            raise RuntimeError("invalid lemma entry")
        sid = str(it.get("from_node", "")).strip()
        claim = str(it.get("claim", "")).strip()
        purpose = str(it.get("purpose", "")).strip()
        if sid not in required_ids:
            raise RuntimeError(f"lemma from_node out of chain: {sid}")
        if len(claim) < 12 or len(purpose) < 6:
            raise RuntimeError("lemma too vague")
        if (not c_text) or (normalize_text(claim) not in c_text):
            novel_claims += 1
        covered.add(sid)
    miss = [x for x in required_ids if x not in covered]
    if miss:
        raise RuntimeError(f"lemma plan missing nodes: {miss}")
    if len(str(plan.get("final_assembly", "")).strip()) < 12:
        raise RuntimeError("lemma final_assembly too short")
    if novel_claims < 2:
        raise RuntimeError("lemma plan lacks nontrivial intermediate claims")


def _check_spec_grounding(problem: str, spec: dict[str, Any]) -> None:
    spec_text = " ".join(
        [
            str(spec.get("goal", "")),
            " ".join(str(x) for x in spec.get("variables", []) if x is not None),
            " ".join(str(x) for x in spec.get("constraints", []) if x is not None),
        ]
    )
    spec_toks = {t for t in _tokenize(spec_text) if t not in {"such", "that", "where", "given", "prove"}}
    prob_toks = _tokenize(problem)
    if not spec_toks:
        return
    hit = len(spec_toks & prob_toks)
    need = max(4, min(12, len(spec_toks) // 6))
    if hit < need:
        raise RuntimeError("problem not grounded enough in formal spec")


def _check_lemma_grounding(outline: Any, plan: dict[str, Any]) -> None:
    text = normalize_text(" ".join(str(s) for s in outline if isinstance(outline, list)))
    lemmas = plan.get("lemmas", []) if isinstance(plan, dict) else []
    if not lemmas:
        return
    hit = 0
    for l in lemmas:
        claim = normalize_text(str(l.get("claim", "")))
        toks = [t for t in claim.split() if len(t) >= 4]
        if not toks:
            continue
        if any(t in text for t in toks[:3]):
            hit += 1
    if hit < max(2, len(lemmas) // 2):
        raise RuntimeError("solution_outline does not reflect lemma plan")


def _check_proof_mode_realized(outline: Any, spec: dict[str, Any]) -> None:
    mode = str(spec.get("proof_mode", "")).strip().lower()
    text = str(" ".join(str(s) for s in outline if isinstance(outline, list)))
    t = normalize_text(text)
    if mode == "contradiction":
        _ = any(k in t for k in ["contradiction", "assume", "suppose", "反证", "假设"])
    elif mode == "case_split":
        marks = ["case 1", "case 2", "cases", "情形一", "情形二", "分类讨论", "case i", "case ii"]
        cnt = sum(1 for k in marks if k in t)
        _ = cnt >= 2


HARD_STRUCT_TOKENS = [
    "for all",
    "exists",
    "mod",
    "divides",
    "integer",
    "inequality",
    "recurrence",
    "distinct",
    "nonzero",
    "positive integer",
    "对所有",
    "存在",
    "同余",
    "整除",
    "不等式",
    "递推",
    "互异",
    "正整数",
]


def _check_hard_nontrivial(problem: str, outline: Any) -> None:
    t = normalize_text(problem)
    hit = sum(1 for k in HARD_STRUCT_TOKENS if normalize_text(k) in t)
    if hit < 2:
        raise RuntimeError("hard problem lacks structural math constraints")
    if isinstance(outline, list):
        if len(outline) < 5:
            raise RuntimeError("hard outline too short")
        outline_text = normalize_text(" ".join(str(x) for x in outline))
        if any(k in outline_text for k in ["directly", "trivial", "obvious", "显然"]):
            raise RuntimeError("hard outline appears trivialized")


def _collect_chain_seed_records(
    chain_subnodes: list[str],
    chain_subnodes_info: list[dict[str, Any]],
    sub_to_qids: dict[str, list[str]],
    qid_to_seed: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    info_by_sid = {
        str(x.get("subnode_id", "")): x
        for x in chain_subnodes_info
        if str(x.get("subnode_id", ""))
    }
    rows: list[dict[str, Any]] = []
    seen = set()
    for sid in chain_subnodes:
        qids = sub_to_qids.get(sid, [])
        qids_sorted = sorted(
            qids,
            key=lambda q: _level_num(qid_to_seed.get(str(q), {}).get("level", "")),
            reverse=True,
        )
        meta = info_by_sid.get(str(sid), {})
        for qid in qids_sorted:
            q = str(qid)
            if not q or q in seen:
                continue
            row = qid_to_seed.get(q, {})
            if not row:
                continue
            seen.add(q)
            rows.append(
                {
                    "qid": q,
                    "level": str(row.get("level", "")),
                    "problem": str(row.get("problem", "")),
                    "solution": str(row.get("solution", "")),
                    "subnode_id": str(sid),
                    "parent_node_id": str(meta.get("parent_node_id", "")),
                    "concept_cluster": str(meta.get("concept_cluster", "")),
                }
            )
    return rows


def _select_prompt_seed_records(
    seed_records: list[dict[str, Any]],
    rng: random.Random,
    per_node_cap: int = 6,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in seed_records:
        sid = str(r.get("subnode_id", ""))
        grouped.setdefault(sid, []).append(r)
    out: list[dict[str, Any]] = []
    for sid, rows in grouped.items():
        by_level: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            lv = _level_num(r.get("level", ""))
            by_level.setdefault(lv, []).append(r)
        picked_qids: set[str] = set()
        chosen: list[dict[str, Any]] = []
        for lv in range(5, 0, -1):
            bucket = by_level.get(lv, [])
            if not bucket:
                continue
            cand = sorted(bucket, key=lambda x: str(x.get("qid", "")))[0]
            qid = str(cand.get("qid", ""))
            if qid and qid not in picked_qids:
                chosen.append(cand)
                picked_qids.add(qid)
            if len(chosen) >= int(per_node_cap):
                break
        if len(chosen) < int(per_node_cap):
            leftovers = [r for r in rows if str(r.get("qid", "")) not in picked_qids]
            rng.shuffle(leftovers)
            need = int(per_node_cap) - len(chosen)
            chosen.extend(leftovers[:need])
        out.extend(chosen[: int(per_node_cap)])
    return out


def _seeds_to_text(seed_records: list[dict[str, Any]]) -> str:
    if not seed_records:
        return "(no examples found)"
    blocks = []
    for r in seed_records:
        blocks.append(
            f"[{r.get('qid','')}] ({r.get('level','')})\n"
            f"- Problem: {_short(r.get('problem',''), 650)}\n"
            f"- Solution: {_short(r.get('solution',''), 650)}"
        )
    return "\n\n".join(blocks)


def _edge_relation_hint(edge_type: str) -> str:
    et = str(edge_type or "")
    if et == "pre":
        return "The downstream node should consume an intermediate, lemma, or constraint produced by the upstream node."
    if et == "main_sem":
        return "This edge is a semantic bridge across parent concepts; make the bridge mathematically necessary, not decorative."
    if et == "sem":
        return "Transfer a method or representation change across adjacent nodes instead of restarting from a parallel fact."
    return "Preserve the local dependency between adjacent nodes."


def _ordered_chain_seed_prompt(
    required_ids: list[str],
    edge_types: list[str],
    chain_subnodes_info: list[dict[str, Any]],
    seed_records: list[dict[str, Any]],
) -> str:
    if not seed_records:
        return "(no seeds found on this chain)"
    info_by_sid = {
        str(x.get("subnode_id", "")): x
        for x in chain_subnodes_info
        if str(x.get("subnode_id", ""))
    }
    seeds_by_sid: dict[str, list[dict[str, Any]]] = {}
    for r in seed_records:
        sid = str(r.get("subnode_id", ""))
        seeds_by_sid.setdefault(sid, []).append(r)
    blocks: list[str] = []
    for i, sid in enumerate(required_ids):
        info = info_by_sid.get(str(sid), {})
        lines = [
            f"Node {i + 1}: {sid}",
            f"- parent: {info.get('parent_node_id', '')}",
            f"- concepts: {info.get('concept_cluster', '')}",
        ]
        if i == 0:
            lines.append("- incoming relation: start")
        else:
            edge_type = str(edge_types[i - 1]) if i - 1 < len(edge_types) else ""
            prev_sid = required_ids[i - 1]
            lines.append(f"- incoming relation from {prev_sid}: {edge_type}")
            lines.append(f"- relation meaning: {_edge_relation_hint(edge_type)}")
        local = seeds_by_sid.get(str(sid), [])
        if not local:
            lines.append("- local seeds: (none)")
        else:
            lines.append(f"- selected local seeds ({len(local)} shown; use 6 per node when available, otherwise include all):")
            for r in local:
                lines.append(f"  [{r.get('qid','')}] ({r.get('level','')}) problem: {_short(r.get('problem',''), 420)}")
                lines.append(f"  solution cue: {_short(r.get('solution',''), 260)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)



def _outline_has_placeholder(outline: Any) -> bool:
    text = normalize_text(" ".join(str(s) for s in outline if isinstance(outline, list)))
    bad = ["[node:", "apply the concept tied to this node", "placeholder", "todo"]
    return any(k in text for k in bad)


def _concept_tokens(concept_cluster: str) -> set[str]:
    parts = []
    for x in re.split(r"[|/;,，]+", str(concept_cluster or "")):
        x = x.strip()
        if x:
            parts.append(x)
    toks = set()
    for p in parts[:3]:
        for t in _tokenize(p):
            if len(t) >= 4:
                toks.add(t)
    return toks


def _collect_outline_issues(
    problem: str,
    outline: Any,
    required_ids: list[str],
    chain_subnodes_info: list[dict[str, Any]],
    seed_records: list[dict[str, Any]],
    is_hard: bool,
    history_problems: list[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(outline, list) or not outline:
        return ["solution_outline missing or empty"]
    if _outline_has_placeholder(outline):
        issues.append("outline contains placeholder steps")
    raw = " ".join(str(s) for s in outline)
    for sid in required_ids:
        if sid not in raw:
            issues.append(f"missing required node tag: {sid}")
    concept_by_sid = {
        str(x.get("subnode_id", "")): _concept_tokens(str(x.get("concept_cluster", "")))
        for x in chain_subnodes_info
    }
    for sid in required_ids:
        toks = concept_by_sid.get(sid, set())
        found = False
        for step in outline:
            st = ""
            ssid = ""
            if isinstance(step, dict):
                st = normalize_text(str(step.get("statement", "")))
                ssid = str(step.get("subnode_id", ""))
            else:
                st = normalize_text(str(step))
                ssid = sid if sid in str(step) else ""
            if ssid != sid:
                continue
            if not toks:
                found = True
                break
            if any(t in st for t in toks):
                found = True
                break
        if False and is_hard and (not found):
            issues.append(f"node {sid} not functionally used with its concept")
    copy_score, copy_qid = _max_seed_copy_score(problem, seed_records)
    th = 0.84 if is_hard else 0.90
    if copy_score >= th and copy_qid:
        issues.append(f"near-copy risk from seed {copy_qid} (score={copy_score:.2f})")
    if history_problems:
        hist_best = 0.0
        for hp in history_problems:
            hist_best = max(hist_best, _copy_score(problem, hp))
        if hist_best >= 0.90:
            issues.append(f"near-duplicate risk with generated history (score={hist_best:.2f})")
    return issues


def _rewrite_with_structure(
    llm: Any,
    obj: dict[str, Any],
    issues: list[str],
    required_ids: list[str],
    chain_subnodes_info: list[dict[str, Any]],
    edge_types: list[str],
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    schema = """{
  "problem": "...",
  "answer": "...",
  "seed_usage_plan": [{"seed_qid":"MATH_...","role":"anchor|contrast|support","target_subnode_id":"S...","used_in_step":2,"contribution":"..."}],
  "solution_outline": [{"step_id":1,"subnode_id":"S...","concept":"...","edge_type_from_prev":"start|pre|sem|main_sem","statement":"..."}],
  "seed_mapping": [{"from_seed":"MATH_...","reused_structure":"...","applied_to":"..."}],
  "used_nodes": ["..."],
  "used_nodes_info": [{"subnode_id":"...","parent_node_id":"...","concept_cluster":"..."}]
}"""
    prompt = f"""
Rewrite the generated math item to fix structural issues.

Issues:
{issues}

Required node ids:
{required_ids}
Edge types:
{edge_types}
Chain node info:
{chain_subnodes_info}
Blueprint:
{blueprint}

Current object:
{obj}

Rules:
- Keep one objective.
- No placeholder text like [NODE:...].
- Every required node must appear with a real reasoning step.
- Keep seed_mapping concrete and nontrivial.
Return JSON:
{schema}

Before writing, check that the set of returned slot_id values is exactly the slot plan and the item count is exactly {total_questions}.
""".strip()
    try:
        out = llm.json_completion(
            system_prompt="You are a math editor. Return strict JSON only.",
            user_prompt=prompt,
            temperature=0.15,
        )
    except Exception:
        return obj
    if not isinstance(out, dict):
        return obj
    for k in ("problem", "answer", "seed_usage_plan", "solution_outline", "seed_mapping"):
        if k not in out:
            out[k] = obj.get(k)
    if "used_nodes" not in out:
        out["used_nodes"] = obj.get("used_nodes", required_ids)
    if "used_nodes_info" not in out:
        out["used_nodes_info"] = obj.get("used_nodes_info", chain_subnodes_info)
    return out


def _generate_seed_skeleton(llm: Any, seed_records: list[dict[str, Any]], is_hard: bool) -> dict[str, Any]:
    schema = """{
  "core_task": "short task type",
  "domain": "math domain",
  "variable_roles": ["..."],
  "must_have_constraints": ["..."],
  "key_lemmas": ["..."],
  "method_tags": ["..."],
  "prohibited_shortcuts": ["..."],
  "reference_seeds": ["MATH_..."]
}"""
    mode_line = (
        "Hard mode: extract at least 2 nontrivial constraints and at least 2 lemma-level intermediate claims."
        if is_hard
        else "Medium mode: extract at least 1 explicit constraint and 1 intermediate claim."
    )
    prompt = f"""
Given the seed problems below, build a reusable structural skeleton.

Seeds:
{_seeds_to_text(seed_records)}

Return JSON:
{schema}

Rules:
- reference_seeds must be chosen from provided qids.
- must_have_constraints should be formal math constraints, not generic wording.
- key_lemmas should be intermediate mathematical statements, not final answer restatements.
- {mode_line}
""".strip()
    obj = llm.json_completion(
        system_prompt="You are a math abstraction engine. Return strict JSON only.",
        user_prompt=prompt,
        temperature=0.1,
    )
    return obj if isinstance(obj, dict) else {}


def _validate_seed_skeleton(
    sk: dict[str, Any], seed_qids: set[str], is_medium: bool, is_hard: bool
) -> None:
    if not isinstance(sk, dict):
        raise RuntimeError("seed_skeleton invalid")
    refs = sk.get("reference_seeds", [])
    if not isinstance(refs, list):
        raise RuntimeError("seed_skeleton reference_seeds invalid")
    refs = [str(x) for x in refs if str(x)]
    if is_hard and len(refs) < 2:
        raise RuntimeError("hard seed_skeleton needs >=2 reference seeds")
    if is_medium and len(refs) < 1:
        raise RuntimeError("medium seed_skeleton needs >=1 reference seed")
    bad = [q for q in refs if q not in seed_qids]
    if bad:
        raise RuntimeError(f"seed_skeleton contains unknown seed ids: {bad}")
    cons = sk.get("must_have_constraints", [])
    lem = sk.get("key_lemmas", [])
    if not isinstance(cons, list) or not cons:
        raise RuntimeError("seed_skeleton missing must_have_constraints")
    if not isinstance(lem, list) or not lem:
        raise RuntimeError("seed_skeleton missing key_lemmas")
    if is_hard and len(cons) < 2:
        raise RuntimeError("hard seed_skeleton constraints too few")
    if is_hard and len(lem) < 2:
        raise RuntimeError("hard seed_skeleton lemmas too few")
    # must include formal relation: symbolic OR semantic (e.g., congruence/divisibility/integer-domain)
    c_join = " ".join(str(x) for x in cons)
    rel_cnt = _relation_count(c_join)
    min_rel = 2 if is_hard else 1
    if rel_cnt < min_rel:
        c_norm = normalize_text(c_join)
        semantic_formal = any(
            k in c_norm
            for k in [
                "mod",
                "congr",
                "divis",
                "integer",
                "nonzero",
                "parity",
                "coprime",
                "inequal",
                "bound",
                "prime",
                "同余",
                "整除",
                "整数",
                "非零",
                "奇偶",
                "互素",
                "不等式",
                "界",
                "素数",
            ]
        )
        if not semantic_formal:
            raise RuntimeError("seed_skeleton constraints not formal enough")




SEED_OP_VOCAB = [
    "substitution",
    "factorization",
    "common denominator",
    "congruence",
    "mod",
    "parity",
    "inequality",
    "bounding",
    "construction",
    "case split",
    "contradiction",
    "invariant",
    "recurrence",
    "extremal",
    "double counting",
    "inclusion-exclusion",
    "代换",
    "因式分解",
    "同余",
    "模",
    "奇偶",
    "不等式",
    "构造",
    "分类",
    "反证",
    "不变量",
    "递推",
    "极值",
    "双计数",
    "容斥",
]


def _outline_text(outline: Any) -> str:
    if isinstance(outline, list):
        parts = []
        for s in outline:
            if isinstance(s, dict):
                parts.append(str(s.get("statement", "")))
                parts.append(str(s.get("concept", "")))
                parts.append(str(s.get("edge_type_from_prev", "")))
            else:
                parts.append(str(s))
        return normalize_text(" ".join(parts))
    return normalize_text(str(outline or ""))


def _extract_required_ops(seed_skeleton: dict[str, Any], chain_subnodes_info: list[dict[str, Any]], is_medium: bool, is_hard: bool) -> list[str]:
    text_parts: list[str] = []
    if isinstance(seed_skeleton, dict):
        for k in ("must_have_constraints", "key_lemmas", "method_tags"):
            v = seed_skeleton.get(k, [])
            if isinstance(v, list):
                text_parts.extend(str(x) for x in v if x is not None)
    for info in chain_subnodes_info:
        text_parts.append(str(info.get("concept_cluster", "")))
    merged = " ".join(text_parts)

    picked: list[str] = []
    for op in SEED_OP_VOCAB:
        if _phrase_in_text(op, merged):
            picked.append(op)

    if not picked:
        atoms: list[str] = []
        for info in chain_subnodes_info:
            for a in str(info.get("concept_cluster", "")).split("|"):
                a = a.strip()
                if len(a) >= 4:
                    atoms.append(a)
        seen = set()
        for a in atoms:
            k = normalize_text(a)
            if not k or k in seen:
                continue
            seen.add(k)
            picked.append(a)
            if len(picked) >= 6:
                break

    need = 2 if is_hard else (1 if is_medium else 0)
    cap = max(need, 4)
    out: list[str] = []
    seen = set()
    for op in picked:
        k = normalize_text(str(op))
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(str(op))
        if len(out) >= cap:
            break
    return out


def _check_required_ops_executed(problem: str, outline: Any, required_ops: list[str], is_medium: bool, is_hard: bool) -> None:
    need = 2 if is_hard else (1 if is_medium else 0)
    if need <= 0:
        return
    if not required_ops:
        raise RuntimeError("required_ops empty for medium/hard")
    t_outline = _outline_text(outline)
    t_all = normalize_text(str(problem or "")) + " " + t_outline
    hit_outline = 0
    hit_all = 0
    for op in required_ops:
        if _phrase_in_text(op, t_outline):
            hit_outline += 1
        if _phrase_in_text(op, t_all):
            hit_all += 1
    req = min(need, len(required_ops))
    if is_hard:
        # hard: keep as soft preference to avoid over-rejecting valid but concise outlines.
        _ = hit_all >= 1 and hit_outline >= 1
    else:
        if hit_all < req:
            raise RuntimeError("seed operations not executed in solution_outline")


def _check_edge_obligations(outline: Any, edge_types: list[str]) -> None:
    if not edge_types:
        return
    # Keep only structure-level edge realization; remove wording-cue checks.
    if isinstance(outline, list) and outline and all(isinstance(s, dict) for s in outline):
        marks = [normalize_text(str(s.get("edge_type_from_prev", ""))) for s in outline[1:]]
        for e in set(str(x) for x in edge_types):
            en = normalize_text(e)
            if en and en not in marks:
                raise RuntimeError(f"edge type {e} not tagged in outline")

def _check_seed_mapping(
    mapping: Any, seed_qids: set[str], is_medium: bool, is_hard: bool
) -> None:
    def _is_concrete_structure_text(text: str) -> bool:
        t = str(text or "")
        if _relation_count(t) >= 1:
            return True
        if re.search(r"[=<>≤≥]|\\bmod\\b|\\d", t, flags=re.IGNORECASE):
            return True
        tn = normalize_text(t)
        cues = [
            "mod",
            "congru",
            "divis",
            "gcd",
            "lcm",
            "parity",
            "factor",
            "recurr",
            "invariant",
            "bijection",
            "同余",
            "整除",
            "因式",
            "递推",
            "不变量",
            "构造",
            "不等式",
        ]
        if any(k in tn for k in cues):
            return True
        if len(tn) >= 20 and re.search(r"[a-z]", tn):
            return True
        return False

    if not isinstance(mapping, list) or not mapping:
        raise RuntimeError("seed_mapping missing")
    valid = 0
    concrete_valid = 0
    used = set()
    for m in mapping:
        if not isinstance(m, dict):
            continue
        sid = str(m.get("from_seed", "")).strip()
        st = str(m.get("reused_structure", "")).strip()
        app = str(m.get("applied_to", "")).strip()
        if sid in seed_qids and len(st) >= 6 and len(app) >= 6:
            valid += 1
            used.add(sid)
            if _is_concrete_structure_text(st) or _is_concrete_structure_text(app):
                concrete_valid += 1
    if valid < 1:
        raise RuntimeError("seed_mapping too weak")
    if is_hard and len(used) < 1:
        raise RuntimeError("hard requires mapping to >=1 seed")
    if is_hard and concrete_valid < 1:
        raise RuntimeError("hard seed_mapping must include concrete reused transform/constraint")


def _seed_role_entries(seed_role_pack: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not isinstance(seed_role_pack, dict):
        return rows
    anchor = seed_role_pack.get("anchor")
    if isinstance(anchor, dict) and str(anchor.get("qid", "")).strip():
        rows.append({"qid": str(anchor.get("qid", "")).strip(), "role": "anchor"})
    contrast = seed_role_pack.get("contrast")
    if isinstance(contrast, dict) and str(contrast.get("qid", "")).strip():
        rows.append({"qid": str(contrast.get("qid", "")).strip(), "role": "contrast"})
    support = seed_role_pack.get("support_pool", [])
    if isinstance(support, list):
        for it in support:
            if not isinstance(it, dict):
                continue
            qid = str(it.get("qid", "")).strip()
            if not qid:
                continue
            rows.append({"qid": qid, "role": str(it.get("role", "support") or "support")})
    return rows


def _plan_seed_qids(
    seed_role_pack: dict[str, Any],
    selected_seed_qids: list[str],
    is_medium: bool,
    is_hard: bool,
) -> list[str]:
    out: list[str] = []
    seen = set()
    for it in _seed_role_entries(seed_role_pack):
        qid = str(it.get("qid", "")).strip()
        if qid and qid not in seen:
            seen.add(qid)
            out.append(qid)
    min_need = 2 if (is_medium or is_hard) else 1
    for qid in selected_seed_qids:
        q = str(qid).strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    if len(out) < min_need:
        return out
    return out


def _fallback_seed_usage_plan(
    required_ids: list[str],
    mechanism_plan: dict[str, Any],
    seed_role_pack: dict[str, Any],
    plan_seed_qids: list[str],
) -> list[dict[str, Any]]:
    if not plan_seed_qids:
        return []
    role_by_seed = {str(it["qid"]): str(it["role"]) for it in _seed_role_entries(seed_role_pack)}
    sid_steps: list[tuple[str, int]] = []
    sub_roles = mechanism_plan.get("subnode_roles", []) if isinstance(mechanism_plan, dict) else []
    if isinstance(sub_roles, list):
        for idx, it in enumerate(sub_roles, 1):
            if not isinstance(it, dict):
                continue
            sid = str(it.get("subnode_id", "")).strip()
            if sid:
                sid_steps.append((sid, idx))
    if not sid_steps:
        sid_steps = [(sid, i + 1) for i, sid in enumerate(required_ids)]
    out: list[dict[str, Any]] = []
    for i, qid in enumerate(plan_seed_qids):
        sid, step = sid_steps[i % len(sid_steps)] if sid_steps else ("", i + 1)
        out.append(
            {
                "seed_qid": str(qid),
                "role": role_by_seed.get(str(qid), "support"),
                "target_subnode_id": sid,
                "used_in_step": int(step),
                "contribution": f"Provide a concrete transform for [NODE:{sid}] and pass it to next step.",
            }
        )
    return out


def _validate_seed_usage_plan(
    plan: Any,
    required_ids: list[str],
    seed_role_pack: dict[str, Any],
    plan_seed_qids: list[str],
    mechanism_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    fallback = _fallback_seed_usage_plan(required_ids, mechanism_plan, seed_role_pack, plan_seed_qids)
    if not isinstance(plan, list):
        return fallback
    role_by_seed = {str(it["qid"]): str(it["role"]) for it in _seed_role_entries(seed_role_pack)}
    required_set = {str(x) for x in required_ids}
    wanted = [str(x) for x in plan_seed_qids if str(x)]
    normalized: list[dict[str, Any]] = []
    used = set()
    for it in plan:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("seed_qid", "")).strip()
        if not qid or qid in used or qid not in wanted:
            continue
        sid = str(it.get("target_subnode_id", "")).strip()
        if sid not in required_set:
            sid = required_ids[len(normalized) % len(required_ids)] if required_ids else ""
        raw_step = it.get("used_in_step", len(normalized) + 1)
        try:
            step = max(1, int(raw_step))
        except Exception:
            step = len(normalized) + 1
        role = str(it.get("role", "")).strip() or role_by_seed.get(qid, "support")
        if role not in {"anchor", "contrast", "support"}:
            role = role_by_seed.get(qid, "support")
        contribution = str(it.get("contribution", "") or it.get("usage", "")).strip()
        if len(contribution) < 8:
            contribution = f"Use this seed's structure at step {step} for [NODE:{sid}]."
        normalized.append(
            {
                "seed_qid": qid,
                "role": role,
                "target_subnode_id": sid,
                "used_in_step": step,
                "contribution": contribution,
            }
        )
        used.add(qid)
    for it in fallback:
        qid = str(it.get("seed_qid", "")).strip()
        if not qid or qid in used:
            continue
        normalized.append(it)
        used.add(qid)
    return normalized


def _generate_seed_usage_plan(
    llm: Any,
    required_ids: list[str],
    chain_subnodes_info: list[dict[str, Any]],
    edge_types: list[str],
    ordered_seed_text: str,
    mechanism_plan: dict[str, Any],
    seed_role_pack: dict[str, Any],
    plan_seed_qids: list[str],
    is_easy: bool,
    is_medium: bool,
    is_hard: bool,
) -> list[dict[str, Any]]:
    fallback = _fallback_seed_usage_plan(required_ids, mechanism_plan, seed_role_pack, plan_seed_qids)
    if not plan_seed_qids:
        return fallback
    schema = """{
  "seed_usage_plan": [
    {"seed_qid":"MATH_...","role":"anchor|contrast|support","target_subnode_id":"S...","used_in_step":2,"contribution":"..."}
  ]
}"""
    prompt = f"""
Create a batch-level seed usage plan BEFORE writing the final problems.

Required subnodes:
{required_ids}

Chain subnodes info:
{chain_subnodes_info}

Chain edge types:
{edge_types}

Ordered chain seed dossier:
{ordered_seed_text}

Mechanism plan:
{mechanism_plan}

Seed role pack:
{seed_role_pack}

Plan these seeds:
{plan_seed_qids}

Return JSON:
{schema}

Rules:
- include every qid in plan_seed_qids exactly once
- each seed must map to one target_subnode_id in required_ids
- prefer mapping a seed to the node it is attached to in the ordered chain seed dossier
- when multiple seeds belong to the same node, assign distinct local jobs instead of repeating one generic use
- if a seed is attached to a node reached by a pre edge, its contribution should build on an upstream intermediate rather than restart independently
- used_in_step must be a positive integer and should respect chain order
- contribution must state a concrete intermediate transform/constraint (not vague wording)
- contribution should also state what becomes weaker/easier if this seed is removed
- objective_count is implicit: keep single-objective support only
{_seed_usage_prompt_block(is_easy=is_easy, is_medium=is_medium, is_hard=is_hard)}
""".strip()
    try:
        obj = llm.json_completion(
            system_prompt="You are a math planning assistant. Return strict JSON only.",
            user_prompt=prompt,
            temperature=0.1,
        )
    except Exception:
        return fallback
    plan = obj.get("seed_usage_plan", []) if isinstance(obj, dict) else []
    return _validate_seed_usage_plan(
        plan=plan,
        required_ids=required_ids,
        seed_role_pack=seed_role_pack,
        plan_seed_qids=plan_seed_qids,
        mechanism_plan=mechanism_plan,
    )


def _seed_usage_soft_reward(
    obj: dict[str, Any],
    plan_seed_qids: list[str],
) -> dict[str, Any]:
    wanted = [str(x) for x in plan_seed_qids if str(x)]
    if not wanted:
        return {"reward": 0.0, "used_count": 0, "entropy": 0.0, "usage_counts": {}}
    counts = {qid: 0.0 for qid in wanted}
    plan = obj.get("seed_usage_plan", [])
    if isinstance(plan, list):
        for it in plan:
            if not isinstance(it, dict):
                continue
            qid = str(it.get("seed_qid", "")).strip()
            if qid in counts:
                counts[qid] += 1.0
    mapping = obj.get("seed_mapping", [])
    if isinstance(mapping, list):
        for it in mapping:
            if not isinstance(it, dict):
                continue
            qid = str(it.get("from_seed", "")).strip()
            if qid in counts:
                counts[qid] += 1.0
    used = sum(1 for v in counts.values() if v > 0.0)
    coverage = used / max(1, len(wanted))
    total = sum(counts.values())
    if total <= 0:
        entropy = 0.0
    else:
        import math

        probs = [v / total for v in counts.values() if v > 0.0]
        raw_h = -sum(p * math.log(p) for p in probs)
        denom = math.log(max(2, len(wanted)))
        entropy = raw_h / denom if denom > 0 else 0.0
    reward = 0.5 * coverage + 0.5 * entropy
    return {
        "reward": float(reward),
        "used_count": int(used),
        "entropy": float(entropy),
        "usage_counts": counts,
    }


def _chain_avg_difficulty(required_ids: list[str], sub_d: dict[str, float]) -> float:
    vals = [float(sub_d.get(sid, 0.5)) for sid in required_ids]
    return float(sum(vals) / len(vals)) if vals else 0.5


def _allocate_counts(total: int, weights: list[tuple[str, float]], tie_priority: dict[str, int]) -> dict[str, int]:
    total = max(1, int(total))
    raw = [(name, total * float(w)) for name, w in weights]
    base = {name: int(v) for name, v in raw}
    left = total - sum(base.values())
    remainders = sorted(
        ((name, v - int(v)) for name, v in raw),
        key=lambda x: (x[1], tie_priority.get(x[0], 0)),
        reverse=True,
    )
    for name, _ in remainders[:left]:
        base[name] += 1
    return base


def _chain_mix_counts(total_questions: int, chain_avg_difficulty: float) -> tuple[str, dict[str, int]]:
    if float(chain_avg_difficulty) < 0.65:
        counts = _allocate_counts(
            total_questions,
            [("easy", 0.7), ("medium", 0.3)],
            {"easy": 1, "medium": 0},
        )
        counts.setdefault("hard", 0)
        return "low", counts
    counts = _allocate_counts(
        total_questions,
        [("easy", 0.4), ("medium", 0.3), ("hard", 0.3)],
        {"hard": 2, "medium": 1, "easy": 0},
    )
    return "high", counts


def _difficulty_labels_from_counts(counts: dict[str, int]) -> list[str]:
    out: list[str] = []
    for name in ("easy", "medium", "hard"):
        out.extend([name] * max(0, int(counts.get(name, 0))))
    return out


def _batch_slot_plan(counts: dict[str, int]) -> list[dict[str, Any]]:
    labels = _difficulty_labels_from_counts(counts)
    return [{"slot_id": i + 1, "difficulty": label} for i, label in enumerate(labels)]


def _min_seed_coverage(plan_seed_qids: list[str], min_need: int = 15) -> int:
    uniq = {str(x) for x in plan_seed_qids if str(x)}
    return min(max(0, int(min_need)), len(uniq))


def _pre_edge_notes(required_ids: list[str], edge_types: list[str], chain_subnodes_info: list[dict[str, Any]]) -> list[str]:
    concept_by_sid = {
        str(x.get("subnode_id", "")): str(x.get("concept_cluster", ""))
        for x in chain_subnodes_info
    }
    out: list[str] = []
    for i, edge_type in enumerate(edge_types):
        if str(edge_type) != "pre":
            continue
        if i + 1 >= len(required_ids):
            continue
        src = required_ids[i]
        dst = required_ids[i + 1]
        out.append(
            f"{src} ({concept_by_sid.get(src,'')}) is a prerequisite for {dst} ({concept_by_sid.get(dst,'')})."
        )
    return out


def _batch_level_guidance() -> str:
    return """
Difficulty guidance:
- easy: use at most two essential ideas, keep the solution short, and avoid fake complexity.
- medium: use one clear pivot plus a real dependency between at least two nodes or seed jobs.
- hard: use a scaffold seed plus a pivot seed, preserve the chain bottleneck, and make the difficulty come from structure rather than wording.
""".strip()


def _seed_target_node(seed_qid: str, seed_usage_plan: list[dict[str, Any]], seed_records: list[dict[str, Any]]) -> str:
    qid = str(seed_qid or "").strip()
    if not qid:
        return ""
    for it in seed_usage_plan:
        if not isinstance(it, dict):
            continue
        if str(it.get("seed_qid", "")).strip() == qid:
            return str(it.get("target_subnode_id", "")).strip()
    for it in seed_records:
        if not isinstance(it, dict):
            continue
        if str(it.get("qid", "")).strip() == qid:
            return str(it.get("subnode_id", "")).strip()
    return ""


def _chain_node_bank(
    required_ids: list[str],
    edge_types: list[str],
    chain_subnodes_info: list[dict[str, Any]],
    seed_usage_plan: list[dict[str, Any]],
    seed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    info_by_sid = {
        str(x.get("subnode_id", "")): x
        for x in chain_subnodes_info
        if str(x.get("subnode_id", ""))
    }
    seed_by_sid: dict[str, list[str]] = {sid: [] for sid in required_ids}
    for qid in [str(x.get("seed_qid", "")).strip() for x in seed_usage_plan if isinstance(x, dict)]:
        sid = _seed_target_node(qid, seed_usage_plan, seed_records)
        if sid in seed_by_sid and qid and qid not in seed_by_sid[sid]:
            seed_by_sid[sid].append(qid)
    for rec in seed_records:
        if not isinstance(rec, dict):
            continue
        sid = str(rec.get("subnode_id", "")).strip()
        qid = str(rec.get("qid", "")).strip()
        if sid in seed_by_sid and qid and qid not in seed_by_sid[sid]:
            seed_by_sid[sid].append(qid)
    rows: list[dict[str, Any]] = []
    for i, sid in enumerate(required_ids):
        info = info_by_sid.get(sid, {})
        incoming = "start" if i == 0 else str(edge_types[i - 1])
        rows.append(
            {
                "subnode_id": sid,
                "parent_node_id": str(info.get("parent_node_id", "")),
                "concept_cluster": str(info.get("concept_cluster", "")),
                "incoming_edge": incoming,
                "local_seed_qids": seed_by_sid.get(sid, []),
            }
        )
    return rows


def _chain_digest(
    required_ids: list[str],
    edge_types: list[str],
    chain_subnodes_info: list[dict[str, Any]],
    mechanism_plan: dict[str, Any],
    blueprint: dict[str, Any],
    seed_usage_plan: list[dict[str, Any]],
    seed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    role_by_sid = {}
    for it in mechanism_plan.get("subnode_roles", []) if isinstance(mechanism_plan, dict) else []:
        if isinstance(it, dict):
            role_by_sid[str(it.get("subnode_id", ""))] = {
                "intermediate": str(it.get("intermediate", "")),
                "contribution": str(it.get("contribution", "")),
            }
    transitions = []
    for it in mechanism_plan.get("transitions", []) if isinstance(mechanism_plan, dict) else []:
        if isinstance(it, dict):
            transitions.append(
                {
                    "from_subnode": str(it.get("from_subnode", "")),
                    "to_subnode": str(it.get("to_subnode", "")),
                    "edge_type": str(it.get("edge_type", "")),
                    "transition": str(it.get("transition", "")),
                }
            )
    node_bank = _chain_node_bank(required_ids, edge_types, chain_subnodes_info, seed_usage_plan, seed_records)
    for row in node_bank:
        extra = role_by_sid.get(str(row.get("subnode_id", "")), {})
        if extra:
            row.update(extra)
    critical_pre = []
    for i, e in enumerate(edge_types):
        if str(e) != "pre" or i + 1 >= len(required_ids):
            continue
        critical_pre.append(
            {
                "from_subnode": required_ids[i],
                "to_subnode": required_ids[i + 1],
                "reason": "downstream step should consume an upstream intermediate rather than restart locally",
            }
        )
    return {
        "theme": str(blueprint.get("objective", "") or mechanism_plan.get("objective", "") or "single-objective chain fusion"),
        "node_bank": node_bank,
        "critical_pre_edges": critical_pre,
        "transition_bank": transitions,
        "seed_job_bank": seed_usage_plan,
        "global_fusion_rules": [
            "Absorb the full chain before deciding any single item.",
            "Every item should have one explicit backbone path and optional auxiliary nodes from the same chain.",
            "Backbone nodes must drive the solution; auxiliary nodes should change a constraint, target form, or closure step instead of being sentence-end decoration.",
        ],
    }


def _select_backbone_window(required_ids: list[str], edge_types: list[str], width: int, cursor: int) -> tuple[list[str], list[str]]:
    if not required_ids:
        return [], []
    width = max(1, min(int(width), len(required_ids)))
    windows: list[tuple[int, int, list[str], list[str]]] = []
    for start in range(0, len(required_ids) - width + 1):
        nodes = required_ids[start : start + width]
        edges = edge_types[start : start + width - 1]
        pre_cnt = sum(1 for e in edges if str(e) == "pre")
        sem_cnt = sum(1 for e in edges if str(e) in {"sem", "main_sem"})
        windows.append((pre_cnt, sem_cnt, nodes, edges))
    windows.sort(key=lambda x: (x[0], x[1], -required_ids.index(x[2][0])), reverse=True)
    pick = windows[int(cursor) % len(windows)]
    return list(pick[2]), list(pick[3])


def _slot_seed_bundle(
    slot_id: int,
    difficulty: str,
    backbone_nodes: list[str],
    required_ids: list[str],
    plan_seed_qids: list[str],
    seed_usage_plan: list[dict[str, Any]],
    seed_records: list[dict[str, Any]],
    global_cursor: int,
) -> tuple[list[str], int]:
    target_count = 1 if difficulty == "easy" else (2 if difficulty == "medium" else 3)
    selected: list[str] = []
    seen = set()
    ordered = [str(x) for x in plan_seed_qids if str(x)]
    for qid in ordered[global_cursor:]:
        if qid in seen:
            continue
        selected.append(qid)
        seen.add(qid)
        global_cursor += 1
        break
    preferred_nodes = list(backbone_nodes) + [sid for sid in required_ids if sid not in backbone_nodes]
    for sid in preferred_nodes:
        for qid in ordered:
            if qid in seen:
                continue
            if _seed_target_node(qid, seed_usage_plan, seed_records) != sid:
                continue
            selected.append(qid)
            seen.add(qid)
            if len(selected) >= target_count:
                return selected, global_cursor
    for qid in ordered:
        if qid in seen:
            continue
        selected.append(qid)
        seen.add(qid)
        if len(selected) >= target_count:
            break
    return selected, global_cursor


def _seed_job_name(difficulty: str, idx: int) -> str:
    if difficulty == "hard":
        return ["scaffold", "pivot", "closure", "constraint"][min(idx, 3)]
    if difficulty == "medium":
        return ["scaffold", "pivot", "closure"][min(idx, 2)]
    return ["scaffold", "support"][min(idx, 1)]


def _hard_core_name(slot_id: int, difficulty: str) -> str:
    if difficulty != "hard":
        return "none"
    cores = [
        "coupled_constraints",
        "hidden_pivot",
        "reverse_construction",
        "extremal_parameter",
        "forced_case_split",
    ]
    return cores[(slot_id - 1) % len(cores)]


def _slot_mutation_requirements(slot_id: int, difficulty: str) -> list[str]:
    rotate = [
        "change the asked quantity or target form relative to the dominant seed template",
        "change the parameter roles or domain restrictions so the givens are not a cosmetic reskin",
        "add one cross-node coupling constraint that materially changes the solution path",
        "make one auxiliary node alter a constraint or closure step rather than appear as decoration",
        "change the intermediate object that the solver must construct, transform, or eliminate",
    ]
    req = [rotate[(slot_id - 1) % len(rotate)]]
    req.append("a pure context swap, renamed variables, or changed numbers alone does not count as mutation")
    if difficulty == "easy":
        req.append("change at least one math-level axis among: target form, constraint shape, variable roles, or intermediate object")
        return req
    req.append("change at least two math-level axes among: target form, constraint family, variable roles, intermediate object, or closure step")
    if difficulty == "hard":
        req.append("make the hard_core the real source of difficulty rather than reusing the seed surface form")
    return req


def _slot_forbidden_copy_axes(difficulty: str) -> list[str]:
    out = [
        "same opening sentence pattern as any single seed",
        "same numeric tuple or an obvious affine rescaling of one seed",
        "same objective with only renamed variables",
        "domain/context swap without a mathematical change in target, constraints, or intermediate object",
        "number changes alone without a new reasoning bottleneck",
    ]
    if difficulty in {"medium", "hard"}:
        out.append("same one-node solution path as the anchor seed")
    if difficulty == "hard":
        out.append("same proof bottleneck as one seed without adding cross-node dependence")
    return out


def _slot_execution_plan(
    slot_plan: list[dict[str, Any]],
    required_ids: list[str],
    edge_types: list[str],
    plan_seed_qids: list[str],
    seed_usage_plan: list[dict[str, Any]],
    seed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_cursor = 0
    easy_cursor = 0
    medium_cursor = 0
    hard_cursor = 0
    for slot in slot_plan:
        slot_id = int(slot.get("slot_id", len(rows) + 1))
        difficulty = str(slot.get("difficulty", "easy")).strip().lower() or "easy"
        if difficulty == "easy":
            sid = required_ids[easy_cursor % len(required_ids)] if required_ids else ""
            backbone_nodes = [sid] if sid else []
            backbone_edges: list[str] = []
            easy_cursor += 1
        elif difficulty == "medium":
            backbone_nodes, backbone_edges = _select_backbone_window(required_ids, edge_types, width=2, cursor=medium_cursor)
            medium_cursor += 1
        else:
            width = 3 if len(required_ids) >= 3 else max(1, len(required_ids))
            backbone_nodes, backbone_edges = _select_backbone_window(required_ids, edge_types, width=width, cursor=hard_cursor)
            hard_cursor += 1
        auxiliary_nodes = [sid for sid in required_ids if sid not in backbone_nodes]
        must_cover_seed_qids, global_cursor = _slot_seed_bundle(
            slot_id=slot_id,
            difficulty=difficulty,
            backbone_nodes=backbone_nodes,
            required_ids=required_ids,
            plan_seed_qids=plan_seed_qids,
            seed_usage_plan=seed_usage_plan,
            seed_records=seed_records,
            global_cursor=global_cursor,
        )
        seed_jobs = []
        for i, qid in enumerate(must_cover_seed_qids):
            sid = _seed_target_node(qid, seed_usage_plan, seed_records)
            if sid not in required_ids:
                sid = backbone_nodes[min(i, len(backbone_nodes) - 1)] if backbone_nodes else ""
            seed_jobs.append({
                "seed_qid": qid,
                "job": _seed_job_name(difficulty, i),
                "target_subnode_id": sid,
            })
        handoff_plan = []
        for i in range(len(backbone_nodes) - 1):
            handoff_plan.append(
                {
                    "from_node": backbone_nodes[i],
                    "to_node": backbone_nodes[i + 1],
                    "edge_type": backbone_edges[i] if i < len(backbone_edges) else "",
                    "requirement": "upstream intermediate must be consumed downstream",
                }
            )
        rows.append(
            {
                "slot_id": slot_id,
                "difficulty": difficulty,
                "backbone_nodes": backbone_nodes,
                "auxiliary_nodes": auxiliary_nodes,
                "must_cover_seed_qids": must_cover_seed_qids,
                "seed_jobs": seed_jobs,
                "handoff_plan": handoff_plan,
                "hard_core": _hard_core_name(slot_id, difficulty),
                "mutation_requirements": _slot_mutation_requirements(slot_id, difficulty),
                "forbidden_copy_axes": _slot_forbidden_copy_axes(difficulty),
                "design_goal": "use the whole chain as context, but make the backbone nodes the actual solution path",
            }
        )
    return rows


def _fallback_rewrite_blueprints(
    slot_execution_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in slot_execution_plan:
        if not isinstance(slot, dict):
            continue
        slot_id = int(slot.get("slot_id", len(rows) + 1))
        difficulty = str(slot.get("difficulty", "easy")).strip().lower() or "easy"
        seed_jobs = slot.get("seed_jobs", []) if isinstance(slot.get("seed_jobs", []), list) else []
        base_seed = ""
        if seed_jobs and isinstance(seed_jobs[0], dict):
            base_seed = str(seed_jobs[0].get("seed_qid", "")).strip()
        rows.append(
            {
                "slot_id": slot_id,
                "difficulty": difficulty,
                "base_seed_to_mutate_away_from": base_seed,
                "changed_axes": [
                    "target_form",
                    "constraint_family" if difficulty in {"medium", "hard"} else "numeric_pattern",
                ],
                "new_target_form": "change the asked quantity or final target form away from the base seed",
                "new_constraint_family": "introduce a different mathematical constraint pattern instead of a surface reskin",
                "new_intermediate_object": "use a different intermediate object or closure step than the base seed",
                "why_not_a_surface_reskin": "the new item should differ in mathematical structure, not just story, numbers, or variable names",
            }
        )
    return rows


def _generate_rewrite_blueprints(
    llm: Any,
    slot_execution_plan: list[dict[str, Any]],
    chain_digest: dict[str, Any],
    ordered_seed_text: str,
) -> list[dict[str, Any]]:
    fallback = _fallback_rewrite_blueprints(slot_execution_plan)
    if not slot_execution_plan:
        return fallback
    schema = """{
  "rewrite_blueprints": [
    {
      "slot_id": 1,
      "difficulty": "easy|medium|hard",
      "base_seed_to_mutate_away_from": "MATH_...",
      "changed_axes": ["target_form", "constraint_family"],
      "new_target_form": "...",
      "new_constraint_family": "...",
      "new_intermediate_object": "...",
      "why_not_a_surface_reskin": "..."
    }
  ]
}"""
    prompt = f"""
Build a rewrite blueprint for each slot BEFORE writing final problems.

Slot execution plan:
{slot_execution_plan}

Chain digest:
{chain_digest}

Ordered chain seed dossier:
{ordered_seed_text}

Return JSON:
{schema}

Rules:
- return exactly one blueprint for each slot_id in slot_execution_plan
- choose one base_seed_to_mutate_away_from from that slot's must_cover_seed_qids or seed_jobs
- changed_axes must describe mathematical axes, not story/context wording
- forbidden as changed_axes: context swap only, renamed variables only, changed numbers only
- easy: at least 1 math-level changed axis
- medium/hard: at least 2 math-level changed axes
- new_target_form must say how the final asked quantity/objective changes
- new_constraint_family must say how the givens/constraints change mathematically
- new_intermediate_object must say what new intermediate relation/object must be built, transformed, or eliminated
- why_not_a_surface_reskin must explicitly explain why this is not just a cosmetic rewrite of the base seed
""".strip()
    try:
        obj = llm.json_completion(
            system_prompt="You are a math rewrite planner. Return strict JSON only.",
            user_prompt=prompt,
            temperature=0.1,
        )
    except Exception:
        return fallback
    items = obj.get("rewrite_blueprints", []) if isinstance(obj, dict) else []
    if not isinstance(items, list):
        return fallback
    out: list[dict[str, Any]] = []
    by_slot = {int(x.get("slot_id")): x for x in slot_execution_plan if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            slot_id = int(it.get("slot_id"))
        except Exception:
            continue
        if slot_id not in by_slot:
            continue
        difficulty = str(it.get("difficulty", by_slot[slot_id].get("difficulty", "easy"))).strip().lower() or str(by_slot[slot_id].get("difficulty", "easy"))
        changed_axes = it.get("changed_axes", [])
        if not isinstance(changed_axes, list):
            changed_axes = []
        changed_axes = [str(x).strip() for x in changed_axes if str(x).strip()]
        min_axes = 1 if difficulty == "easy" else 2
        if len(changed_axes) < min_axes:
            changed_axes = list(fallback[slot_id - 1].get("changed_axes", [])) if 0 < slot_id <= len(fallback) else changed_axes
        out.append(
            {
                "slot_id": slot_id,
                "difficulty": difficulty,
                "base_seed_to_mutate_away_from": str(it.get("base_seed_to_mutate_away_from", "")).strip(),
                "changed_axes": changed_axes,
                "new_target_form": str(it.get("new_target_form", "")).strip(),
                "new_constraint_family": str(it.get("new_constraint_family", "")).strip(),
                "new_intermediate_object": str(it.get("new_intermediate_object", "")).strip(),
                "why_not_a_surface_reskin": str(it.get("why_not_a_surface_reskin", "")).strip(),
            }
        )
    if len(out) != len(by_slot):
        return fallback
    return sorted(out, key=lambda x: int(x.get("slot_id", 0)))


_PLACEHOLDER_PHRASES = [
    "new inequality",
    "additional constraint",
    "additional condition",
    "given a condition on",
    "given a constraint on",
    "include constraints",
    "certain property",
    "some value",
    "some constraint",
    "some condition",
    "under a new inequality",
    "with a new inequality",
    "specific constraints",
    "specific constraint",
    "specific steps",
    "specific step",
    "specific digit constraints",
    "specific group sizes",
    "certain points",
    "certain labeled points",
    "certain steps",
    "intermediate points",
    "different types",
    "new set of linear transformations",
    "new set of eigenvalue conditions",
    "new orthogonality conditions",
    "new proportionality conditions",
    "new collinearity conditions",
    "fixed number of games",
    "given conditions",
    "under the given conditions",
    "under given conditions",
    "specific pattern",
    "specific rule",
    "specified range",
    "modified equation",
    "modified polynomial",
    "modified function",
    "modified matrix",
    "specific symmetry property",
    "special symmetry property",
    "specific transformation",
    "certain transformation",
    "forced case split",
    "new scoring system",
    "additional term",
]


def _has_placeholder_phrase(text: str) -> bool:
    t = normalize_text(text)
    return any(p in t for p in _PLACEHOLDER_PHRASES)


def _problem_surface_ok(problem: str) -> None:
    t = str(problem or "").strip()
    if not t or len(t) < 8:
        raise RuntimeError("invalid batch problem")
    if _is_multi_task(t):
        raise RuntimeError("invalid batch problem")
    if _has_placeholder_phrase(t):
        raise RuntimeError("batch problem contains placeholder wording")


def _fallback_problem_specs(
    slot_execution_plan: list[dict[str, Any]],
    rewrite_blueprints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_slot = {int(x.get("slot_id", 0)): x for x in rewrite_blueprints if isinstance(x, dict)}
    rows: list[dict[str, Any]] = []
    for slot in slot_execution_plan:
        if not isinstance(slot, dict):
            continue
        slot_id = int(slot.get("slot_id", len(rows) + 1))
        difficulty = str(slot.get("difficulty", "easy")).strip().lower() or "easy"
        rb = by_slot.get(slot_id, {})
        backbone = [str(x) for x in slot.get("backbone_nodes", []) if str(x)]
        aux = [str(x) for x in slot.get("auxiliary_nodes", []) if str(x)]
        rows.append(
            {
                "slot_id": slot_id,
                "difficulty": difficulty,
                "objects": [
                    f"explicit mathematical objects tied to backbone nodes {backbone}",
                    f"optional auxiliary objects from nodes {aux}" if aux else "no auxiliary object is required",
                ],
                "variables": ["every variable and parameter must be introduced with an explicit domain"],
                "givens": [
                    "all numerical data must be explicit in the final statement",
                    str(rb.get("new_constraint_family", "")).strip() or "state one explicit mathematical constraint family",
                ],
                "constraints": [
                    "every condition in the final problem must be mathematically checkable and written explicitly",
                ],
                "sampling_space": "",
                "representation_mode": "standard",
                "goal": str(rb.get("new_target_form", "")).strip() or "ask for one explicit mathematical quantity",
                "answer_type": "expression",
                "existence_clause": "state the problem so every referenced object is explicitly defined or guaranteed to exist",
                "anti_placeholder_note": "do not use vague phrases like new inequality, additional constraint, or given a condition",
            }
        )
    return rows


def _validate_problem_spec(spec: Any, slot_map: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise RuntimeError("invalid problem_spec")
    try:
        slot_id = int(spec.get("slot_id"))
    except Exception:
        raise RuntimeError("invalid problem_spec slot_id")
    if slot_id not in slot_map:
        raise RuntimeError("problem_spec slot mismatch")
    difficulty = str(spec.get("difficulty", slot_map[slot_id].get("difficulty", "easy"))).strip().lower()
    if difficulty != str(slot_map[slot_id].get("difficulty", "")).strip().lower():
        raise RuntimeError("problem_spec difficulty mismatch")

    def _norm_lines(v: Any, field: str, min_need: int = 1) -> list[str]:
        if not isinstance(v, list):
            raise RuntimeError(f"problem_spec {field} invalid")
        out = [str(x).strip() for x in v if str(x).strip()]
        if len(out) < min_need:
            raise RuntimeError(f"problem_spec {field} too short")
        if any(_has_placeholder_phrase(x) for x in out):
            raise RuntimeError(f"problem_spec {field} contains placeholder wording")
        return out

    objects = _norm_lines(spec.get("objects", []), "objects")
    variables = _norm_lines(spec.get("variables", []), "variables")
    givens = _norm_lines(spec.get("givens", []), "givens")
    constraints = _norm_lines(spec.get("constraints", []), "constraints")
    sampling_space = str(spec.get("sampling_space", "")).strip()
    representation_mode = str(spec.get("representation_mode", "")).strip()
    goal = str(spec.get("goal", "")).strip()
    answer_type = str(spec.get("answer_type", "")).strip()
    existence_clause = str(spec.get("existence_clause", "")).strip()
    anti_placeholder_note = str(spec.get("anti_placeholder_note", "")).strip()
    for field_name, text in [
        ("goal", goal),
        ("answer_type", answer_type),
        ("representation_mode", representation_mode),
        ("existence_clause", existence_clause),
        ("anti_placeholder_note", anti_placeholder_note),
    ]:
        if not text:
            raise RuntimeError(f"problem_spec {field_name} missing")
        if _has_placeholder_phrase(text):
            raise RuntimeError(f"problem_spec {field_name} contains placeholder wording")

    joined = normalize_text(" ".join(objects + variables + givens + constraints + [goal, representation_mode]))
    if any(k in joined for k in ["probability", "random", "uniform", "expected value", "chance"]):
        if not sampling_space or _has_placeholder_phrase(sampling_space):
            raise RuntimeError("problem_spec sampling_space missing")
    if "polar" in joined and "polar" not in normalize_text(representation_mode):
        raise RuntimeError("problem_spec polar representation mismatch")
    if "coordinate plane" in joined and "polar" in normalize_text(representation_mode):
        raise RuntimeError("problem_spec coordinate representation mismatch")
    if any(k in joined for k in ["eigenvalue", "eigenvector", "matrix"]):
        if "matrix" not in normalize_text(representation_mode):
            raise RuntimeError("problem_spec matrix representation missing")
    if _has_numeric_goal_with_free_parameters(
        answer_type=answer_type,
        goal=goal,
        declaration_lines=variables + givens + constraints,
        assignment_lines=givens + constraints,
    ):
        raise RuntimeError("problem_spec numeric goal still contains unresolved free parameters")
    return {
        "slot_id": slot_id,
        "difficulty": difficulty,
        "objects": objects,
        "variables": variables,
        "givens": givens,
        "constraints": constraints,
        "sampling_space": sampling_space,
        "representation_mode": representation_mode,
        "goal": goal,
        "answer_type": answer_type,
        "existence_clause": existence_clause,
        "anti_placeholder_note": anti_placeholder_note,
    }


def _generate_problem_specs(
    llm: Any,
    slot_execution_plan: list[dict[str, Any]],
    chain_digest: dict[str, Any],
    ordered_seed_text: str,
    rewrite_blueprints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fallback = _fallback_problem_specs(slot_execution_plan, rewrite_blueprints)
    if not slot_execution_plan:
        return fallback
    schema = """{
  "problem_specs": [
    {
      "slot_id": 1,
      "difficulty": "easy|medium|hard",
      "objects": ["..."],
      "variables": ["x: real", "n: positive integer"],
      "givens": ["..."],
      "constraints": ["..."],
      "sampling_space": "... or empty string if not probabilistic",
      "representation_mode": "standard|cartesian|polar|complex|matrix",
      "goal": "...",
      "answer_type": "integer|rational|expression|set|coordinate|proof_tag",
      "existence_clause": "...",
      "anti_placeholder_note": "...",
      "self_containment_checks": ["every symbol/object in the final statement is explicitly defined", "no unstated parameter or hidden sample space remains"]
    }
  ]
}"""
    prompt = f"""
Create a fully explicit problem_spec for each slot BEFORE writing any final problem statement.

Slot execution plan:
{slot_execution_plan}

Rewrite blueprints:
{rewrite_blueprints}

Chain digest:
{chain_digest}

Ordered chain seed dossier:
{ordered_seed_text}

Return JSON:
{schema}

Rules:
- return exactly one problem_spec for each slot_id
- objects must name the mathematical objects that actually appear in the final statement
- variables must introduce domains or admissible ranges explicitly
- givens must be explicit mathematical facts, data, or constructions; no placeholders
- constraints must be explicit and checkable; do not write phrases like "new inequality" or "additional constraint"
- if the goal expects a unique numeric answer, do not leave two or more coefficients/parameters as free symbolic quantities
- self_containment_checks must explicitly mention undefined-symbol risk, hidden-parameter risk, and missing-sample-space risk when relevant
- the final problem must be self-contained: a solver should not need the seed, outside context, or unstated conventions to parse the statement
- do not introduce a symbol, object, set, function, matrix, angle, prime, parameter, region, or random experiment unless it is explicitly defined in the final statement
- do not use filler clauses such as "assume solutions exist", "under new conditions", "with a fixed row pattern", or "for a given prime" unless the object is actually specified
- if the slot is probabilistic or randomized, sampling_space must be explicit and non-empty
- representation_mode must be consistent if the problem uses cartesian coordinates, polar coordinates, complex numbers, or matrices
- if a goal depends on an object existing, existence_clause must guarantee existence or phrase the task conditionally
- forbid editorial residue such as "include constraints", "using matrix operations", or "given a condition on ..."
- anti_placeholder_note must explicitly say what vague wording was replaced by explicit math
""".strip()
    try:
        obj = llm.json_completion(
            system_prompt="You are a math problem spec planner. Return strict JSON only.",
            user_prompt=prompt,
            temperature=0.1,
        )
    except Exception:
        return fallback
    items = obj.get("problem_specs", []) if isinstance(obj, dict) else []
    if not isinstance(items, list):
        return fallback
    slot_map = {int(x.get("slot_id")): x for x in slot_execution_plan if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    out: list[dict[str, Any]] = []
    try:
        for it in items:
            out.append(_validate_problem_spec(it, slot_map))
    except Exception:
        return fallback
    if len(out) != len(slot_map):
        return fallback
    return sorted(out, key=lambda x: int(x.get("slot_id", 0)))



def _FAMILY_CHOICES() -> tuple[str, ...]:
    return ("equation_system", "counting", "probability", "geometry", "algebraic_object")


_ALGEBRAIC_OBJECT_TOKENS = [
    "matrix",
    "determinant",
    "trace",
    "eigenvalue",
    "eigenvector",
    "polynomial",
    "coefficients",
    "root",
    "roots",
    "complex number",
    "complex plane",
    "modular inverse",
    "remainder",
]

_ADVANCED_SURFACE_TOKENS = [
    "eigenvalue",
    "eigenvector",
    "spectral",
    "vector space",
    "linear transformation",
    "basis",
    "subspace",
]

_NUMERIC_ANSWER_TYPES = {"integer", "rational", "coordinate"}


def _has_strong_algebraic_object_signal(problem_spec: dict[str, Any]) -> bool:
    joined = normalize_text(json.dumps(problem_spec or {}, ensure_ascii=False))
    return any(k in joined for k in _ALGEBRAIC_OBJECT_TOKENS)


def _has_concrete_algebraic_object_signal(problem_spec: dict[str, Any]) -> bool:
    joined_raw = json.dumps(problem_spec or {}, ensure_ascii=False)
    joined = normalize_text(joined_raw)
    if not _has_strong_algebraic_object_signal(problem_spec):
        return False
    concrete_markers = ["=", "[[", "entries", "coefficients", "roots", "det(", "trace", "x^", "mod"]
    return any(k in joined_raw for k in concrete_markers) or any(k in joined for k in concrete_markers)


def _algebraic_object_allowed(problem_spec: dict[str, Any], difficulty: str) -> bool:
    joined = normalize_text(json.dumps(problem_spec or {}, ensure_ascii=False))
    difficulty = str(difficulty or "").strip().lower()
    if difficulty == "easy":
        return _has_concrete_algebraic_object_signal(problem_spec) and not any(k in joined for k in _ADVANCED_SURFACE_TOKENS)
    if difficulty == "medium":
        return _has_strong_algebraic_object_signal(problem_spec)
    return True


def _infer_problem_family(problem_spec: dict[str, Any], difficulty: str | None = None) -> str:
    joined = normalize_text(json.dumps(problem_spec or {}, ensure_ascii=False))
    if any(k in joined for k in ["probability", "random", "uniform", "expected value", "deck", "dice", "card", "sample space", "chance"]):
        return "probability"
    if any(k in joined for k in ["triangle", "circle", "ellipse", "line", "point", "angle", "perimeter", "area", "coordinate", "geometry", "vector", "plane"]):
        return "geometry"
    if any(k in joined for k in ["number of ways", "count", "arrange", "permutation", "combination", "subset", "select", "choose"]):
        return "counting"
    if any(k in joined for k in ["solve", "equation", "system", "congruence", "mod", "root", "solution"]):
        return "equation_system"
    if difficulty is not None and not _algebraic_object_allowed(problem_spec, difficulty):
        return "equation_system"
    return "algebraic_object"


_FREE_PARAM_DECL_RE = re.compile(
    r"\b(?:real|positive real|nonnegative real|non-negative real|integer|positive integer|nonnegative integer|non-negative integer|complex)(?:\s+numbers?)?\s+((?:[A-Za-z]\w*\s*(?:,\s*[A-Za-z]\w*\s*)+))"
)
_LET_FREE_PARAM_RE = re.compile(
    r"\blet\s+((?:[A-Za-z]\w*\s*(?:,\s*[A-Za-z]\w*\s*)+))\s+be\s+(?:real|positive real|nonnegative real|non-negative real|integer|positive integer|nonnegative integer|non-negative integer|complex)(?:\s+numbers?)?\b"
)
_ASSIGNED_SYMBOL_RE = re.compile(r"\b([A-Za-z]\w*)\s*=")
_TARGET_QUERY_HINTS = ["find", "determine", "compute", "calculate", "what is", "求", "计算", "确定"]
_NONUNIQUE_QUERY_HINTS = ["range", "all values", "set of", "classify", "for all", "求所有", "范围", "分类", "所有"]


def _extract_free_declared_symbols(lines: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        for pat in (_FREE_PARAM_DECL_RE, _LET_FREE_PARAM_RE):
            for m in pat.finditer(text):
                for sym in re.split(r"\s*,\s*", m.group(1).strip()):
                    sym = sym.strip()
                    if sym:
                        out.add(sym)
    return out


def _extract_assigned_symbols(lines: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        for m in _ASSIGNED_SYMBOL_RE.finditer(text):
            out.add(m.group(1))
    return out


def _has_numeric_goal_with_free_parameters(
    answer_type: str,
    goal: str,
    declaration_lines: Iterable[str],
    assignment_lines: Iterable[str],
) -> bool:
    at = normalize_text(answer_type)
    g = normalize_text(goal)
    if at not in _NUMERIC_ANSWER_TYPES and not any(k in g for k in _TARGET_QUERY_HINTS):
        return False
    if any(k in g for k in _NONUNIQUE_QUERY_HINTS):
        return False
    declared = _extract_free_declared_symbols(declaration_lines)
    if len(declared) < 2:
        return False
    assigned = _extract_assigned_symbols(assignment_lines)
    unresolved = declared - assigned
    return len(unresolved) >= 2


_WEAK_CORE_PHRASES = [
    "specific conditions",
    "specific configuration",
    "new set of conditions",
    "new conditions",
    "alter the constraints",
    "different set of angles",
    "a quadratic inequality",
    "a linear system",
    "a specific obstacle",
    "new divisibility conditions",
    "specific role constraints",
    "specific digit constraints",
    "specific group sizes",
    "certain labeled points",
    "specific steps",
    "fixed number of games",
    "specific pattern",
    "specific rule",
    "specified range",
    "given conditions",
    "under given conditions",
    "modified equation",
    "modified polynomial",
    "modified function",
    "modified matrix",
    "specific symmetry property",
    "special symmetry property",
    "specific transformation",
    "forced case split",
    "new scoring system",
    "additional term",
]


def _has_weak_core_phrase(text: str) -> bool:
    t = normalize_text(text)
    return any(p in t for p in _WEAK_CORE_PHRASES) or _has_placeholder_phrase(t)


def _strict_text_line(v: Any, field: str) -> str:
    s = str(v or "").strip()
    if not s:
        raise RuntimeError(f"field {field} missing")
    if _has_weak_core_phrase(s):
        raise RuntimeError(f"field {field} is underspecified")
    return s


def _strict_text_list(v: Any, field: str, min_need: int = 1) -> list[str]:
    if not isinstance(v, list):
        raise RuntimeError(f"field {field} invalid")
    out = [str(x).strip() for x in v if str(x).strip()]
    if len(out) < min_need:
        raise RuntimeError(f"field {field} too short")
    if any(_has_weak_core_phrase(x) for x in out):
        raise RuntimeError(f"field {field} is underspecified")
    return out


_GENERIC_LATENT_PHRASES = [
    "multiple teams",
    "different types of plants",
    "different types of",
    "specific constraints",
    "specific digit constraints",
    "specific group sizes",
    "specific steps",
    "specific labeled points",
    "certain labeled points",
    "certain points",
    "intermediate points",
    "fixed number of games",
    "specific pattern",
    "specific rule",
    "specified range",
    "given conditions",
    "modified equation",
    "modified polynomial",
    "modified function",
    "specific symmetry property",
    "specific transformation",
    "additional term",
]


def _has_generic_latent_phrase(text: str) -> bool:
    t = normalize_text(text)
    return any(p in t for p in _GENERIC_LATENT_PHRASES)


def _strict_latent_line(v: Any, field: str) -> str:
    s = _strict_text_line(v, field)
    if _has_generic_latent_phrase(s):
        raise RuntimeError(f"field {field} is underspecified")
    return s


def _strict_latent_list(v: Any, field: str, min_need: int = 1) -> list[str]:
    out = _strict_text_list(v, field, min_need=min_need)
    if any(_has_generic_latent_phrase(x) for x in out):
        raise RuntimeError(f"field {field} is underspecified")
    return out


def _target_already_given(target_expression: str, clauses: list[str]) -> bool:
    t = normalize_text(target_expression)
    if len(t) < 6:
        return False
    for clause in clauses:
        c = normalize_text(clause)
        if not c:
            continue
        if t == c or t in c:
            return True
    return False


def _fallback_problem_families(
    slot_execution_plan: list[dict[str, Any]],
    problem_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    spec_map = {int(x.get("slot_id", 0)): x for x in problem_specs if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    rows: list[dict[str, Any]] = []
    for slot in slot_execution_plan:
        if not isinstance(slot, dict):
            continue
        slot_id = int(slot.get("slot_id", len(rows) + 1))
        difficulty = str(slot.get("difficulty", "easy")).strip().lower() or "easy"
        family = _infer_problem_family(spec_map.get(slot_id, {}), difficulty=difficulty)
        rows.append(
            {
                "slot_id": slot_id,
                "difficulty": difficulty,
                "family": family,
                "family_rationale": f"Use the {family} schema so the final statement is rendered from explicit mathematical slots instead of free-form prose.",
            }
        )
    return rows


def _validate_problem_family(
    item: Any,
    slot_map: dict[int, dict[str, Any]],
    spec_map: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RuntimeError("invalid problem family")
    try:
        slot_id = int(item.get("slot_id"))
    except Exception:
        raise RuntimeError("invalid family slot_id")
    if slot_id not in slot_map:
        raise RuntimeError("family slot mismatch")
    difficulty = str(item.get("difficulty", slot_map[slot_id].get("difficulty", "easy"))).strip().lower()
    if difficulty != str(slot_map[slot_id].get("difficulty", "")).strip().lower():
        raise RuntimeError("family difficulty mismatch")
    family = str(item.get("family", "")).strip().lower()
    if family not in _FAMILY_CHOICES():
        raise RuntimeError("invalid family value")
    if family == "algebraic_object" and not _algebraic_object_allowed(spec_map.get(slot_id, {}), difficulty):
        raise RuntimeError("algebraic_object family rejected for this slot difficulty/spec")
    rationale = str(item.get("family_rationale", "")).strip()
    if not rationale or _has_placeholder_phrase(rationale):
        raise RuntimeError("invalid family rationale")
    return {
        "slot_id": slot_id,
        "difficulty": difficulty,
        "family": family,
        "family_rationale": rationale,
    }


def _generate_problem_families(
    llm: Any,
    slot_execution_plan: list[dict[str, Any]],
    chain_digest: dict[str, Any],
    ordered_seed_text: str,
    rewrite_blueprints: list[dict[str, Any]],
    problem_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fallback = _fallback_problem_families(slot_execution_plan, problem_specs)
    if not slot_execution_plan:
        return fallback
    schema = """{
  "problem_families": [
    {
      "slot_id": 1,
      "difficulty": "easy|medium|hard",
      "family": "equation_system|counting|probability|geometry|algebraic_object",
      "family_rationale": "..."
    }
  ]
}"""
    prompt = f"""
Select one typed mathematical family for each slot BEFORE building the semantic core.

Slot execution plan:
{slot_execution_plan}

Rewrite blueprints:
{rewrite_blueprints}

Problem specs:
{problem_specs}

Chain digest:
{chain_digest}

Ordered chain seed dossier:
{ordered_seed_text}

Allowed families:
- equation_system: unknowns/domains/equations/restrictions/target
- counting: universe/admissibility/counted_property/target
- probability: experiment/sample_space/event/target
- geometry: named entities/relations/target
- algebraic_object: named algebraic objects/relations/target

Return JSON:
{schema}

Rules:
- return exactly one family choice for each slot_id
- choose the family that best forces the statement to be rendered from explicit mathematical slots
- prefer the family that minimizes hidden assumptions for that slot
- prefer counting/probability/equation_system/geometry for easy slots; only use algebraic_object when the spec already contains a concrete algebraic object such as an explicit polynomial, determinant, or matrix with explicit entries
- family_rationale must mention which semantic slots are needed in the final statement
""".strip()
    try:
        obj = llm.json_completion(
            system_prompt="You are a math task classifier. Return strict JSON only.",
            user_prompt=prompt,
            temperature=0.0,
        )
    except Exception:
        return fallback
    items = obj.get("problem_families", []) if isinstance(obj, dict) else []
    if not isinstance(items, list):
        return fallback
    slot_map = {int(x.get("slot_id")): x for x in slot_execution_plan if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    spec_map = {int(x.get("slot_id", 0)): x for x in problem_specs if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    out: list[dict[str, Any]] = []
    try:
        for it in items:
            out.append(_validate_problem_family(it, slot_map, spec_map))
    except Exception:
        return fallback
    if len(out) != len(slot_map):
        return fallback
    return sorted(out, key=lambda x: int(x.get("slot_id", 0)))


_WORLD_TEMPLATE_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "equation_system": {
        "eq_trig_reduce_to_cos_sum": {
            "required_fields": ["defined_entities", "equations", "side_conditions", "target_expression"],
            "template_constraints": ["introduce explicit trigonometric expressions", "state a bounded x-domain when aggregating all solutions"],
        },
        "eq_trig_bounded_solution_set": {
            "required_fields": ["defined_entities", "equations", "side_conditions", "target_expression"],
            "template_constraints": ["define the exact trigonometric equation", "state an explicit interval for x"],
        },
        "eq_integer_divisibility_system": {
            "required_fields": ["defined_entities", "equations", "side_conditions", "target_expression"],
            "template_constraints": ["state exact integer/divisibility conditions", "bound the search space or characterize all solutions finitely"],
        },
        "eq_algebraic_parameter_system": {
            "required_fields": ["defined_entities", "equations", "side_conditions", "target_expression"],
            "template_constraints": ["define all parameters explicitly", "avoid implicit existence assumptions"],
        },
    },
    "counting": {
        "count_labeled_arrangement_pattern": {
            "required_fields": ["defined_entities", "universe_definition", "admissibility_rules", "counted_property"],
            "template_constraints": ["state exact counts and labels/types", "define the arrangement pattern explicitly"],
        },
        "count_digit_number_with_constraints": {
            "required_fields": ["defined_entities", "universe_definition", "admissibility_rules", "counted_property"],
            "template_constraints": ["define digit positions and allowed digits explicitly", "state the exact divisibility/pattern constraints"],
        },
        "count_paths_on_defined_graph": {
            "required_fields": ["defined_entities", "universe_definition", "admissibility_rules", "counted_property"],
            "template_constraints": ["define the graph/board/step set explicitly", "define start, end, and forbidden states explicitly"],
        },
        "count_subset_selection_property": {
            "required_fields": ["defined_entities", "universe_definition", "admissibility_rules", "counted_property"],
            "template_constraints": ["define the ground set explicitly", "define the selection property explicitly"],
        },
    },
    "probability": {
        "prob_uniform_number_selection": {
            "required_fields": ["defined_entities", "experiment_definition", "outcome_encoding", "sampling_rule", "sample_space", "event_definition"],
            "template_constraints": ["define the finite uniform sample space explicitly", "define the event in terms of encoded outcomes"],
        },
        "prob_uniform_card_or_dice": {
            "required_fields": ["defined_entities", "experiment_definition", "outcome_encoding", "sampling_rule", "sample_space", "event_definition"],
            "template_constraints": ["define the deck/dice/urn composition exactly", "define the event exactly"],
        },
        "prob_biased_discrete_distribution": {
            "required_fields": ["defined_entities", "experiment_definition", "outcome_encoding", "sampling_rule", "sample_space", "event_definition", "probability_table"],
            "template_constraints": ["state exact outcome probabilities", "avoid unnamed biased objects"],
        },
        "prob_uniform_geometric_choice": {
            "required_fields": ["defined_entities", "experiment_definition", "outcome_encoding", "sampling_rule", "sample_space", "event_definition"],
            "template_constraints": ["define the geometric randomization rule explicitly", "define the event without hidden geometry"],
        },
    },
    "geometry": {
        "geom_parabola_chord_fixed_point": {
            "required_fields": ["defined_entities", "figure_entities", "given_relations", "target_expression"],
            "template_constraints": ["name the conic and fixed point explicitly", "state how the chord/intersection is constrained"],
        },
        "geom_circle_line_bounded_region": {
            "required_fields": ["defined_entities", "figure_entities", "given_relations", "target_expression"],
            "template_constraints": ["state the circle and line equations explicitly", "state the bounded region explicitly"],
        },
        "geom_conic_intersection_metric": {
            "required_fields": ["defined_entities", "figure_entities", "given_relations", "target_expression"],
            "template_constraints": ["define the conic objects explicitly", "define the metric target explicitly from named points/segments"],
        },
        "geom_vector_triangle_metric": {
            "required_fields": ["defined_entities", "figure_entities", "given_relations", "target_expression"],
            "template_constraints": ["define the vectors/points explicitly", "state exactly which metric is asked"],
        },
    },
    "algebraic_object": {
        "alg_matrix_spectral_invariant": {
            "required_fields": ["defined_entities", "object_definitions", "algebraic_relations", "target_expression"],
            "template_constraints": ["define the matrix entries/object explicitly", "state the spectral/algebraic invariant target explicitly"],
        },
        "alg_polynomial_root_configuration": {
            "required_fields": ["defined_entities", "object_definitions", "algebraic_relations", "target_expression"],
            "template_constraints": ["define the polynomial explicitly", "state root constraints explicitly"],
        },
        "alg_complex_expression_evaluation": {
            "required_fields": ["defined_entities", "object_definitions", "algebraic_relations", "target_expression"],
            "template_constraints": ["define the complex objects explicitly", "state the evaluation target explicitly"],
        },
    },
}


def _infer_target_mode(problem_spec: dict[str, Any], family: str) -> str:
    joined = normalize_text(json.dumps(problem_spec or {}, ensure_ascii=False))
    if family == "counting":
        return "count"
    if family == "probability":
        return "probability"
    if any(k in joined for k in ["maximum", "minimum", "largest", "smallest", "extrem"]):
        return "extremal_value"
    if any(k in joined for k in ["area", "perimeter", "length", "distance", "volume", "angle", "ratio"]):
        return "geometric_measure"
    if any(k in joined for k in ["remainder", "determinant", "trace", "eigenvalue", "sum", "product"]):
        return "derived_expression"
    return "solve_or_evaluate"


def _choose_world_kind(family: str, problem_spec: dict[str, Any], rewrite_blueprint: dict[str, Any]) -> str:
    text = normalize_text(json.dumps(problem_spec or {}, ensure_ascii=False) + " " + json.dumps(rewrite_blueprint or {}, ensure_ascii=False))
    if family == "equation_system":
        if any(k in text for k in ["sin", "cos", "tan", "trigon"]):
            if any(k in text for k in ["sum-to-product", "reduce", "cos ax + cos bx + cos cx", "cos a", "cos b", "cos c"]):
                return "eq_trig_reduce_to_cos_sum"
            return "eq_trig_bounded_solution_set"
        if any(k in text for k in ["integer", "divis", "mod", "congruence", "prime"]):
            return "eq_integer_divisibility_system"
        return "eq_algebraic_parameter_system"
    if family == "counting":
        if any(k in text for k in ["digit", "three-digit", "number", "divisible"]):
            return "count_digit_number_with_constraints"
        if any(k in text for k in ["path", "grid", "graph", "walk", "step"]):
            return "count_paths_on_defined_graph"
        if any(k in text for k in ["arrange", "line up", "shelf", "seat", "alternating", "order"]):
            return "count_labeled_arrangement_pattern"
        return "count_subset_selection_property"
    if family == "probability":
        if any(k in text for k in ["biased", "loaded"]):
            return "prob_biased_discrete_distribution"
        if any(k in text for k in ["card", "dice", "die", "coin", "urn"]):
            return "prob_uniform_card_or_dice"
        if any(k in text for k in ["point on", "random chord", "random point", "segment"]):
            return "prob_uniform_geometric_choice"
        return "prob_uniform_number_selection"
    if family == "geometry":
        if "parabola" in text and "chord" in text:
            return "geom_parabola_chord_fixed_point"
        if "circle" in text and "line" in text:
            return "geom_circle_line_bounded_region"
        if any(k in text for k in ["ellipse", "conic", "focus", "directrix"]):
            return "geom_conic_intersection_metric"
        return "geom_vector_triangle_metric"
    if any(k in text for k in ["matrix", "eigen", "determinant", "trace"]):
        return "alg_matrix_spectral_invariant"
    if any(k in text for k in ["polynomial", "root"]):
        return "alg_polynomial_root_configuration"
    return "alg_complex_expression_evaluation"


def _build_world_templates(
    slot_execution_plan: list[dict[str, Any]],
    rewrite_blueprints: list[dict[str, Any]],
    problem_specs: list[dict[str, Any]],
    problem_families: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    spec_map = {int(x.get("slot_id", 0)): x for x in problem_specs if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    fam_map = {int(x.get("slot_id", 0)): x for x in problem_families if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    rb_map = {int(x.get("slot_id", 0)): x for x in rewrite_blueprints if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    rows: list[dict[str, Any]] = []
    for slot in slot_execution_plan:
        if not isinstance(slot, dict):
            continue
        slot_id = int(slot.get("slot_id", len(rows) + 1))
        spec = spec_map.get(slot_id, {})
        difficulty = str(slot.get("difficulty", "easy")).strip().lower() or "easy"
        family = str(fam_map.get(slot_id, {}).get("family", _infer_problem_family(spec, difficulty=difficulty))).strip().lower() or "algebraic_object"
        world_kind = _choose_world_kind(family=family, problem_spec=spec, rewrite_blueprint=rb_map.get(slot_id, {}))
        tmpl = dict(_WORLD_TEMPLATE_CATALOG[family][world_kind])
        rows.append(
            {
                "slot_id": slot_id,
                "difficulty": difficulty,
                "family": family,
                "world_kind": world_kind,
                "target_mode": _infer_target_mode(spec, family),
                "required_fields": list(tmpl.get("required_fields", [])),
                "template_constraints": list(tmpl.get("template_constraints", [])),
            }
        )
    return rows


def _validate_world_instance(item: Any, slot_map: dict[int, dict[str, Any]], template_map: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RuntimeError("invalid world_instance")
    try:
        slot_id = int(item.get("slot_id"))
    except Exception:
        raise RuntimeError("invalid world_instance slot_id")
    if slot_id not in slot_map:
        raise RuntimeError("world_instance slot mismatch")
    if slot_id not in template_map:
        raise RuntimeError("world_instance template missing")
    difficulty = str(item.get("difficulty", slot_map[slot_id].get("difficulty", "easy"))).strip().lower()
    if difficulty != str(slot_map[slot_id].get("difficulty", "")).strip().lower():
        raise RuntimeError("world_instance difficulty mismatch")
    family = str(item.get("family", "")).strip().lower()
    if family != str(template_map[slot_id].get("family", "")).strip().lower():
        raise RuntimeError("world_instance family mismatch")

    defined_entities = _strict_latent_list(item.get("defined_entities", []), "defined_entities", min_need=2)
    given_clauses = _strict_latent_list(item.get("given_clauses", []), "given_clauses", min_need=2)
    finiteness_clause = _strict_latent_line(item.get("finiteness_clause", ""), "finiteness_clause")
    query = _strict_latent_line(item.get("query", ""), "world_instance query")
    target_expression = _strict_latent_line(item.get("target_expression", ""), "target_expression")
    target_reference_objects = _strict_latent_list(item.get("target_reference_objects", []), "target_reference_objects", min_need=1)
    nontriviality_basis = _strict_latent_line(item.get("nontriviality_basis", ""), "nontriviality_basis")
    hidden_solution_path = _strict_latent_list(item.get("hidden_solution_path", []), "hidden_solution_path", min_need=1)
    anti_ambiguity_notes = _strict_latent_list(item.get("anti_ambiguity_notes", []), "anti_ambiguity_notes", min_need=2)

    unknowns = _strict_latent_list(item.get("unknowns", []), "unknowns", min_need=1) if family == "equation_system" else []
    equations = _strict_latent_list(item.get("equations", []), "equations", min_need=1) if family == "equation_system" else []
    side_conditions = _strict_latent_list(item.get("side_conditions", []), "side_conditions", min_need=0)

    universe_definition = _strict_latent_line(item.get("universe_definition", ""), "universe_definition") if family == "counting" else ""
    admissibility_rules = _strict_latent_list(item.get("admissibility_rules", []), "admissibility_rules", min_need=1) if family == "counting" else []
    counted_property = _strict_latent_line(item.get("counted_property", ""), "counted_property") if family == "counting" else ""

    experiment_definition = _strict_latent_line(item.get("experiment_definition", ""), "experiment_definition") if family == "probability" else ""
    outcome_encoding = _strict_latent_line(item.get("outcome_encoding", ""), "outcome_encoding") if family == "probability" else ""
    sampling_rule = _strict_latent_line(item.get("sampling_rule", ""), "sampling_rule") if family == "probability" else ""
    sample_space = _strict_latent_line(item.get("sample_space", ""), "sample_space") if family == "probability" else ""
    event_definition = _strict_latent_line(item.get("event_definition", ""), "event_definition") if family == "probability" else ""
    probability_table = _strict_latent_list(item.get("probability_table", []), "probability_table", min_need=0)

    figure_entities = _strict_latent_list(item.get("figure_entities", []), "figure_entities", min_need=2) if family == "geometry" else []
    given_relations = _strict_latent_list(item.get("given_relations", []), "given_relations", min_need=1) if family == "geometry" else []

    object_definitions = _strict_latent_list(item.get("object_definitions", []), "object_definitions", min_need=1) if family == "algebraic_object" else []
    algebraic_relations = _strict_latent_list(item.get("algebraic_relations", []), "algebraic_relations", min_need=1) if family == "algebraic_object" else []

    if family == "probability":
        sr = normalize_text(sampling_rule)
        if not any(k in sr for k in ["uniform", "without replacement", "with replacement", "independently", "equally likely"]):
            if "biased" not in sr:
                raise RuntimeError("world_instance probability sampling rule underspecified")
        if "biased" in normalize_text(experiment_definition + " " + sampling_rule) and len(probability_table) < 2:
            raise RuntimeError("world_instance biased probability model missing probability_table")
    if family == "geometry":
        ent_joined = normalize_text(" ".join(figure_entities))
        if not any(tok in ent_joined for tok in ["point", "line", "circle", "triangle", "polygon", "vector", "plane", "ray", "parabola", "ellipse"]):
            raise RuntimeError("world_instance geometry entities too vague")
    if family == "counting":
        q = normalize_text(query)
        if "number" not in q and "count" not in q and "how many" not in q:
            raise RuntimeError("world_instance counting query must ask for a count")
    world_joined = normalize_text(json.dumps(item, ensure_ascii=False))
    if difficulty == "easy" and any(k in world_joined for k in _ADVANCED_SURFACE_TOKENS):
        raise RuntimeError("world_instance easy slot introduced advanced algebraic surface")
    if _has_numeric_goal_with_free_parameters(
        answer_type="",
        goal=f"{query} {target_expression}",
        declaration_lines=defined_entities + given_clauses + unknowns + figure_entities + object_definitions,
        assignment_lines=(
            given_clauses
            + equations
            + side_conditions
            + admissibility_rules
            + given_relations
            + algebraic_relations
            + probability_table
        ),
    ):
        raise RuntimeError("world_instance numeric query still contains unresolved free parameters")
    support_clauses = (
        list(given_clauses)
        + list(equations)
        + list(side_conditions)
        + list(admissibility_rules)
        + list(given_relations)
        + list(algebraic_relations)
        + ([counted_property] if counted_property else [])
        + ([event_definition] if event_definition else [])
    )
    if _target_already_given(target_expression, support_clauses):
        raise RuntimeError("world_instance target already given by clauses")
    required_fields = set(str(x) for x in template_map[slot_id].get("required_fields", []) if str(x))
    field_payload = {
        "defined_entities": defined_entities,
        "equations": equations,
        "side_conditions": side_conditions,
        "target_expression": target_expression,
        "universe_definition": universe_definition,
        "admissibility_rules": admissibility_rules,
        "counted_property": counted_property,
        "experiment_definition": experiment_definition,
        "outcome_encoding": outcome_encoding,
        "sampling_rule": sampling_rule,
        "sample_space": sample_space,
        "event_definition": event_definition,
        "probability_table": probability_table,
        "figure_entities": figure_entities,
        "given_relations": given_relations,
        "object_definitions": object_definitions,
        "algebraic_relations": algebraic_relations,
    }
    missing_required = [k for k in sorted(required_fields) if not field_payload.get(k)]
    if missing_required:
        raise RuntimeError(f"world_instance missing required fields: {missing_required}")

    return {
        "slot_id": slot_id,
        "difficulty": difficulty,
        "family": family,
        "defined_entities": defined_entities,
        "given_clauses": given_clauses,
        "finiteness_clause": finiteness_clause,
        "query": query,
        "target_expression": target_expression,
        "target_reference_objects": target_reference_objects,
        "nontriviality_basis": nontriviality_basis,
        "unknowns": unknowns,
        "equations": equations,
        "side_conditions": side_conditions,
        "universe_definition": universe_definition,
        "admissibility_rules": admissibility_rules,
        "counted_property": counted_property,
        "experiment_definition": experiment_definition,
        "outcome_encoding": outcome_encoding,
        "sampling_rule": sampling_rule,
        "sample_space": sample_space,
        "event_definition": event_definition,
        "probability_table": probability_table,
        "figure_entities": figure_entities,
        "given_relations": given_relations,
        "object_definitions": object_definitions,
        "algebraic_relations": algebraic_relations,
        "hidden_solution_path": hidden_solution_path,
        "anti_ambiguity_notes": anti_ambiguity_notes,
    }


def _world_instance_schema_for_family(family: str) -> str:
    common = """
{
  "slot_id": 1,
  "difficulty": "easy|medium|hard",
  "family": "FAMILY",
  "defined_entities": ["..."],
  "given_clauses": ["...", "..."],
  "finiteness_clause": "...",
  "query": "...",
  "target_expression": "...",
  "target_reference_objects": ["..."],
  "nontriviality_basis": "...",
  "hidden_solution_path": ["..."],
  "anti_ambiguity_notes": ["...", "..."]
""".strip()
    if family == "equation_system":
        return common.replace('"FAMILY"', '"equation_system"') + """,
  "unknowns": ["x: real"],
  "equations": ["..."],
  "side_conditions": ["..."]
}"""
    if family == "counting":
        return common.replace('"FAMILY"', '"counting"') + """,
  "universe_definition": "...",
  "admissibility_rules": ["..."],
  "counted_property": "..."
}"""
    if family == "probability":
        return common.replace('"FAMILY"', '"probability"') + """,
  "experiment_definition": "...",
  "outcome_encoding": "...",
  "sampling_rule": "...",
  "sample_space": "...",
  "event_definition": "...",
  "probability_table": ["..."]
}"""
    if family == "geometry":
        return common.replace('"FAMILY"', '"geometry"') + """,
  "figure_entities": ["..."],
  "given_relations": ["..."]
}"""
    return common.replace('"FAMILY"', '"algebraic_object"') + """,
  "object_definitions": ["..."],
  "algebraic_relations": ["..."]
}"""


def _generate_world_instance_for_slot(
    llm: Any,
    slot: dict[str, Any],
    chain_digest: dict[str, Any],
    ordered_seed_text: str,
    rewrite_blueprint: dict[str, Any],
    problem_spec: dict[str, Any],
    problem_family: dict[str, Any],
    world_template: dict[str, Any],
) -> dict[str, Any]:
    slot_id = int(slot.get("slot_id"))
    family = str(world_template.get("family", "")).strip().lower()
    schema = _world_instance_schema_for_family(family)
    slot_map = {slot_id: slot}
    template_map = {slot_id: world_template}
    errors: list[str] = []
    for attempt in range(2):
        retry_note = ""
        if errors:
            retry_note = f"\nPrevious failure to fix:\n- {errors[-1]}\n"
        prompt = f"""
Instantiate one concrete world_instance for slot {slot_id}.

Stage contract:
- world_template has already selected the world kind
- you must instantiate every required field concretely
- do not write a final polished contest problem; only the structured instance

Slot execution plan:
{slot}

Rewrite blueprint:
{rewrite_blueprint}

Problem spec:
{problem_spec}

Problem family:
{problem_family}

World template:
{world_template}
{retry_note}
Return JSON:
{schema}

Rules:
- keep family exactly as {family}
- keep slot_id exactly {slot_id}
- fill every field required by world_template.required_fields
- every named object must be concretely defined; no placeholders, no editorial residue
- if geometry, name the actual points/lines/chords/conics and the exact relations
- if equation_system, give the exact equations and explicit domains/ranges
- if counting, state exact counts/labels/types and exact admissibility rules
- if probability, state exact sample space and exact event; if biased, provide explicit probabilities
- if the query expects a unique numeric result, do not leave two or more coefficients/parameters as unresolved symbolic quantities
- target_expression must depend on multiple clauses, not restate a given fact
""".strip()
        try:
            obj = llm.json_completion(
                system_prompt="You are a math world instantiator. Return strict JSON only.",
                user_prompt=prompt,
                temperature=0.1,
            )
        except Exception as e:
            errors.append(f"slot {slot_id} attempt {attempt + 1}: llm_error: {e}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"slot {slot_id} attempt {attempt + 1}: response_not_dict")
            continue
        candidate = obj.get("world_instance", obj)
        try:
            return _validate_world_instance(candidate, slot_map=slot_map, template_map=template_map)
        except Exception as e:
            errors.append(f"slot {slot_id} attempt {attempt + 1}: validate_error: {e}")
            continue
    raise RuntimeError("; ".join(errors) if errors else f"slot {slot_id}: unknown world_instance failure")


def _generate_world_instances(
    llm: Any,
    slot_execution_plan: list[dict[str, Any]],
    chain_digest: dict[str, Any],
    ordered_seed_text: str,
    rewrite_blueprints: list[dict[str, Any]],
    problem_specs: list[dict[str, Any]],
    problem_families: list[dict[str, Any]],
    world_templates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not slot_execution_plan:
        return []
    spec_map = {int(x.get("slot_id", 0)): x for x in problem_specs if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    fam_map = {int(x.get("slot_id", 0)): x for x in problem_families if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    tmpl_map = {int(x.get("slot_id", 0)): x for x in world_templates if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    rb_map = {int(x.get("slot_id", 0)): x for x in rewrite_blueprints if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    out: list[dict[str, Any]] = []
    debug_slots: list[dict[str, Any]] = []
    missing_templates: list[int] = []
    for slot in slot_execution_plan:
        if not isinstance(slot, dict):
            continue
        slot_id = int(slot.get("slot_id"))
        tmpl = tmpl_map.get(slot_id)
        if tmpl is None:
            missing_templates.append(slot_id)
            debug_slots.append({"slot_id": slot_id, "state": "error", "error": "missing world_template"})
            continue
        try:
            inst = _generate_world_instance_for_slot(
                llm=llm,
                slot=slot,
                chain_digest=chain_digest,
                ordered_seed_text=ordered_seed_text,
                rewrite_blueprint=rb_map.get(slot_id, {}),
                problem_spec=spec_map.get(slot_id, {}),
                problem_family=fam_map.get(slot_id, {}),
                world_template=tmpl,
            )
            out.append(inst)
            debug_slots.append({"slot_id": slot_id, "state": "ok", "world_kind": tmpl.get("world_kind", "")})
        except Exception as e:
            debug_slots.append({"slot_id": slot_id, "state": "error", "world_kind": tmpl.get("world_kind", ""), "error": str(e)})
    _generate_world_instances.last_debug = {
        "expected_slot_ids": [int(x.get("slot_id")) for x in slot_execution_plan if isinstance(x, dict) and str(x.get("slot_id", "")).strip()],
        "produced_slot_ids": [int(x.get("slot_id")) for x in out if isinstance(x, dict)],
        "missing_templates": missing_templates,
        "slots": debug_slots,
    }
    return sorted(out, key=lambda x: int(x.get("slot_id", 0)))


def _filter_stage_rows_by_slot_ids(rows: list[dict[str, Any]], active_slot_ids: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            slot_id = int(row.get("slot_id"))
        except Exception:
            continue
        if slot_id in active_slot_ids:
            out.append(row)
    return out


def _active_plan_seed_qids(slot_execution_plan: list[dict[str, Any]], plan_seed_qids: list[str]) -> list[str]:
    active_seed_set: set[str] = set()
    for slot in slot_execution_plan:
        if not isinstance(slot, dict):
            continue
        for qid in slot.get("must_cover_seed_qids", []):
            qid_s = str(qid).strip()
            if qid_s:
                active_seed_set.add(qid_s)
    return [str(qid) for qid in plan_seed_qids if str(qid) in active_seed_set]


def _question_render_from_world_instances(
    llm: Any,
    slot_plan: list[dict[str, Any]],
    slot_execution_plan: list[dict[str, Any]],
    chain_subnodes_info: list[dict[str, Any]],
    edge_types: list[str],
    ordered_seed_text: str,
    seed_role_pack: dict[str, Any],
    mechanism_plan: dict[str, Any],
    blueprint: dict[str, Any],
    seed_usage_plan: list[dict[str, Any]],
    chain_digest: dict[str, Any],
    rewrite_blueprints: list[dict[str, Any]],
    problem_specs: list[dict[str, Any]],
    problem_families: list[dict[str, Any]],
    world_templates: list[dict[str, Any]],
    world_instances: list[dict[str, Any]],
    plan_seed_qids: list[str],
    required_ids: list[str],
    chain_avg_difficulty: float,
    chain_bucket: str,
    difficulty_counts: dict[str, int],
    total_questions: int,
    min_seed_coverage: int,
) -> Any:
    world_map = {
        int(x.get("slot_id")): x
        for x in world_instances
        if isinstance(x, dict) and str(x.get("slot_id", "")).strip()
    }
    render_slots: list[dict[str, Any]] = []
    for slot in slot_execution_plan:
        if not isinstance(slot, dict):
            continue
        try:
            slot_id = int(slot.get("slot_id"))
        except Exception:
            continue
        world = world_map.get(slot_id)
        if world is None:
            continue
        render_slots.append(
            {
                "slot_id": slot_id,
                "difficulty": str(slot.get("difficulty", "")).strip().lower(),
                "backbone_nodes": slot.get("backbone_nodes", []),
                "auxiliary_nodes": slot.get("auxiliary_nodes", []),
                "must_cover_seed_qids": slot.get("must_cover_seed_qids", []),
                "seed_jobs": slot.get("seed_jobs", []),
                "handoff_plan": slot.get("handoff_plan", []),
                "hard_core": slot.get("hard_core", "none"),
                "design_goal": slot.get("design_goal", ""),
                "world_instance": world,
            }
        )
    render_slot_text = json.dumps(render_slots, ensure_ascii=False)
    pre_notes = _pre_edge_notes(required_ids, edge_types, chain_subnodes_info)
    schema = """{
  "items": [
    {
      "slot_id": 1,
      "difficulty": "easy|medium|hard",
      "problem": "...",
      "used_seed_qids": ["MATH_..."],
      "used_nodes": ["S..."],
      "backbone_nodes": ["S..."],
      "auxiliary_nodes": ["S..."],
      "seed_jobs": [{"seed_qid":"MATH_...","job":"scaffold|pivot|closure|constraint","target_subnode_id":"S..."}],
      "handoff_plan": [{"from_node":"S...","to_node":"S...","edge_type":"pre|sem|main_sem","requirement":"..."}],
      "hard_core": "none|coupled_constraints|hidden_pivot|reverse_construction|extremal_parameter|forced_case_split",
      "rewrite_blueprint_summary": "...",
      "anti_copy_moves": ["math-level mutation 1", "math-level mutation 2"],
      "slot_design_note": "...",
      "seed_mapping": [{"from_seed":"MATH_...","reused_structure":"...","applied_to":"..."}],
      "problem_family": "equation_system|counting|probability|geometry|algebraic_object",
      "world_instance_summary": "..."
    }
  ]
}"""
    prompt = f"""
Render final math problem statements from structured slot packets.
The slot packet's world_instance is the only mathematical source of truth.
Do not add mathematics absent from it.

Chain concepts:
{chain_subnodes_info}

Chain edge types:
{edge_types}

Pre-edge prerequisite notes:
{pre_notes if pre_notes else ['(no explicit pre edge on this chain)']}

Ordered chain seed dossier:
{ordered_seed_text}

Render slots:
{render_slot_text}

Batch target:
- exactly {total_questions} items
- exactly one item for each render slot, with the same slot_id and difficulty
- difficulty mix = {difficulty_counts}
- required nodes used across batch = {required_ids}
- planned seeds in scope = {plan_seed_qids}
- batch seed coverage threshold = at least {min_seed_coverage} distinct planned seeds across the whole batch

Per-item required fields:
- slot_id, difficulty, problem, used_seed_qids, used_nodes
- backbone_nodes, auxiliary_nodes, seed_jobs, handoff_plan, hard_core
- rewrite_blueprint_summary, anti_copy_moves, slot_design_note
- seed_mapping, problem_family, world_instance_summary

Rendering rules:
- render each item only from its slot packet's world_instance
- define every symbol before use
- keep one objective
- do not output answers, proofs, or solution outlines
- do not copy any seed statement
- do not invent objects, intervals, sample spaces, equations, labels, counts, or relations
- backbone_nodes must be the actual path; auxiliary_nodes may only add side conditions or target perturbations
- use all must_cover_seed_qids listed in that slot packet
- keep bounded domains, intervals, finite sets, and sample spaces explicit whenever present
- do not ask for a fact already explicitly given in the world_instance
- do not produce two items whose givens/body skeleton are the same up to variable renaming and number changes
- equation_system order: entities -> givens -> equations -> side conditions -> query
- counting order: entities -> universe/rules -> finiteness -> query
- probability order: entities -> experiment/sample space -> probabilities -> event -> query
- geometry order: entities -> givens/relations -> query
- algebraic_object order: entities -> object definitions/relations -> query
- never stop early; even if one slot is less elegant, still return one valid item for every render slot

Return JSON:
{schema}
""".strip()
    return llm.json_completion(
        system_prompt="You are a math question renderer from world instances. Return strict JSON only.",
        user_prompt=prompt,
        temperature=0.15,
    )


_BODY_QUERY_SPLIT_RE = re.compile(
    r"\b(find|determine|compute|calculate|evaluate|identify|what is|求|求出|计算|确定|判断|证明)\b",
    re.IGNORECASE,
)


def _problem_body_skeleton(problem: str) -> str:
    raw = str(problem or "").strip()
    if not raw:
        return ""
    matches = list(_BODY_QUERY_SPLIT_RE.finditer(raw))
    body = raw[: matches[-1].start()] if matches else raw
    text = normalize_text(body)
    text = re.sub(r"\b\d+\b", "N", text)
    text = re.sub(r"\b[a-z]\d*\b", "v", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _validate_batch_items(
    items: Any,
    expected_total: int,
    expected_counts: dict[str, int],
    slot_plan: list[dict[str, Any]],
    slot_execution_plan: list[dict[str, Any]],
    problem_specs: list[dict[str, Any]],
    world_templates: list[dict[str, Any]],
    world_instances: list[dict[str, Any]],
    plan_seed_qids: list[str],
    required_ids: list[str],
    seed_records: list[dict[str, Any]],
    min_seed_coverage: int,
) -> list[dict[str, Any]]:
    if not isinstance(items, list) or len(items) != expected_total:
        raise RuntimeError("batch item count mismatch")
    seed_set = set(str(x) for x in plan_seed_qids if str(x))
    node_set = set(str(x) for x in required_ids if str(x))
    slot_map = {int(x.get("slot_id")): str(x.get("difficulty", "")).strip().lower() for x in slot_plan}
    slot_exec_map = {int(x.get("slot_id")): x for x in slot_execution_plan}
    spec_map = {int(x.get("slot_id")): x for x in problem_specs if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    template_map = {int(x.get("slot_id")): x for x in world_templates if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    world_map = {int(x.get("slot_id")): x for x in world_instances if isinstance(x, dict) and str(x.get("slot_id", "")).strip()}
    if set(template_map) != set(slot_map):
        raise RuntimeError(f"world_template coverage mismatch: have {sorted(template_map)} expected {sorted(slot_map)}")
    if set(world_map) != set(slot_map):
        raise RuntimeError(f"world_instance coverage mismatch: have {sorted(world_map)} expected {sorted(slot_map)}")
    seen_seed = set()
    seen_node = set()
    counts = {"easy": 0, "medium": 0, "hard": 0}
    seen_slots: set[int] = set()
    seen_problem_skeletons: dict[str, int] = {}
    normalized: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            raise RuntimeError("invalid batch item")
        try:
            slot_id = int(it.get("slot_id"))
        except Exception:
            raise RuntimeError("invalid batch slot_id")
        if slot_id not in slot_map or slot_id in seen_slots:
            raise RuntimeError("batch slot mismatch")
        seen_slots.add(slot_id)
        level = str(it.get("difficulty", "")).strip().lower()
        if level not in counts:
            raise RuntimeError("invalid batch difficulty")
        if level != slot_map[slot_id]:
            raise RuntimeError("batch slot difficulty mismatch")
        counts[level] += 1
        problem = str(it.get("problem", "")).strip()
        _problem_surface_ok(problem)
        body_skeleton = _problem_body_skeleton(problem)
        if body_skeleton and len(body_skeleton.split()) >= 8:
            prev_slot = seen_problem_skeletons.get(body_skeleton)
            if prev_slot is not None:
                raise RuntimeError(f"batch item duplicates problem skeleton from slot {prev_slot}")
            seen_problem_skeletons[body_skeleton] = slot_id
        used_seed_qids = [str(x) for x in it.get("used_seed_qids", []) if str(x) in seed_set]
        used_nodes = [str(x) for x in it.get("used_nodes", []) if str(x) in node_set]
        if not used_seed_qids:
            raise RuntimeError("batch item missing used seeds")
        if not used_nodes:
            raise RuntimeError("batch item missing used nodes")
        seen_seed.update(used_seed_qids)
        seen_node.update(used_nodes)
        copy_score, _ = _max_seed_copy_score(problem, seed_records)
        if copy_score >= 0.96:
            raise RuntimeError("batch item too close to seed")
        normalized.append(
            {
                "slot_id": slot_id,
                "difficulty": level,
                "problem": problem,
                "used_seed_qids": used_seed_qids,
                "used_nodes": used_nodes,
                "seed_mapping": it.get("seed_mapping", []),
                "backbone_nodes": it.get("backbone_nodes", slot_exec_map.get(slot_id, {}).get("backbone_nodes", [])),
                "auxiliary_nodes": it.get("auxiliary_nodes", slot_exec_map.get(slot_id, {}).get("auxiliary_nodes", [])),
                "seed_jobs": it.get("seed_jobs", slot_exec_map.get(slot_id, {}).get("seed_jobs", [])),
                "handoff_plan": it.get("handoff_plan", slot_exec_map.get(slot_id, {}).get("handoff_plan", [])),
                "hard_core": str(it.get("hard_core", slot_exec_map.get(slot_id, {}).get("hard_core", "none"))),
                "rewrite_blueprint_summary": str(it.get("rewrite_blueprint_summary", "")).strip(),
                "anti_copy_moves": it.get("anti_copy_moves", []),
                "slot_design_note": str(it.get("slot_design_note", slot_exec_map.get(slot_id, {}).get("design_goal", ""))),
                "problem_spec": spec_map.get(slot_id, {}),
                "problem_family": str(it.get("problem_family", world_map.get(slot_id, {}).get("family", ""))).strip(),
                "world_template": template_map.get(slot_id, {}),
                "world_instance": world_map.get(slot_id, {}),
                "world_instance_summary": str(it.get("world_instance_summary", "")).strip(),
            }
        )
    if seen_slots != set(slot_map):
        raise RuntimeError("batch slot coverage mismatch")
    if counts != {k: int(expected_counts.get(k, 0)) for k in counts}:
        raise RuntimeError(f"batch difficulty counts mismatch: {counts} vs {expected_counts}")
    if len(seen_seed) < int(min_seed_coverage):
        missing = sorted(seed_set - seen_seed)
        raise RuntimeError(
            f"batch seed coverage too low: used {len(seen_seed)} < required {int(min_seed_coverage)}; missing examples: {missing[:20]}"
        )
    if seen_node != node_set:
        raise RuntimeError(f"batch missing nodes: {sorted(node_set - seen_node)}")
    return normalized


def _generate_chain_question_batch(
    llm: Any,
    chain_subnodes_info: list[dict[str, Any]],
    edge_types: list[str],
    required_ids: list[str],
    ordered_seed_text: str,
    seed_role_pack: dict[str, Any],
    mechanism_plan: dict[str, Any],
    blueprint: dict[str, Any],
    seed_usage_plan: list[dict[str, Any]],
    plan_seed_qids: list[str],
    chain_avg_difficulty: float,
    chain_bucket: str,
    difficulty_counts: dict[str, int],
    total_questions: int,
    seed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slot_plan = _batch_slot_plan(difficulty_counts)
    slot_execution_plan = _slot_execution_plan(
        slot_plan=slot_plan,
        required_ids=required_ids,
        edge_types=edge_types,
        plan_seed_qids=plan_seed_qids,
        seed_usage_plan=seed_usage_plan,
        seed_records=seed_records,
    )
    chain_digest = _chain_digest(
        required_ids=required_ids,
        edge_types=edge_types,
        chain_subnodes_info=chain_subnodes_info,
        mechanism_plan=mechanism_plan,
        blueprint=blueprint,
        seed_usage_plan=seed_usage_plan,
        seed_records=seed_records,
    )
    rewrite_blueprints = _generate_rewrite_blueprints(
        llm=llm,
        slot_execution_plan=slot_execution_plan,
        chain_digest=chain_digest,
        ordered_seed_text=ordered_seed_text,
    )
    problem_specs = _generate_problem_specs(
        llm=llm,
        slot_execution_plan=slot_execution_plan,
        chain_digest=chain_digest,
        ordered_seed_text=ordered_seed_text,
        rewrite_blueprints=rewrite_blueprints,
    )
    problem_families = _generate_problem_families(
        llm=llm,
        slot_execution_plan=slot_execution_plan,
        chain_digest=chain_digest,
        ordered_seed_text=ordered_seed_text,
        rewrite_blueprints=rewrite_blueprints,
        problem_specs=problem_specs,
    )
    world_templates = _build_world_templates(
        slot_execution_plan=slot_execution_plan,
        rewrite_blueprints=rewrite_blueprints,
        problem_specs=problem_specs,
        problem_families=problem_families,
    )
    try:
        world_instances = _generate_world_instances(
            llm=llm,
            slot_execution_plan=slot_execution_plan,
            chain_digest=chain_digest,
            ordered_seed_text=ordered_seed_text,
            rewrite_blueprints=rewrite_blueprints,
            problem_specs=problem_specs,
            problem_families=problem_families,
            world_templates=world_templates,
        )
    except Exception:
        _generate_chain_question_batch.last_debug = {
            "world_template_slot_ids": [int(x.get("slot_id")) for x in world_templates if isinstance(x, dict)],
            "world_instance_debug": getattr(_generate_world_instances, "last_debug", {}),
        }
        raise
    active_slot_ids = {
        int(x.get("slot_id"))
        for x in world_instances
        if isinstance(x, dict) and str(x.get("slot_id", "")).strip()
    }
    if not active_slot_ids:
        _generate_chain_question_batch.last_debug = {
            "world_template_slot_ids": [int(x.get("slot_id")) for x in world_templates if isinstance(x, dict)],
            "world_instance_debug": getattr(_generate_world_instances, "last_debug", {}),
        }
        raise RuntimeError("world_instance generation produced zero usable slots")
    dropped_slot_ids = [
        int(x.get("slot_id"))
        for x in slot_plan
        if isinstance(x, dict) and str(x.get("slot_id", "")).strip() and int(x.get("slot_id")) not in active_slot_ids
    ]
    slot_plan = _filter_stage_rows_by_slot_ids(slot_plan, active_slot_ids)
    slot_execution_plan = _filter_stage_rows_by_slot_ids(slot_execution_plan, active_slot_ids)
    rewrite_blueprints = _filter_stage_rows_by_slot_ids(rewrite_blueprints, active_slot_ids)
    problem_specs = _filter_stage_rows_by_slot_ids(problem_specs, active_slot_ids)
    problem_families = _filter_stage_rows_by_slot_ids(problem_families, active_slot_ids)
    world_templates = _filter_stage_rows_by_slot_ids(world_templates, active_slot_ids)
    plan_seed_qids = _active_plan_seed_qids(slot_execution_plan, plan_seed_qids)
    difficulty_counts = {
        "easy": sum(1 for x in slot_plan if str(x.get("difficulty", "")).strip().lower() == "easy"),
        "medium": sum(1 for x in slot_plan if str(x.get("difficulty", "")).strip().lower() == "medium"),
        "hard": sum(1 for x in slot_plan if str(x.get("difficulty", "")).strip().lower() == "hard"),
    }
    total_questions = len(slot_plan)
    min_seed_coverage = _min_seed_coverage(plan_seed_qids, min_need=10)
    obj = _question_render_from_world_instances(
        llm=llm,
        slot_plan=slot_plan,
        slot_execution_plan=slot_execution_plan,
        chain_subnodes_info=chain_subnodes_info,
        edge_types=edge_types,
        ordered_seed_text=ordered_seed_text,
        seed_role_pack=seed_role_pack,
        mechanism_plan=mechanism_plan,
        blueprint=blueprint,
        seed_usage_plan=seed_usage_plan,
        chain_digest=chain_digest,
        rewrite_blueprints=rewrite_blueprints,
        problem_specs=problem_specs,
        problem_families=problem_families,
        world_templates=world_templates,
        world_instances=world_instances,
        plan_seed_qids=plan_seed_qids,
        required_ids=required_ids,
        chain_avg_difficulty=chain_avg_difficulty,
        chain_bucket=chain_bucket,
        difficulty_counts=difficulty_counts,
        total_questions=total_questions,
        min_seed_coverage=min_seed_coverage,
    )
    items = obj.get("items", []) if isinstance(obj, dict) else []
    _generate_chain_question_batch.last_debug = {
        "expected_total": int(total_questions),
        "returned_item_count": len(items) if isinstance(items, list) else None,
        "returned_slot_ids": [it.get("slot_id") for it in items if isinstance(it, dict)] if isinstance(items, list) else [],
        "world_template_slot_ids": [int(x.get("slot_id")) for x in world_templates if isinstance(x, dict)],
        "world_instance_slot_ids": [int(x.get("slot_id")) for x in world_instances if isinstance(x, dict)],
        "dropped_slot_ids": dropped_slot_ids,
    }
    return _validate_batch_items(
        items=items,
        expected_total=total_questions,
        expected_counts=difficulty_counts,
        slot_plan=slot_plan,
        slot_execution_plan=slot_execution_plan,
        problem_specs=problem_specs,
        world_templates=world_templates,
        world_instances=world_instances,
        plan_seed_qids=plan_seed_qids,
        required_ids=required_ids,
        seed_records=seed_records,
        min_seed_coverage=min_seed_coverage,
    )


def _verify_problem_truth(
    llm: Any,
    problem: str,
    answer: str,
    outline: Any,
    is_hard: bool,
) -> dict[str, Any]:
    schema = """{
  "is_well_posed": true,
  "is_mathematically_sound": true,
  "has_counterexample": false,
  "counterexample": "",
  "verdict": "pass|fail",
  "reason": "short reason"
}"""
    strict_line = (
        "Hard mode: actively search for counterexamples; mark fail if claim is not universally true or proof target is invalid."
        if is_hard
        else "Medium mode: ensure statement is well-posed and mathematically coherent."
    )
    prompt = f"""
Check the mathematical validity of the generated problem independently.

Problem:
{problem}

Proposed answer:
{answer}

Solution outline:
{outline}

Return JSON:
{schema}

Rules:
- Do NOT rewrite the problem; only judge validity.
- If you can find a concrete counterexample, set has_counterexample=true and verdict=fail.
- {strict_line}
""".strip()
    obj = llm.json_completion(
        system_prompt="You are an independent math verifier. Return strict JSON only.",
        user_prompt=prompt,
        temperature=0.0,
    )
    return obj if isinstance(obj, dict) else {}


def _validate_truth_report(rep: dict[str, Any], is_hard: bool) -> None:
    if not isinstance(rep, dict):
        raise RuntimeError("truth check invalid")
    if not bool(rep.get("is_well_posed", False)):
        raise RuntimeError("truth check: not well-posed")
    if not bool(rep.get("is_mathematically_sound", False)):
        raise RuntimeError("truth check: mathematically unsound")
    if is_hard and bool(rep.get("has_counterexample", False)):
        raise RuntimeError("truth check: counterexample found")
    if str(rep.get("verdict", "")).strip().lower() == "fail":
        raise RuntimeError(f"truth check failed: {rep.get('reason','')}")


def _repair_problem_from_truth(
    llm: Any,
    obj: dict[str, Any],
    truth_report: dict[str, Any],
    chain_subnodes_info: list[dict[str, Any]],
    required_ids: list[str],
    edge_types: list[str],
    seed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    schema = """{
  "problem": "...",
  "answer": "...",
  "seed_usage_plan": [{"seed_qid":"MATH_...","role":"anchor|contrast|support","target_subnode_id":"S...","used_in_step":2,"contribution":"..."}],
  "solution_outline": [{"step_id":1,"subnode_id":"S...","concept":"...","edge_type_from_prev":"start|pre|sem|main_sem","statement":"..."}],
  "seed_mapping": [{"from_seed":"MATH_...","reused_structure":"...","applied_to":"..."}],
  "used_nodes": ["..."],
  "used_nodes_info": [{"subnode_id":"...","parent_node_id":"...","concept_cluster":"..."}]
}"""
    prompt = f"""
The generated item failed truth verification. Do a MINIMAL structural repair.

Failure report:
{truth_report}

Current object:
{obj}

Chain requirements:
- required node ids: {required_ids}
- edge types: {edge_types}
- chain node info: {chain_subnodes_info}

Seed references:
{_seeds_to_text(seed_records)}

Rules:
- Keep one objective.
- Preserve overall topic and chain.
- Fix mathematical soundness first.
- No placeholder steps.
- Keep seed_mapping concrete.
Return JSON:
{schema}
""".strip()
    out = llm.json_completion(
        system_prompt="You are a math problem repairer. Return strict JSON only.",
        user_prompt=prompt,
        temperature=0.0,
    )
    if not isinstance(out, dict):
        return obj
    for k in ("problem", "answer", "seed_usage_plan", "solution_outline", "seed_mapping"):
        if k not in out:
            out[k] = obj.get(k)
    if "used_nodes" not in out:
        out["used_nodes"] = obj.get("used_nodes", required_ids)
    if "used_nodes_info" not in out:
        out["used_nodes_info"] = obj.get("used_nodes_info", chain_subnodes_info)
    out["truth_repaired"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--chain-retries", type=int, default=30)
    ap.add_argument("--random-seed", type=int, default=None)
    ap.add_argument("--tag-retries", type=int, default=2)
    ap.add_argument("--enforce-judge", type=int, default=0)
    ap.add_argument("--out", default="data/outputs/synth_tmp_hard.jsonl")
    ap.add_argument("--shard-start-index", type=int, default=0)
    ap.add_argument("--status-name", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path(cfg["_abs_project_root"]) / out_path).resolve()
    writer = _RollingJsonlWriter(
        out_path=out_path,
        shard_size=100,
        start_index=(int(args.shard_start_index) if int(args.shard_start_index) > 0 else None),
    )
    status_path = (
        (writer.out_dir / str(args.status_name).strip())
        if str(args.status_name).strip()
        else (writer.out_dir / f"{writer.prefix}.status.json")
    )
    seed = int(args.random_seed) if args.random_seed is not None else int(cfg["runtime"]["random_seed"])
    set_seed(seed)
    rng = random.Random(seed)

    graph_dir = Path(cfg["paths"]["graph_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])

    nodes = load_df(str(graph_dir / "nodes.parquet"))
    subnodes = load_df(str(graph_dir / "subnodes.parquet"))
    edges_pre = load_df(str(graph_dir / "edges_pre.parquet"))
    edges_sem = load_df(str(graph_dir / "edges_sem.parquet"))
    edges_main = load_df(str(graph_dir / "edges_sem_main.parquet"))
    seeds_index = load_df(str(graph_dir / "seeds_index.parquet"))
    seeds_sub_index = load_df(str(graph_dir / "seeds_sub_index.parquet"))
    q2nodes = load_df(str(graph_dir / "q2nodes.parquet"))
    seed_df = load_df(str(interim_dir / "math_seed.parquet"))

    nodes["node_id"] = nodes["node_id"].astype(str)
    if "msc_full" not in nodes.columns and "msc_code3" in nodes.columns:
        nodes["msc_full"] = nodes["msc_code3"].astype(str)
    if "msc_desc" not in nodes.columns and "description" in nodes.columns:
        nodes["msc_desc"] = nodes["description"].astype(str)
    if "concept" not in nodes.columns:
        nodes["concept"] = ""

    subnodes["subnode_id"] = subnodes["subnode_id"].astype(str)
    subnodes["parent_node_id"] = subnodes["parent_node_id"].astype(str)
    if "concept_cluster" not in subnodes.columns and "concept" in subnodes.columns:
        subnodes["concept_cluster"] = subnodes["concept"].astype(str)
    if "concept_cluster" not in subnodes.columns:
        subnodes["concept_cluster"] = ""
    if "freq" not in subnodes.columns:
        subnodes["freq"] = 1

    subnodes_by_parent: dict[str, list[str]] = (
        subnodes.groupby("parent_node_id")["subnode_id"].apply(list).to_dict()
    )
    subnode_parent = dict(zip(subnodes["subnode_id"], subnodes["parent_node_id"]))

    sub_concept = dict(zip(subnodes["subnode_id"], subnodes["concept_cluster"]))
    sub_tokens = {sid: _tokenize(sub_concept.get(sid, "")) for sid in subnodes["subnode_id"]}

    # subnode difficulty from q2nodes y
    q2nodes["qid"] = q2nodes["qid"].astype(str)
    qid_to_y = dict(zip(q2nodes["qid"], q2nodes["y"]))
    seeds_sub_index["qid"] = seeds_sub_index["qid"].astype(str)
    sub_to_qids = seeds_sub_index.groupby("subnode_id")["qid"].apply(list).to_dict()
    sub_d = {}
    for sid, qids in sub_to_qids.items():
        ys = [float(qid_to_y[q]) for q in qids if q in qid_to_y]
        sub_d[sid] = float(sum(ys) / len(ys)) if ys else 0.5

    # cooccur thresholds
    co_vals = []
    for df in (edges_pre, edges_sem):
        if "pair_cooccur" in df.columns:
            co_vals.extend(df["pair_cooccur"].fillna(0).astype(float).tolist())
    p25 = float(pd.Series(co_vals).quantile(0.25)) if co_vals else 0.0

    adj_pre = _build_adj(edges_pre)
    adj_sem = _build_adj(edges_sem)
    adj_main = _build_main_sem(edges_main)
    forward_adj, reverse_adj = _build_cover_adjacency(
        adj_pre=adj_pre,
        adj_sem=adj_sem,
        adj_main=adj_main,
        subnodes_by_parent=subnodes_by_parent,
        sub_tokens=sub_tokens,
        subnode_parent=subnode_parent,
    )
    all_subnode_ids = [str(sid) for sid in subnodes["subnode_id"].tolist()]

    # seeds
    seed_df["qid"] = seed_df["qid"].astype(str)
    qid_to_seed = seed_df.set_index("qid").to_dict(orient="index")

    synth_cfg = cfg.get("synthesis", cfg.get("concept_extraction", {}))
    base_url = synth_cfg.get("base_url", "https://api.deepseek.com")
    api_key_env = synth_cfg.get("api_key_env", "DEEPSEEK_API_KEY")
    api_key = synth_cfg.get("api_key")
    model = synth_cfg.get("model", "gpt-4o")
    planning_timeout_sec = float(synth_cfg.get("planning_timeout_sec", synth_cfg.get("request_timeout_sec", 60)))
    batch_timeout_sec = float(synth_cfg.get("batch_timeout_sec", max(planning_timeout_sec, 180)))
    hard_chain_attempt_cap = max(1, int(synth_cfg.get("hard_chain_attempt_cap", 8)))
    hard_build_try_cap = max(1, int(synth_cfg.get("hard_build_try_cap", 12)))
    hard_stage_retry_cap = max(1, int(synth_cfg.get("hard_stage_retry_cap", 1)))
    hard_resample_cap = max(1, int(synth_cfg.get("hard_resample_cap", 1)))
    chain_sample_topk = max(1, int(synth_cfg.get("chain_sample_topk", 8)))
    chain_sample_temp = float(synth_cfg.get("chain_sample_temp", 0.65))
    chain_recent_window = max(16, int(synth_cfg.get("chain_recent_window", 64)))
    medium_min_words = max(8, int(synth_cfg.get("medium_min_words", 14)))
    hard_min_words = max(20, int(synth_cfg.get("hard_min_words", 24)))

    plan_llm = build_llm_client(
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        model=model,
        timeout_sec=planning_timeout_sec,
    )
    batch_llm = build_llm_client(
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        model=model,
        timeout_sec=batch_timeout_sec,
    )

    run_status: dict[str, Any] = {
        "state": "running",
        "out_path": str(out_path),
        "output_prefix": writer.prefix,
        "output_dir": str(writer.out_dir),
        "planning_timeout_sec": planning_timeout_sec,
        "batch_timeout_sec": batch_timeout_sec,
        "samples": [],
        "generated_rows": 0,
        "current_shard": str(writer.current_path()),
        "current_shard_rows": writer.current_count,
        "touched_shards": [],
        "error": "",
        "max_rows": max(0, int(args.max_rows)),
    }

    def flush_status() -> None:
        run_status["current_shard"] = str(writer.current_path())
        run_status["current_shard_rows"] = int(writer.current_count)
        run_status["touched_shards"] = writer.touched_paths()
        _write_status_file(status_path, run_status)

    def start_stage(attempt_status: dict[str, Any], name: str) -> dict[str, Any]:
        stage = {
            "name": name,
            "state": "running",
            "started_at": time.time(),
            "duration_sec": 0.0,
            "error": "",
            "_t0": time.perf_counter(),
        }
        attempt_status.setdefault("stages", []).append(stage)
        flush_status()
        return stage

    def finish_stage(stage: dict[str, Any], err: Exception | None = None) -> None:
        t0 = float(stage.pop("_t0", time.perf_counter()))
        stage["ended_at"] = time.time()
        stage["duration_sec"] = round(time.perf_counter() - t0, 3)
        stage["state"] = "error" if err is not None else "ok"
        stage["error"] = str(err) if err is not None else ""
        if stage.get("name") == "batch_generation":
            dbg = getattr(_generate_chain_question_batch, "last_debug", None)
            if isinstance(dbg, dict):
                stage["debug"] = dbg
        flush_status()

    flush_status()

    chain_usage: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    recent_chain_queue: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    recent_chain_counts: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    node_cover_counts: dict[str, int] = {}
    start_counts: dict[str, int] = {}
    for sample_idx in range(int(args.n_samples)):
        if int(args.max_rows) > 0 and int(writer.total_appended) >= int(args.max_rows):
            break
        sample_ok = False
        last_err: Exception | None = None
        chain_attempts = min(max(1, int(args.chain_retries)), hard_chain_attempt_cap)
        sample_status: dict[str, Any] = {
            "sample_index": sample_idx + 1,
            "state": "running",
            "attempts": [],
            "error": "",
        }
        run_status["samples"].append(sample_status)
        flush_status()

        for _s in range(chain_attempts):
            attempt_status: dict[str, Any] = {
                "attempt_index": _s + 1,
                "state": "running",
                "target_len": None,
                "chain_sig": None,
                "chain_nodes": [],
                "stages": [],
                "error": "",
            }
            sample_status["attempts"].append(attempt_status)
            flush_status()
            try:
                stage = start_stage(attempt_status, "chain_search")
                try:
                    target_len = rng.randint(3, 5)
                    attempt_status["target_len"] = target_len
                    cand_bank: dict[tuple[tuple[str, ...], tuple[str, ...]], Chain] = {}
                    build_try_cap = min(max(1, int(args.chain_retries)), hard_build_try_cap)
                    for _try in range(build_try_cap):
                        start_sid = _pick_start_for_coverage(
                            all_subnode_ids=all_subnode_ids,
                            forward_adj=forward_adj,
                            reverse_adj=reverse_adj,
                            node_cover_counts=node_cover_counts,
                            start_counts=start_counts,
                            target_len=target_len,
                            rng=rng,
                            topk=max(8, chain_sample_topk * 4),
                            temperature=chain_sample_temp,
                        )
                        cand = _build_chain(
                            start_sid=start_sid,
                            target_len=target_len,
                            beam_size=int(args.beam),
                            adj_pre=adj_pre,
                            adj_sem=adj_sem,
                            adj_main=adj_main,
                            subnodes_by_parent=subnodes_by_parent,
                            sub_tokens=sub_tokens,
                            sub_d=sub_d,
                            subnode_parent=subnode_parent,
                        )
                        if cand is None:
                            continue
                        cand_sig = (tuple(cand.nodes), tuple(cand.edge_types))
                        prev = cand_bank.get(cand_sig)
                        if prev is None:
                            cand_bank[cand_sig] = cand
                            continue
                        prev_conf = min(prev.edge_confs) if prev.edge_confs else 0.0
                        cur_conf = min(cand.edge_confs) if cand.edge_confs else 0.0
                        if cur_conf > prev_conf:
                            cand_bank[cand_sig] = cand
                    if not cand_bank:
                        raise RuntimeError("Failed to build chain")
                    picked = _pick_chain_from_candidates(
                        cand_bank=cand_bank,
                        chain_usage=chain_usage,
                        recent_chain_counts=recent_chain_counts,
                        subnode_parent=subnode_parent,
                        node_cover_counts=node_cover_counts,
                        rng=rng,
                        topk=chain_sample_topk,
                        temperature=chain_sample_temp,
                    )
                    if picked is None:
                        raise RuntimeError("Failed to sample chain")
                    chain_sig, best = picked
                    attempt_status["chain_sig"] = [list(chain_sig[0]), list(chain_sig[1])]
                    attempt_status["chain_nodes"] = list(best.nodes)
                    finish_stage(stage)
                except Exception as e:
                    finish_stage(stage, e)
                    raise
                chain_usage[chain_sig] = int(chain_usage.get(chain_sig, 0)) + 1
                recent_chain_queue.append(chain_sig)
                recent_chain_counts[chain_sig] = int(recent_chain_counts.get(chain_sig, 0)) + 1
                if len(recent_chain_queue) > chain_recent_window:
                    old_sig = recent_chain_queue.pop(0)
                    left = int(recent_chain_counts.get(old_sig, 0)) - 1
                    if left <= 0:
                        recent_chain_counts.pop(old_sig, None)
                    else:
                        recent_chain_counts[old_sig] = left

                start_counts[best.nodes[0]] = int(start_counts.get(best.nodes[0], 0)) + 1
                for sid in dict.fromkeys(best.nodes):
                    node_cover_counts[sid] = int(node_cover_counts.get(sid, 0)) + 1

                chain_subnodes = best.nodes
                required_ids = list(dict.fromkeys(chain_subnodes))

                chain_subnodes_info = []
                for sid in chain_subnodes:
                    row = subnodes[subnodes["subnode_id"] == sid].iloc[0].to_dict()
                    chain_subnodes_info.append(
                        {
                            "subnode_id": str(row["subnode_id"]),
                            "parent_node_id": str(row["parent_node_id"]),
                            "msc_full": "",
                            "domain": "",
                            "concept_cluster": str(row.get("concept_cluster", "")),
                        }
                    )

                chain_avg = _chain_avg_difficulty(required_ids, sub_d)
                chain_bucket, difficulty_counts = _chain_mix_counts(0, chain_avg)
                is_hard_chain = chain_bucket == "high"
                is_medium_chain = True

                all_chain_seed_records = _collect_chain_seed_records(
                    chain_subnodes=chain_subnodes,
                    chain_subnodes_info=chain_subnodes_info,
                    sub_to_qids=sub_to_qids,
                    qid_to_seed=qid_to_seed,
                )
                seed_records = _select_prompt_seed_records(
                    seed_records=all_chain_seed_records,
                    rng=rng,
                    per_node_cap=6,
                )
                selected_seed_qids = [str(r.get("qid", "")) for r in seed_records if str(r.get("qid", ""))]
                if not selected_seed_qids:
                    raise RuntimeError("no seeds found on chain")
                total_questions = 10
                chain_bucket, difficulty_counts = _chain_mix_counts(total_questions, chain_avg)
                ordered_seed_text = _ordered_chain_seed_prompt(
                    required_ids=required_ids,
                    edge_types=best.edge_types,
                    chain_subnodes_info=chain_subnodes_info,
                    seed_records=seed_records,
                )

                anchor_seed, contrast_seed = _select_anchor_contrast(seed_records, rng=rng)
                seed_role_pack = _build_seed_role_pack(
                    seed_records=seed_records,
                    anchor_seed=anchor_seed,
                    contrast_seed=contrast_seed,
                    max_support=max(0, len(selected_seed_qids)),
                )
                stage = start_stage(attempt_status, "mechanism_plan")
                try:
                    mechanism_plan = _generate_mechanism_plan(
                        llm=plan_llm,
                        chain_subnodes_info=chain_subnodes_info,
                        edge_types=best.edge_types,
                        required_ids=required_ids,
                        seed_role_pack=seed_role_pack,
                        ordered_seed_text=ordered_seed_text,
                    )
                    finish_stage(stage)
                except Exception as e:
                    finish_stage(stage, e)
                    raise
                stage = start_stage(attempt_status, "blueprint")
                try:
                    blueprint = _generate_chain_blueprint(
                        llm=plan_llm,
                        chain_subnodes_info=chain_subnodes_info,
                        edge_types=best.edge_types,
                        anchor_seed=anchor_seed,
                        contrast_seed=contrast_seed,
                        ordered_seed_text=ordered_seed_text,
                    )
                    finish_stage(stage)
                except Exception as e:
                    finish_stage(stage, e)
                    raise
                plan_seed_qids = _plan_seed_qids(
                    seed_role_pack=seed_role_pack,
                    selected_seed_qids=selected_seed_qids,
                    is_medium=is_medium_chain,
                    is_hard=is_hard_chain,
                )
                stage = start_stage(attempt_status, "seed_usage_plan")
                try:
                    seed_usage_plan = _generate_seed_usage_plan(
                        llm=plan_llm,
                        required_ids=required_ids,
                        chain_subnodes_info=chain_subnodes_info,
                        edge_types=best.edge_types,
                        ordered_seed_text=ordered_seed_text,
                        mechanism_plan=mechanism_plan,
                        seed_role_pack=seed_role_pack,
                        plan_seed_qids=plan_seed_qids,
                        is_easy=False,
                        is_medium=True,
                        is_hard=is_hard_chain,
                    )
                    finish_stage(stage)
                except Exception as e:
                    finish_stage(stage, e)
                    raise

                stage = start_stage(attempt_status, "batch_generation")
                try:
                    batch_items = _generate_chain_question_batch(
                        llm=batch_llm,
                        chain_subnodes_info=chain_subnodes_info,
                        edge_types=best.edge_types,
                        required_ids=required_ids,
                        ordered_seed_text=ordered_seed_text,
                        seed_role_pack=seed_role_pack,
                        mechanism_plan=mechanism_plan,
                        blueprint=blueprint,
                        seed_usage_plan=seed_usage_plan,
                        plan_seed_qids=plan_seed_qids,
                        chain_avg_difficulty=chain_avg,
                        chain_bucket=chain_bucket,
                        difficulty_counts=difficulty_counts,
                        total_questions=total_questions,
                        seed_records=seed_records,
                    )
                    finish_stage(stage)
                except Exception as e:
                    finish_stage(stage, e)
                    raise

                batch_out_rows: list[dict[str, Any]] = []
                for item in batch_items:
                    used_seed_qids = [str(x) for x in item.get("used_seed_qids", []) if str(x)]
                    used_nodes = [str(x) for x in item.get("used_nodes", []) if str(x)]
                    item_seed_usage_plan = [
                        x for x in seed_usage_plan if str(x.get("seed_qid", "")) in set(used_seed_qids)
                    ]
                    item_seed_usage_score = _seed_usage_soft_reward(
                        {"seed_usage_plan": item_seed_usage_plan, "seed_mapping": item.get("seed_mapping", [])},
                        plan_seed_qids=used_seed_qids,
                    )
                    batch_out_rows.append(
                        {
                            "chain_subnodes": chain_subnodes,
                            "chain_subnodes_info": chain_subnodes_info,
                            "edge_types": best.edge_types,
                            "edge_confs": best.edge_confs,
                            "chain_avg_difficulty": chain_avg,
                            "chain_bucket": chain_bucket,
                            "difficulty": str(item.get("difficulty", "")),
                            "problem": item.get("problem", ""),
                            "selected_seed_qids": selected_seed_qids,
                            "planned_seed_qids": plan_seed_qids,
                            "seed_role_pack": seed_role_pack,
                            "mechanism_plan": mechanism_plan,
                            "seed_usage_plan": item_seed_usage_plan,
                            "seed_usage_reward": item_seed_usage_score.get("reward", 0.0),
                            "seed_usage_count": item_seed_usage_score.get("used_count", 0),
                            "seed_usage_entropy": item_seed_usage_score.get("entropy", 0.0),
                            "seed_usage_counts": item_seed_usage_score.get("usage_counts", {}),
                            "seed_skeleton": {},
                            "seed_mapping": item.get("seed_mapping", []),
                            "problem_spec": item.get("problem_spec", {}),
                            "problem_family": item.get("problem_family", ""),
                            "world_template": item.get("world_template", {}),
                            "world_instance": item.get("world_instance", {}),
                            "world_instance_summary": item.get("world_instance_summary", ""),
                            "backbone_nodes": item.get("backbone_nodes", []),
                            "auxiliary_nodes": item.get("auxiliary_nodes", []),
                            "seed_jobs": item.get("seed_jobs", []),
                            "handoff_plan": item.get("handoff_plan", []),
                            "hard_core": item.get("hard_core", "none"),
                            "rewrite_blueprint_summary": item.get("rewrite_blueprint_summary", ""),
                            "anti_copy_moves": item.get("anti_copy_moves", []),
                            "slot_design_note": item.get("slot_design_note", ""),
                            "truth_check": {},
                            "used_nodes": used_nodes,
                            "used_nodes_info": [
                                x for x in chain_subnodes_info if str(x.get("subnode_id", "")) in set(used_nodes)
                            ],
                        }
                    )
                if int(args.max_rows) > 0:
                    remaining_rows = int(args.max_rows) - int(writer.total_appended)
                    if remaining_rows <= 0:
                        break
                    if len(batch_out_rows) > remaining_rows:
                        batch_out_rows = batch_out_rows[:remaining_rows]
                writer.append_rows(batch_out_rows)
                run_status["generated_rows"] = int(run_status.get("generated_rows", 0)) + len(batch_out_rows)
                sample_ok = True
                last_err = None
                attempt_status["state"] = "ok"
                attempt_status["error"] = ""
                sample_status["state"] = "ok"
                sample_status["error"] = ""
                flush_status()
                break
            except Exception as e:
                last_err = e
                attempt_status["state"] = "error"
                attempt_status["error"] = str(e)
                flush_status()
                continue
        if not sample_ok:
            sample_status["state"] = "error"
            sample_status["error"] = str(last_err)
            run_status["last_sample_error"] = str(last_err)
            flush_status()
            continue

    run_status["state"] = "ok"
    run_status["error"] = ""
    flush_status()
    touched = ", ".join(writer.touched_paths()) if writer.touched_paths() else "(none)"
    print(f"[OK] saved shards: {touched}")


if __name__ == "__main__":
    main()
