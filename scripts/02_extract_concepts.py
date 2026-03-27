from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import threading

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.common import (
    build_llm_client,
    load_config,
    load_df,
    normalize_text,
    save_df,
    set_seed,
)


HEURISTICS = {
    "algebra": ["equation solving", "polynomial factorization", "inequality transformation", "algebraic manipulation"],
    "combinatorics": ["counting principle", "case analysis", "permutation and combination", "probability counting"],
    "geometry": ["angle chasing", "similar triangles", "circle theorem", "area relation"],
    "number_theory": ["divisibility", "modular arithmetic", "prime factorization", "gcd"],
    "calculus": ["function analysis", "limit", "derivative", "integral"],
}

FULL_CODE_RE = re.compile(r"(\d{2}[A-Z]\d{2})")
TOKEN_RE = re.compile(r"[a-z0-9]+")

SYSTEM_PROMPT = (
    "You are an MSC2020 classifier for olympiad-style math. "
    "Use only the provided candidate codes. "
    "Prefer elementary/contest-level interpretation. "
    "Strongly prioritize elementary and contest-math knowledge points. "
    "Avoid graduate-level or research-level advanced topics. "
    "Return strict JSON only."
)

AUX_SYSTEM_PROMPT = (
    "You are a strict auxiliary MSC reviewer. "
    "Select at most one auxiliary code only when primary code is clearly insufficient. "
    "Strongly prefer elementary/contest-level auxiliary codes. "
    "Reject graduate-level or research-level advanced labels unless absolutely unavoidable. "
    "Otherwise return NONE. "
    "Return strict JSON only."
)


def _to_bool(v: object, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n", ""}:
        return False
    return default


def normalize_msc_full(value: str, legal_set: set[str]) -> str:
    s = str(value or "").strip()
    m = FULL_CODE_RE.search(s)
    if not m:
        return ""
    code = m.group(1)
    return code if code in legal_set else ""


def fallback_concepts(domain: str, subject: str, default_code: str) -> tuple[str, str, list[str]]:
    cands = HEURISTICS.get(domain, ["algebraic manipulation", "equation setup", "proof strategy", "result verification"])
    return subject, default_code, cands[:4]


def load_msc_catalog(cfg: dict) -> pd.DataFrame:
    msc_cfg = cfg.get("msc", {})
    catalog_csv = msc_cfg.get("catalog_csv", "data/interim/msc2020_codes.csv")
    p = Path(catalog_csv)
    if not p.is_absolute():
        p = Path(cfg["_abs_project_root"]) / p
    p = p.resolve()

    if not p.exists():
        raise RuntimeError(
            f"MSC catalog missing: {p}. Run scripts/00_build_msc_catalog.py first."
        )

    df = pd.read_csv(p)
    required = {"code", "l1", "desc"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"MSC catalog missing columns: {required}")

    df = df[df["code"].astype(str).str.match(r"^\d{2}[A-Z]\d{2}$")].copy()
    df = df.drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("MSC full-code catalog is empty")

    df["l1"] = df["l1"].astype(str).str.zfill(2)
    df["desc"] = df["desc"].astype(str)
    return df


def filter_msc_catalog(df: pd.DataFrame, msc_cfg: dict) -> pd.DataFrame:
    out = df.copy()
    allowed_l1 = [str(x).zfill(2) for x in msc_cfg.get("allowed_l1", []) if str(x).strip()]
    allowed_prefix = [str(x).upper() for x in msc_cfg.get("allowed_prefix", []) if str(x).strip()]
    block_re = str(msc_cfg.get("blocklist_regex", "")).strip()

    if allowed_l1:
        out = out[out["l1"].astype(str).isin(allowed_l1)].copy()
    if allowed_prefix:
        out = out[out["code"].astype(str).str.startswith(tuple(allowed_prefix))].copy()
    if block_re:
        out = out[~out["desc"].astype(str).str.contains(block_re, case=False, regex=True)].copy()
    # Drop generic catch-all codes ending with 99.
    out = out[~out["code"].astype(str).str.endswith("99")].copy()

    return out.reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    s = normalize_text(text)
    return TOKEN_RE.findall(s)


class BM25Retriever:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)

        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 1.0
        if self.avgdl <= 0:
            self.avgdl = 1.0

        self.tf = []
        self.df = Counter()
        self.postings = defaultdict(list)
        for i, toks in enumerate(docs):
            c = Counter(toks)
            self.tf.append(c)
            for t in c.keys():
                self.df[t] += 1
                self.postings[t].append(i)

        self.idf = {}
        n = max(1, self.n_docs)
        for t, dft in self.df.items():
            # BM25 idf (Robertson-Sparck Jones)
            self.idf[t] = float(np.log(1.0 + (n - dft + 0.5) / (dft + 0.5)))

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
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
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                s = idf * (tf * (self.k1 + 1.0)) / max(1e-8, denom)
                scores[i] += qtf * s
        return scores


def build_msc_retriever(df: pd.DataFrame):
    texts = (df["code"] + " " + df["desc"]).tolist()
    docs = [tokenize(t) for t in texts]
    retriever = BM25Retriever(docs=docs, k1=1.5, b=0.75)

    by_l1 = defaultdict(list)
    for i, l1 in enumerate(df["l1"].tolist()):
        by_l1[str(l1)].append(i)

    return retriever, by_l1


def topk_indices(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []
    if len(scores) <= k:
        return list(np.argsort(-scores))
    part = np.argpartition(-scores, k)[:k]
    return list(part[np.argsort(-scores[part])])


def build_prompt(
    problem: str,
    solution: str,
    subject: str,
    min_concepts: int,
    max_concepts: int,
    candidates_df: pd.DataFrame,
    allowed_hint: str = "",
) -> str:
    # Keep prompt bounded.
    p = problem[:1200]
    s = solution[:800]
    full_text = f"{problem}\n{solution}".lower()
    explicit_calc = any(
        k in full_text
        for k in ("derivative", "differentiate", "integral", "integrate", "limit", "d/dx", "dy/dx")
    )
    calc_guard = ""
    if subject in {"Algebra", "Intermediate Algebra", "Prealgebra"} and not explicit_calc:
        calc_guard = (
            "- For this subject, avoid 26* analysis/calculus codes unless the task explicitly "
            "asks for derivative/integral/limit computation.\n"
        )
    text_l = full_text
    explicit_integer = any(
        k in text_l
        for k in (
            "integer",
            "integers",
            "positive integer",
            "nonnegative integer",
            "lattice point",
            "diophantine",
        )
    )
    dioph_guard = ""
    if not explicit_integer:
        dioph_guard = (
            "- Do NOT use 11D* Diophantine codes unless the task explicitly imposes integer-solution constraints.\n"
        )
    candidate_lines = [f"{r.code}: {r.desc}" for r in candidates_df.itertuples(index=False)]
    candidate_text = "\n".join(candidate_lines)

    return f"""
Task: choose one primary MSC code and extract atomic solving concepts.

Constraints:
- prioritize elementary/contest mathematics interpretation
- do not select graduate-level/research-level advanced topics
- if a sophisticated interpretation exists, choose the simpler contest-level interpretation
- concepts count: {min_concepts}..{max_concepts}
- concepts must be noun phrases, <= 4 words each
- keep only concepts directly used in solving
- avoid duplicates and generic words (e.g., math, equation, problem solving)
- msc_primary must be selected EXACTLY from candidate list below
{calc_guard}{dioph_guard}{allowed_hint}

Candidate MSC full codes:
{candidate_text}

Output schema:
{{
  "msc_primary": "<one code from candidate list>",
  "concepts": ["c1", "c2", "..."]
}}

subject: {subject}
problem: {p}
solution: {s}
    """.strip()


def build_aux_prompt(
    problem: str,
    solution: str,
    subject: str,
    primary_code: str,
    primary_desc: str,
    aux_candidates_df: pd.DataFrame,
) -> str:
    p = problem[:1200]
    s = solution[:800]
    cand_lines = [f"{r.code}: {r.desc} (score={float(getattr(r, 'bm25_score', 0.0)):.4f})" for r in aux_candidates_df.itertuples(index=False)]
    cand_text = "\n".join(cand_lines)
    return f"""
Task: decide whether one auxiliary MSC code is needed.

Rules:
- choose at most one aux_code from candidate list OR choose NONE
- choose NONE unless problem needs a distinct secondary competency not covered by primary_code
- strongly prefer elementary/contest-level auxiliary codes
- avoid graduate-level/research-level auxiliary labels whenever possible
- do not choose broad, weakly related, or redundant labels
- do not choose a code outside the candidate list

Output schema:
{{
  "aux_code": "<one code from candidates or NONE>",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

subject: {subject}
primary_code: {primary_code}
primary_desc: {primary_desc}

Candidate auxiliary MSC full codes:
{cand_text}

problem: {p}
solution: {s}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict-llm", type=int, default=None)
    parser.add_argument("--allow-fallback", type=int, default=None)
    parser.add_argument("--max-fallback-rate", type=float, default=None)
    parser.add_argument("--probe-llm", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["runtime"]["random_seed"])

    seed_path = str(Path(cfg["paths"]["interim_dir"]) / "math_seed.parquet")
    seed = load_df(seed_path)

    msc_df_full = load_msc_catalog(cfg)
    legal_codes = set(msc_df_full["code"].astype(str).tolist())
    code_to_desc = dict(zip(msc_df_full["code"], msc_df_full["desc"]))
    code_to_l1 = dict(zip(msc_df_full["code"], msc_df_full["l1"].astype(str)))

    msc_cfg = cfg.get("msc", {})
    msc_df = filter_msc_catalog(msc_df_full, msc_cfg)
    allowed_set = set(msc_df["code"].astype(str).tolist())

    retriever_full, by_l1_full = build_msc_retriever(msc_df_full)
    if msc_df.empty:
        msc_df = msc_df_full
        retriever, by_l1 = retriever_full, by_l1_full
    else:
        retriever, by_l1 = build_msc_retriever(msc_df)

    subj_to_domain = cfg["mappings"]["subject_to_domain"]
    subj_to_msc_l1 = cfg["mappings"].get("subject_to_msc_l1", {})
    subj_to_allowed_l1 = cfg["mappings"].get("subject_to_allowed_l1", {})
    subj_to_allowed_prefix = cfg["mappings"].get("subject_to_allowed_prefix", {})

    ce = cfg["concept_extraction"]
    workers = int(ce.get("workers", 1))
    flush_every = int(ce.get("flush_every", 200))
    resume = bool(ce.get("resume", True))
    strict_llm = _to_bool(
        ce.get("strict_llm", True) if args.strict_llm is None else args.strict_llm,
        default=True,
    )
    allow_fallback = _to_bool(
        ce.get("allow_fallback", False) if args.allow_fallback is None else args.allow_fallback,
        default=False,
    )
    max_fallback_rate = float(
        ce.get("max_fallback_rate", 0.01) if args.max_fallback_rate is None else args.max_fallback_rate
    )
    probe_llm = _to_bool(
        ce.get("probe_llm_before_run", True) if args.probe_llm is None else args.probe_llm,
        default=True,
    )
    if strict_llm and allow_fallback:
        print("[WARN] strict_llm=1 overrides allow_fallback; fallback on LLM errors will be disabled.")
        allow_fallback = False

    top_global = int(msc_cfg.get("top_global", 40))
    top_l1 = int(msc_cfg.get("top_l1", 40))
    max_candidates = int(msc_cfg.get("max_candidates", 80))

    out_dir = Path(cfg["paths"]["interim_dir"])
    mention_path = str(out_dir / "concept_mentions.parquet")
    raw_path = str(out_dir / "concept_sets.parquet")

    rows: list[dict] = []
    raw_rows: list[dict] = []

    if resume and Path(mention_path).exists() and Path(raw_path).exists():
        try:
            exist_mentions = load_df(mention_path)
            exist_raw = load_df(raw_path)
            # Old checkpoints (before msc_full integration) are incompatible.
            if "msc_full" not in exist_mentions.columns or "msc_full" not in exist_raw.columns:
                raise RuntimeError("checkpoint schema is old (missing msc_full), rebuild required")
            rows = exist_mentions.to_dict("records")
            raw_rows = exist_raw.to_dict("records")
            done_qids = set(exist_raw["qid"].astype(str).tolist())
            if done_qids:
                seed = seed[~seed["qid"].astype(str).isin(done_qids)].reset_index(drop=True)
                print(f"[INFO] resume enabled, skip done qids: {len(done_qids)}, pending: {len(seed)}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] failed to load existing checkpoints, start fresh: {e}")
            rows, raw_rows = [], []

    thread_local = threading.local()

    def get_llm_client():
        if not ce["enable_llm"]:
            return None
        if not hasattr(thread_local, "llm"):
            thread_local.llm = build_llm_client(
                base_url=ce["base_url"],
                model=ce["model"],
                api_key_env=ce["api_key_env"],
                api_key=ce.get("api_key"),
            )
        return thread_local.llm

    if ce.get("enable_llm", True) and probe_llm:
        probe = build_llm_client(
            base_url=ce["base_url"],
            model=ce["model"],
            api_key_env=ce["api_key_env"],
            api_key=ce.get("api_key"),
        )
        probe_obj = probe.json_completion(
            system_prompt="Return strict JSON only.",
            user_prompt='Output {"ok": true}',
            temperature=0.0,
        )
        if not isinstance(probe_obj, dict) or not probe_obj:
            raise RuntimeError("LLM probe failed: empty/non-dict JSON response")
        print(
            "[INFO] llm probe passed: "
            + f"request_mode={getattr(probe, 'last_request_mode', 'unknown')}, "
            + f"parse_mode={getattr(probe, 'last_parse_mode', 'unknown')}"
        )

    def _subject_whitelist(subject: str) -> tuple[set[str], set[str]]:
        l1s = {str(x).zfill(2) for x in subj_to_allowed_l1.get(subject, []) if str(x).strip()}
        prefs = {str(x).upper() for x in subj_to_allowed_prefix.get(subject, []) if str(x).strip()}
        return l1s, prefs

    def _apply_subject_whitelist(cand: pd.DataFrame, subject: str) -> pd.DataFrame:
        l1_allow, pref_allow = _subject_whitelist(subject)
        out = cand.copy()
        if l1_allow:
            out = out[out["l1"].astype(str).isin(l1_allow)].copy()
        if pref_allow:
            out = out[out["code"].astype(str).str.startswith(tuple(sorted(pref_allow)))].copy()
        return out.reset_index(drop=True)

    def build_candidates(query: str, default_l1: str, subject: str) -> pd.DataFrame:
        # Try allowed catalog first; fallback to full if empty.
        q_tokens = tokenize(query)
        scores = retriever.score(q_tokens)

        idx_global = topk_indices(scores, top_global)
        idx_l1 = []
        l1_idxs = by_l1.get(default_l1, [])
        if l1_idxs:
            l1_scores = scores[l1_idxs]
            rel = topk_indices(l1_scores, top_l1)
            idx_l1 = [l1_idxs[i] for i in rel]

        merged = []
        seen = set()
        for i in idx_l1 + idx_global:
            if i not in seen:
                seen.add(i)
                merged.append(i)
            if len(merged) >= max_candidates:
                break

        if not merged:
            merged = list(range(min(max_candidates, len(msc_df))))

        cand = msc_df.iloc[merged][["code", "l1", "desc"]].copy().reset_index(drop=True)
        cand["bm25_score"] = [float(scores[i]) for i in merged]
        if cand.empty:
            scores2 = retriever_full.score(q_tokens)
            idx_global2 = topk_indices(scores2, top_global)
            idx_l12 = []
            l1_idxs2 = by_l1_full.get(default_l1, [])
            if l1_idxs2:
                l1_scores2 = scores2[l1_idxs2]
                rel2 = topk_indices(l1_scores2, top_l1)
                idx_l12 = [l1_idxs2[i] for i in rel2]
            merged2 = []
            seen2 = set()
            for i in idx_l12 + idx_global2:
                if i not in seen2:
                    seen2.add(i)
                    merged2.append(i)
                if len(merged2) >= max_candidates:
                    break
            if not merged2:
                merged2 = list(range(min(max_candidates, len(msc_df_full))))
            cand = msc_df_full.iloc[merged2][["code", "l1", "desc"]].copy().reset_index(drop=True)
            cand["bm25_score"] = [float(scores2[i]) for i in merged2]
        cand = _apply_subject_whitelist(cand, subject)
        if cand.empty:
            l1_allow, pref_allow = _subject_whitelist(subject)
            raise RuntimeError(
                f"subject whitelist filtered all candidates for subject={subject}, "
                + f"allowed_l1={sorted(l1_allow)}, allowed_prefix={sorted(pref_allow)}"
            )
        return cand

    def pick_default_code(default_l1: str, cand: pd.DataFrame) -> str:
        hit = cand[cand["l1"].astype(str) == default_l1]
        if not hit.empty:
            return str(hit.iloc[0]["code"])
        return str(cand.iloc[0]["code"])


    def build_aux_candidates(
        final_code: str,
        cand: pd.DataFrame,
        min_score_ratio: float,
        min_abs_score: float,
        aux_policy: str,
        topk: int,
    ) -> pd.DataFrame:
        final_code = str(final_code).upper()
        if cand.empty:
            return cand.iloc[0:0].copy()

        score_map = {
            str(r.code): float(getattr(r, "bm25_score", 0.0))
            for r in cand.itertuples(index=False)
        }
        base = score_map.get(str(final_code), 0.0)
        if base <= 0:
            base = float(cand["bm25_score"].max()) if "bm25_score" in cand.columns else 0.0

        final_l1 = final_code[:2]
        final_prefix3 = final_code[:3]
        aux_policy = str(aux_policy or "same_prefix3").strip().lower()
        topk = max(1, int(topk))

        def allow_aux(code: str) -> bool:
            code = str(code).upper()
            if aux_policy == "none":
                return False
            if aux_policy == "same_l1":
                return code[:2] == final_l1
            if aux_policy == "same_prefix3":
                return code[:3] == final_prefix3
            # "free": keep backward-compatible behavior.
            return True

        rows = []
        ranked_df = cand.sort_values("bm25_score", ascending=False).copy()
        for r in ranked_df.itertuples(index=False):
            c = str(r.code).upper()
            if c == final_code:
                continue
            if not allow_aux(c):
                continue
            sc = score_map.get(c, 0.0)
            if sc < max(float(min_abs_score), float(min_score_ratio) * max(base, 1e-8)):
                continue
            rows.append({"code": c, "l1": str(r.l1), "desc": str(r.desc), "bm25_score": float(sc)})
            if len(rows) >= topk:
                break
        if not rows:
            return cand.iloc[0:0].copy()
        return pd.DataFrame(rows)

    def _pick_from_cand(
        cand: pd.DataFrame,
        preferred: list[str],
        forbid_prefixes: tuple[str, ...] = (),
        forbid_exact: set[str] | None = None,
    ) -> str:
        if cand.empty:
            return ""
        forbid_exact = forbid_exact or set()
        ranked = cand.sort_values("bm25_score", ascending=False).copy()
        ranked["code"] = ranked["code"].astype(str).str.upper()

        def legal(code: str) -> bool:
            if code in forbid_exact:
                return False
            if forbid_prefixes and code.startswith(forbid_prefixes):
                return False
            return True

        for p in preferred:
            p = str(p).strip().upper()
            if not p:
                continue
            if FULL_CODE_RE.match(p):
                hit = ranked[ranked["code"] == p]
            else:
                hit = ranked[ranked["code"].str.startswith(p)]
            for _, rr in hit.iterrows():
                code = str(rr["code"])
                if legal(code):
                    return code
        for _, rr in ranked.iterrows():
            code = str(rr["code"])
            if legal(code):
                return code
        return ""

    def _apply_high_risk_gate(
        *,
        subject: str,
        problem: str,
        solution: str,
        concepts: list[str],
        final_code: str,
        cand: pd.DataFrame,
        default_code: str,
    ) -> tuple[str, str]:
        code = str(final_code).upper()
        if not code:
            return default_code, ""

        text = normalize_text(
            f"{problem}\n{solution}\n" + " ".join(str(x) for x in concepts if str(x).strip())
        )
        has_int = bool(
            re.search(
                r"\b(integer|integers|positive integer|nonnegative integer|natural number|integral)\b",
                text,
            )
        )
        has_eq = bool(re.search(r"\b(equation|equations|solve|solutions?|system)\b", text))
        has_lattice = "lattice point" in text
        has_dioph = "diophant" in text or has_lattice or (has_int and has_eq)

        has_seq_or_mod = bool(
            re.search(
                r"\b(sequence|series|term|recurrence|fibonacci|period|periodic|cycle|mod\b|congru|residue|remainder|units digit|last digit)\b",
                text,
            )
        )

        if subject == "Counting & Probability":
            pref = ["05A18", "05A10", "05A17", "05A15", "05C15", "11A05", "11A67", "11B75"]
        elif subject in {"Algebra", "Intermediate Algebra", "Prealgebra"}:
            pref = ["11D04", "11A05", "11A67", "11B25", "11B37", "15A06", "51M30", "05A10"]
        elif subject == "Geometry":
            pref = ["51M04", "51M25", "51N20", "51M30", "11A05", "11B25"]
        else:
            pref = ["26A09", "15A24", "51M30", "11A05", "11B25"]

        # Hard gate 1: Diophantine family must have integer-solution evidence.
        if code.startswith("11D") and not has_dioph:
            new_code = _pick_from_cand(
                cand,
                preferred=pref,
                forbid_prefixes=("11D",),
                forbid_exact={"11B50", "11B57"},
            )
            if not new_code:
                new_code = default_code if not str(default_code).startswith("11D") else code
            if new_code and new_code != code:
                return new_code, "gate_11d_requires_integer_constraints"

        # Hard gate 2: 11B57 / 11B50 need sequence/mod periodic evidence.
        if code in {"11B57", "11B50"} and not has_seq_or_mod:
            new_code = _pick_from_cand(
                cand,
                preferred=pref,
                forbid_exact={"11B50", "11B57"},
            )
            if not new_code:
                new_code = default_code
            if new_code and new_code != code:
                return new_code, f"gate_{code.lower()}_requires_sequence_or_mod_signal"

        return code, ""

    def choose_aux_code_with_llm(
        llm,
        qid: str,
        subject: str,
        problem: str,
        solution: str,
        final_code: str,
        final_desc: str,
        aux_cand: pd.DataFrame,
        aux_votes: int,
        max_aux_codes: int,
        strict_llm: bool,
        allow_fallback: bool,
    ) -> list[str]:
        if max_aux_codes <= 0 or aux_cand.empty:
            return [final_code]

        cand_codes = set(aux_cand["code"].astype(str).tolist())
        if llm is None:
            return [final_code]

        vote_n = max(1, int(aux_votes))
        votes = []
        prompt = build_aux_prompt(
            problem=problem,
            solution=solution,
            subject=subject,
            primary_code=final_code,
            primary_desc=final_desc,
            aux_candidates_df=aux_cand,
        )
        for _ in range(vote_n):
            try:
                obj = llm.json_completion(
                    system_prompt=AUX_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=0.0,
                )
                raw = str(obj.get("aux_code", "")).strip().upper()
                code = normalize_msc_full(raw, legal_codes) if raw != "NONE" else ""
                if code and code in cand_codes and code != final_code:
                    votes.append(code)
                else:
                    votes.append("")
            except Exception as e:  # noqa: BLE001
                if strict_llm and not allow_fallback:
                    raise RuntimeError(f"qid={qid} aux llm vote failed: {e}") from e
                votes.append("")

        cnt = Counter(votes)
        best_code, best_n = cnt.most_common(1)[0]
        maj = max(1, vote_n // 2 + 1)
        if best_code and best_n >= maj:
            return [final_code, best_code][: max_aux_codes + 1]
        return [final_code]

    def process_one(record: dict) -> tuple[list[dict], dict]:
        qid = str(record["qid"])
        subject = str(record["subject"])
        problem = str(record["problem"])
        solution = str(record["solution"])

        domain = subj_to_domain.get(subject, "algebra")
        default_l1 = str(subj_to_msc_l1.get(subject, "00")).zfill(2)

        query = f"{subject}\n{problem}\n{solution[:1200]}"
        cand = build_candidates(query, default_l1, subject)
        default_code = pick_default_code(default_l1, cand)

        max_concepts = int(ce.get("max_concepts_per_item", 8))
        min_concepts = int(ce.get("min_concepts_per_item", 3))

        votes = []
        parse_modes: list[str] = []
        used_fallback = False
        llm_errors: list[str] = []
        fallback_reasons: list[str] = []
        llm = get_llm_client()
        if llm is not None:
            for _ in range(int(ce.get("votes", 1))):
                allowed_hint = ""
                if allowed_set:
                    ap = sorted({c[:3] for c in allowed_set if len(c) >= 3})
                    allowed_hint = (
                        f"- Prefer elementary MSC prefixes: {', '.join(ap)}\n"
                        "- Avoid advanced topics (ideal, algebraic number field, Galois, scheme, sheaf, cohomology, "
                        "automorphic, p-adic, modular form, manifold, category).\n"
                    )
                prompt = build_prompt(
                    problem,
                    solution,
                    subject,
                    min_concepts,
                    max_concepts,
                    cand,
                    allowed_hint=allowed_hint,
                )
                try:
                    obj = llm.json_completion(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=prompt,
                        temperature=float(ce.get("temperature", 0.2)),
                    )
                    parse_modes.append(str(getattr(llm, "last_parse_mode", "unknown")))
                    msc_primary = normalize_msc_full(obj.get("msc_primary", ""), legal_codes)
                    if not msc_primary:
                        msc_primary = default_code
                    concepts = [normalize_text(x) for x in obj.get("concepts", []) if str(x).strip()]
                    concepts = [c for c in concepts if len(c) >= 3]
                except Exception as e:  # noqa: BLE001
                    llm_errors.append(str(e))
                    if strict_llm and not allow_fallback:
                        raise RuntimeError(f"qid={qid} llm concept vote failed: {e}") from e
                    used_fallback = True
                    fallback_reasons.append("llm_vote_error")
                    _, msc_primary, concepts = fallback_concepts(domain, subject, default_code)
                votes.append((msc_primary, concepts))
        else:
            if ce.get("enable_llm", True) and strict_llm and not allow_fallback:
                raise RuntimeError(f"qid={qid} llm client unavailable under strict_llm")
            used_fallback = True
            fallback_reasons.append("llm_disabled_or_unavailable")
            _, msc_primary, concepts = fallback_concepts(domain, subject, default_code)
            votes = [(msc_primary, concepts)]

        concept_counter: Counter[str] = Counter()
        code_counter: Counter[str] = Counter()
        for msc_primary, concepts in votes:
            code_counter[msc_primary] += 1
            concept_counter.update(concepts)

        vote_n = max(1, int(ce.get("votes", 1)))
        kept = [c for c, n in concept_counter.items() if n >= max(1, (vote_n // 2) + 1)]
        if not kept:
            used_fallback = True
            fallback_reasons.append("no_majority_concepts")
            kept = [c for c, _ in concept_counter.most_common(max(min_concepts, 4))]
        if len(kept) < min_concepts:
            used_fallback = True
            fallback_reasons.append("pad_to_min_concepts")
            _, _, backup = fallback_concepts(domain, subject, default_code)
            for c in backup:
                c = normalize_text(c)
                if c and c not in kept:
                    kept.append(c)
                if len(kept) >= min_concepts:
                    break

        final_code = code_counter.most_common(1)[0][0] if code_counter else default_code
        final_code = normalize_msc_full(final_code, legal_codes) or default_code

        if allowed_set and final_code not in allowed_set:
            # Try to pull a legal code from candidates or by L1.
            cand_allowed = cand[cand["code"].astype(str).isin(allowed_set)]
            if not cand_allowed.empty:
                final_code = str(cand_allowed.iloc[0]["code"])
            else:
                l1_hit = msc_df[msc_df["l1"].astype(str) == default_l1]
                if not l1_hit.empty:
                    final_code = str(l1_hit.iloc[0]["code"])
                else:
                    final_code = str(next(iter(allowed_set)))
        # Enforce strict subject whitelist for the final primary code.
        l1_allow, pref_allow = _subject_whitelist(subject)
        if l1_allow and final_code[:2] not in l1_allow:
            final_code = str(cand.iloc[0]["code"])
        if pref_allow and not any(final_code.startswith(p) for p in pref_allow):
            final_code = str(cand.iloc[0]["code"])
        gated_code, gate_reason = _apply_high_risk_gate(
            subject=subject,
            problem=problem,
            solution=solution,
            concepts=kept,
            final_code=final_code,
            cand=cand,
            default_code=default_code,
        )
        if gated_code != final_code:
            final_code = gated_code
            if gate_reason:
                fallback_reasons.append(gate_reason)
        if allowed_set and final_code not in allowed_set:
            final_code = str(cand.iloc[0]["code"])
        l1_allow, pref_allow = _subject_whitelist(subject)
        if l1_allow and final_code[:2] not in l1_allow:
            final_code = str(cand.iloc[0]["code"])
        if pref_allow and not any(final_code.startswith(p) for p in pref_allow):
            final_code = str(cand.iloc[0]["code"])
        max_codes_total = max(1, min(2, int(ce.get("max_msc_codes_per_item", 2))))
        max_aux_codes = max(0, min(1, max_codes_total - 1))
        aux_cand = build_aux_candidates(
            final_code=final_code,
            cand=cand,
            min_score_ratio=float(ce.get("msc_aux_min_score_ratio", 0.85)),
            min_abs_score=float(ce.get("msc_aux_min_abs_score", 0.05)),
            aux_policy=str(ce.get("msc_aux_policy", "same_prefix3")),
            topk=int(ce.get("msc_aux_candidate_topk", 6)),
        )
        msc_codes = choose_aux_code_with_llm(
            llm=llm,
            qid=qid,
            subject=subject,
            problem=problem,
            solution=solution,
            final_code=final_code,
            final_desc=code_to_desc.get(final_code, ""),
            aux_cand=aux_cand,
            aux_votes=int(ce.get("msc_aux_votes", 3)),
            max_aux_codes=max_aux_codes,
            strict_llm=strict_llm,
            allow_fallback=allow_fallback,
        )
        if allowed_set:
            msc_codes = [c for c in msc_codes if c in allowed_set]
            if not msc_codes:
                msc_codes = [final_code]
        final_l1 = code_to_l1.get(final_code, final_code[:2])
        final_desc = code_to_desc.get(final_code, "")
        parse_mode = parse_modes[-1] if parse_modes else "not_applicable"
        llm_error = " | ".join(llm_errors[:3]) if llm_errors else ""
        local_rows: list[dict] = []
        for c in kept:
            local_rows.append(
                {
                    "qid": qid,
                    "subject": subject,
                    "domain": domain,
                    "msc_full": final_code,
                    "msc_l1": final_l1,
                    "msc_desc": final_desc,
                    "concept": c,
                    "vote_count": int(concept_counter[c]),
                    "source": "llm" if llm is not None else "heuristic",
                    "used_fallback": bool(used_fallback),
                    "llm_error": llm_error,
                    "parse_mode": parse_mode,
                }
            )

        local_raw = {
            "qid": qid,
            "msc_full": final_code,
            "msc_codes": msc_codes,
            "msc_l1": final_l1,
            "msc_desc": final_desc,
            "concepts": kept,
            "domain": domain,
            "used_fallback": bool(used_fallback),
            "llm_error": llm_error,
            "parse_mode": parse_mode,
            "fallback_reasons": sorted(set(fallback_reasons)),
        }
        return local_rows, local_raw

    def flush_checkpoint() -> None:
        save_df(pd.DataFrame(rows).drop_duplicates(), mention_path)
        save_df(pd.DataFrame(raw_rows).drop_duplicates(subset=["qid"]), raw_path)

    if workers <= 1:
        for r in tqdm(seed.itertuples(index=False), total=len(seed), desc="extract concepts"):
            local_rows, local_raw = process_one(
                {
                    "qid": r.qid,
                    "subject": r.subject,
                    "problem": r.problem,
                    "solution": r.solution,
                }
            )
            rows.extend(local_rows)
            raw_rows.append(local_raw)
            if len(raw_rows) % flush_every == 0:
                flush_checkpoint()
    else:
        records = seed[["qid", "subject", "problem", "solution"]].to_dict("records")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_one, rec) for rec in records]
            for i, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc="extract concepts"), start=1):
                local_rows, local_raw = fut.result()
                rows.extend(local_rows)
                raw_rows.append(local_raw)
                if i % flush_every == 0:
                    flush_checkpoint()

    flush_checkpoint()

    if raw_rows:
        fb = int(sum(1 for r in raw_rows if bool(r.get("used_fallback", False))))
        fb_rate = float(fb) / float(len(raw_rows))
        err_n = int(sum(1 for r in raw_rows if str(r.get("llm_error", "")).strip()))
        llm_fb = 0
        for r in raw_rows:
            reasons = r.get("fallback_reasons", [])
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
            if any(x in {"llm_vote_error", "llm_disabled_or_unavailable"} for x in reasons):
                llm_fb += 1
            elif str(r.get("llm_error", "")).strip():
                llm_fb += 1
        llm_fb_rate = float(llm_fb) / float(len(raw_rows))
        print(
            f"[QC] fallback rows={fb}/{len(raw_rows)} ({fb_rate:.4f}), "
            + f"llm_fallback_rows={llm_fb}/{len(raw_rows)} ({llm_fb_rate:.4f}), "
            + f"llm_error_rows={err_n}"
        )
        if strict_llm and llm_fb_rate > float(max_fallback_rate):
            raise RuntimeError(
                f"strict_llm check failed: llm_fallback_rate={llm_fb_rate:.4f} > "
                + f"max_fallback_rate={float(max_fallback_rate):.4f}"
            )

    print(f"[OK] concept mentions: {mention_path}")
    print(f"[OK] concept sets: {raw_path}")


if __name__ == "__main__":
    main()
