from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from lib.common import load_config, load_df, normalize_text, save_df


FULL_CODE_RE = re.compile(r"^\d{2}[A-Za-z]\d{2}$")


def _parse_list_like(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            out = value.tolist()
            if isinstance(out, (list, tuple, set)):
                return [str(x) for x in out]
            return [str(out)]
        except Exception:  # noqa: BLE001
            return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [x.strip().strip("'\"") for x in text.split(",") if x.strip()]


def _norm_concepts(value: Any) -> list[str]:
    out: list[str] = []
    for x in _parse_list_like(value):
        c = normalize_text(str(x))
        if c:
            out.append(c)
    # keep order, drop duplicates
    dedup: list[str] = []
    seen: set[str] = set()
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def _norm_codes(value: Any) -> list[str]:
    out: list[str] = []
    for x in _parse_list_like(value):
        c = str(x).strip().upper()
        if FULL_CODE_RE.match(c):
            out.append(c)
    dedup: list[str] = []
    seen: set[str] = set()
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def _subject_allows(subject: str, code: str, subj_l1: dict[str, list[str]], subj_pref: dict[str, list[str]]) -> bool:
    l1s = {str(x).zfill(2) for x in subj_l1.get(subject, []) if str(x).strip()}
    prefs = [str(x).upper() for x in subj_pref.get(subject, []) if str(x).strip()]
    l1_ok = (not l1s) or (code[:2] in l1s)
    pref_ok = (not prefs) or any(code.startswith(p) for p in prefs)
    return l1_ok and pref_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-codes", default="11D72,11B57,11B50")
    parser.add_argument("--target-subjects", default="Algebra,Intermediate Algebra,Prealgebra,Counting & Probability")
    parser.add_argument("--min-code-freq", type=int, default=8)
    parser.add_argument("--min-profile-rows", type=int, default=12)
    parser.add_argument("--top-concepts-per-code", type=int, default=24)
    parser.add_argument("--alpha-concept", type=float, default=0.80)
    parser.add_argument("--beta-prior", type=float, default=0.20)
    parser.add_argument("--min-new-score", type=float, default=0.34)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--min-subject-code-freq", type=int, default=5)
    parser.add_argument("--min-prior-gain", type=float, default=1.20)
    parser.add_argument("--min-new-prior", type=float, default=0.01)
    parser.add_argument("--max-change-rate", type=float, default=0.80)
    parser.add_argument("--apply", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    interim = Path(cfg["paths"]["interim_dir"])

    cs_path = str(interim / "concept_sets.parquet")
    cm_path = str(interim / "concept_mentions.parquet")
    seed_path = str(interim / "math_seed.parquet")
    msc_path = str(cfg["msc"]["catalog_csv"])

    cs = load_df(cs_path)
    cm = load_df(cm_path)
    seed = load_df(seed_path)[["qid", "subject"]]
    msc = load_df(msc_path)[["code", "l1", "desc"]].copy()
    msc["code"] = msc["code"].astype(str).str.upper()
    code_to_l1 = dict(zip(msc["code"], msc["l1"].astype(str)))
    code_to_desc = dict(zip(msc["code"], msc["desc"].astype(str)))

    df = cs.merge(seed, on="qid", how="left")
    df["msc_full"] = df["msc_full"].astype(str).str.upper()
    df["concept_list"] = df["concepts"].map(_norm_concepts)
    df["used_fallback"] = df["used_fallback"].fillna(False).astype(bool)

    target_codes = {x.strip().upper() for x in str(args.target_codes).split(",") if x.strip()}
    target_subjects = {x.strip() for x in str(args.target_subjects).split(",") if x.strip()}
    if not target_codes:
        raise RuntimeError("empty --target-codes")

    subj_l1 = cfg.get("mappings", {}).get("subject_to_allowed_l1", {})
    subj_pref = cfg.get("mappings", {}).get("subject_to_allowed_prefix", {})

    reliable = df[~df["used_fallback"]].copy()
    code_freq = reliable["msc_full"].value_counts().to_dict()

    # Build concept profiles and subject-code priors from reliable rows.
    profile_counter: dict[str, Counter[str]] = defaultdict(Counter)
    profile_rows: Counter[str] = Counter()
    subj_code: Counter[tuple[str, str]] = Counter()
    subj_tot: Counter[str] = Counter()
    for r in reliable.itertuples(index=False):
        code = str(r.msc_full)
        sub = str(r.subject)
        subj_code[(sub, code)] += 1
        subj_tot[sub] += 1
        profile_rows[code] += 1
        for c in r.concept_list:
            profile_counter[code][c] += 1

    profile_top: dict[str, set[str]] = {}
    for code, cnt in profile_counter.items():
        if profile_rows[code] < int(args.min_profile_rows):
            continue
        top = [k for k, _ in cnt.most_common(int(args.top_concepts_per_code))]
        if top:
            profile_top[code] = set(top)

    candidate_codes = {
        c
        for c, f in code_freq.items()
        if int(f) >= int(args.min_code_freq) and c in profile_top
    }
    print(f"[INFO] candidate_codes={len(candidate_codes)} (freq>={args.min_code_freq})")

    def score(subject: str, row_concepts: set[str], code: str, denom: int) -> tuple[float, float, float]:
        top = profile_top.get(code, set())
        c_score = 0.0
        if row_concepts and top:
            c_score = len(row_concepts & top) / max(1, len(row_concepts))
        prior = (subj_code[(subject, code)] + 1.0) / max(1.0, float(subj_tot[subject] + denom))
        total = float(args.alpha_concept) * c_score + float(args.beta_prior) * prior
        return total, c_score, prior

    target = df[df["msc_full"].isin(target_codes) & df["subject"].isin(target_subjects)].copy()
    print(f"[INFO] target_rows={len(target)} target_codes={sorted(target_codes)}")

    changes: list[dict[str, Any]] = []
    for r in target.itertuples(index=False):
        subject = str(r.subject)
        old_code = str(r.msc_full)
        row_concepts = set(r.concept_list)
        if not row_concepts:
            continue

        allowed = [
            c
            for c in candidate_codes
            if _subject_allows(subject, c, subj_l1, subj_pref)
            and subj_code[(subject, c)] >= int(args.min_subject_code_freq)
        ]
        if old_code not in allowed and old_code in profile_top and _subject_allows(subject, old_code, subj_l1, subj_pref):
            allowed.append(old_code)
        if not allowed:
            continue

        denom = max(1, len(allowed))
        old_score, old_cs, old_prior = score(subject, row_concepts, old_code, denom)
        scored: list[tuple[float, str, float, float]] = []
        for c in allowed:
            s_all, s_c, s_p = score(subject, row_concepts, c, denom)
            scored.append((s_all, c, s_c, s_p))
        scored.sort(reverse=True)
        best_score, best_code, best_cs, best_prior = scored[0]

        if best_code == old_code:
            continue
        if best_score < float(args.min_new_score):
            continue
        if best_score - old_score < float(args.min_margin):
            continue
        if best_prior < float(args.min_new_prior):
            continue
        if old_prior > 0 and best_prior < float(args.min_prior_gain) * old_prior:
            continue
        changes.append(
            {
                "qid": str(r.qid),
                "subject": subject,
                "old_code": old_code,
                "new_code": best_code,
                "old_score": round(old_score, 6),
                "new_score": round(best_score, 6),
                "old_concept_score": round(old_cs, 6),
                "new_concept_score": round(best_cs, 6),
                "old_prior": round(old_prior, 6),
                "new_prior": round(best_prior, 6),
                "concepts": list(row_concepts),
            }
        )

    change_df = pd.DataFrame(changes)
    if change_df.empty:
        print("[OK] no changes proposed.")
        return

    change_rate = len(change_df) / max(1, len(target))
    print(f"[INFO] proposed_changes={len(change_df)} / {len(target)} ({change_rate:.4f})")
    print("[INFO] top transitions:")
    print(change_df.groupby(["old_code", "new_code"]).size().sort_values(ascending=False).head(20).to_string())
    print("[INFO] sample:")
    print(change_df.head(12).to_string(index=False))

    log_path = str(interim / "code_repair_log.csv")
    change_df.drop(columns=["concepts"]).to_csv(log_path, index=False)
    print(f"[OK] saved log: {log_path}")

    if int(args.apply) != 1:
        print("[DRY-RUN] no files changed. add --apply 1 to write changes.")
        return

    if change_rate > float(args.max_change_rate):
        raise RuntimeError(
            f"change_rate {change_rate:.4f} exceeds max_change_rate {float(args.max_change_rate):.4f}; aborting apply"
        )

    qid_to_new = dict(zip(change_df["qid"], change_df["new_code"]))
    qid_set = set(qid_to_new.keys())

    cs2 = cs.copy()
    cs2["qid"] = cs2["qid"].astype(str)
    mask_cs = cs2["qid"].isin(qid_set)
    cs2.loc[mask_cs, "msc_full"] = cs2.loc[mask_cs, "qid"].map(qid_to_new)
    cs2.loc[mask_cs, "msc_l1"] = cs2.loc[mask_cs, "msc_full"].astype(str).str[:2]
    cs2.loc[mask_cs, "msc_desc"] = cs2.loc[mask_cs, "msc_full"].map(code_to_desc).fillna(cs2.loc[mask_cs, "msc_desc"])

    def remap_codes(row: pd.Series) -> list[str]:
        qid = str(row["qid"])
        new_code = qid_to_new.get(qid, "")
        old_codes = _norm_codes(row.get("msc_codes"))
        if not new_code:
            return old_codes
        keep = [c for c in old_codes if c != new_code]
        out = [new_code] + keep
        # cap to two codes to stay consistent with current pipeline policy
        return out[:2]

    cs2.loc[mask_cs, "msc_codes"] = cs2.loc[mask_cs].apply(remap_codes, axis=1)

    cm2 = cm.copy()
    cm2["qid"] = cm2["qid"].astype(str)
    mask_cm = cm2["qid"].isin(qid_set)
    cm2.loc[mask_cm, "msc_full"] = cm2.loc[mask_cm, "qid"].map(qid_to_new)
    cm2.loc[mask_cm, "msc_l1"] = cm2.loc[mask_cm, "msc_full"].astype(str).str[:2]
    cm2.loc[mask_cm, "msc_desc"] = cm2.loc[mask_cm, "msc_full"].map(code_to_desc).fillna(cm2.loc[mask_cm, "msc_desc"])

    save_df(cs2, cs_path)
    save_df(cm2, cm_path)
    print(f"[OK] applied changes: concept_sets={mask_cs.sum()}, concept_mentions={mask_cm.sum()}")


if __name__ == "__main__":
    main()
