from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from lib.common import build_llm_client, load_config, load_df, save_df

SYSTEM_PROMPT = (
    "You judge prerequisite direction between two math sub-concepts for elementary/competition-style problems. "
    "Be precise but not overly strict. Return strict JSON only."
)


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


def _build_qsets(seeds_sub: pd.DataFrame) -> dict[str, set[str]]:
    s = seeds_sub[["subnode_id", "qid"]].copy()
    s["subnode_id"] = s["subnode_id"].astype(str)
    s["qid"] = s["qid"].astype(str)
    grp = s.groupby("subnode_id")["qid"].apply(lambda x: set(x.tolist()))
    return {str(k): set(v) for k, v in grp.to_dict().items()}


def _build_prompt(a: dict, b: dict, pair_cooccur: int, sim_concept: float | None, difficulty_gap: float | None) -> str:
    sim_txt = "" if sim_concept is None else f"{sim_concept:.3f}"
    dgap_txt = "" if difficulty_gap is None else f"{difficulty_gap:.3f}"
    return f"""
Decide whether Subconcept A should be a prerequisite of Subconcept B.

Return JSON only:
{{
  "allow": "yes|no",
  "confidence": 0.0,
  "reason": "short"
}}

Subconcept A (candidate prerequisite):
- subnode_id: {a.get("subnode_id", "")}
- parent_node_id: {a.get("parent_node_id", "")}
- msc_full: {a.get("msc_full", "")}
- domain: {a.get("domain", "")}
- concept_cluster: {a.get("concept_cluster", "")}
- freq: {a.get("freq", 0)}

Subconcept B (candidate dependent):
- subnode_id: {b.get("subnode_id", "")}
- parent_node_id: {b.get("parent_node_id", "")}
- msc_full: {b.get("msc_full", "")}
- domain: {b.get("domain", "")}
- concept_cluster: {b.get("concept_cluster", "")}
- freq: {b.get("freq", 0)}

Observed pair signals:
- shared_seed_count: {pair_cooccur}
- sim_concept: {sim_txt}
- difficulty_gap(B-A): {dgap_txt}

Rules:
- allow=yes if A is commonly a useful foundation for B in high-school/competition settings.
- If A and B are mostly parallel with no clear directional dependence, return no.
- For weak-but-plausible prerequisite signals, you may return yes with lower confidence.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--votes", type=int, default=-1)
    parser.add_argument("--min-conf", type=float, default=-1.0)
    parser.add_argument("--pass-policy", choices=["majority", "at_least_one"], default="majority")
    parser.add_argument("--target-keep", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph_dir = Path(cfg["paths"]["graph_dir"])

    in_path = Path(args.input) if args.input else (graph_dir / "edges_pre_sub.parquet")
    out_path = Path(args.output) if args.output else in_path

    pre_cfg = cfg.get("pre_edge", {})
    sub_pre_cfg = cfg.get("sub_pre_edge", {})

    enable_llm = bool(sub_pre_cfg.get("enable_llm_recheck", True)) and bool(pre_cfg.get("enable_llm", True))
    votes = int(sub_pre_cfg.get("llm_recheck_votes", pre_cfg.get("votes", 1)))
    temp = float(sub_pre_cfg.get("llm_recheck_temperature", pre_cfg.get("temperature", 0.0)))
    min_conf = float(sub_pre_cfg.get("llm_recheck_min_conf", pre_cfg.get("min_conf", 0.78)))
    workers = int(sub_pre_cfg.get("llm_recheck_workers", pre_cfg.get("workers", 8)))

    if args.workers is not None and int(args.workers) > 0:
        workers = int(args.workers)
    if args.votes is not None and int(args.votes) > 0:
        votes = int(args.votes)
    if args.min_conf is not None and float(args.min_conf) > 0:
        min_conf = float(args.min_conf)
    pass_policy = str(args.pass_policy).strip().lower()
    target_keep = int(args.target_keep)
    progress_every = max(1, int(args.progress_every))

    if not enable_llm:
        raise RuntimeError("LLM recheck disabled by config (sub_pre_edge.enable_llm_recheck or pre_edge.enable_llm)")

    base_url = str(pre_cfg.get("base_url", ""))
    model = str(pre_cfg.get("model", ""))
    api_key_env = pre_cfg.get("api_key_env")
    api_key = pre_cfg.get("api_key")

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    edges = load_df(str(in_path)).copy()
    if edges.empty:
        print("[WARN] input pre edges empty; nothing to recheck")
        save_df(edges, str(out_path))
        return

    subnodes = _prepare_subnodes(load_df(str(graph_dir / "subnodes.parquet")))
    seeds_sub = load_df(str(graph_dir / "seeds_sub_index.parquet"))
    qsets = _build_qsets(seeds_sub)

    info = {
        str(r.subnode_id): {
            "subnode_id": str(r.subnode_id),
            "parent_node_id": str(r.parent_node_id),
            "msc_full": str(r.msc_full),
            "domain": str(r.domain),
            "concept_cluster": str(r.concept_cluster),
            "freq": float(r.freq),
        }
        for r in subnodes.itertuples(index=False)
    }

    thread_local = threading.local()

    def _get_llm():
        if not hasattr(thread_local, "llm"):
            thread_local.llm = build_llm_client(
                base_url=base_url,
                model=model,
                api_key_env=api_key_env,
                api_key=api_key,
            )
        return thread_local.llm

    def _judge_one(rec: dict) -> dict:
        src = str(rec.get("src", ""))
        dst = str(rec.get("dst", ""))
        a = info.get(src, {"subnode_id": src})
        b = info.get(dst, {"subnode_id": dst})
        co = int(len(qsets.get(src, set()) & qsets.get(dst, set())))

        sim = rec.get("sim_concept")
        dgap = rec.get("difficulty_gap")
        try:
            sim = float(sim) if sim is not None else None
        except Exception:
            sim = None
        try:
            dgap = float(dgap) if dgap is not None else None
        except Exception:
            dgap = None

        prompt = _build_prompt(a, b, co, sim, dgap)
        llm = _get_llm()

        allow_votes = 0
        confs: list[float] = []
        reasons: list[str] = []

        for _ in range(max(1, votes)):
            try:
                obj = llm.json_completion(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=temp,
                )
                allow_raw = str(obj.get("allow", "no")).strip().lower()
                allow = allow_raw in {"yes", "true", "1", "allow"}
                c = float(obj.get("confidence", 0.0))
                reason = str(obj.get("reason", "")).strip()
            except Exception:
                allow = False
                c = 0.0
                reason = ""

            if allow:
                allow_votes += 1
            confs.append(max(0.0, min(1.0, c)))
            if reason:
                reasons.append(reason)

        vote_conf = allow_votes / max(1, votes)
        conf_mean = sum(confs) / max(1, len(confs))
        llm_conf = 0.6 * vote_conf + 0.4 * conf_mean

        if pass_policy == "at_least_one":
            pass_votes = allow_votes >= 1
        else:
            pass_votes = allow_votes > (votes // 2)
        allow_final = bool(pass_votes and (llm_conf >= min_conf))

        return {
            "llm_allow": bool(allow_final),
            "llm_conf": float(llm_conf),
            "llm_vote": float(vote_conf),
            "llm_reason": reasons[0] if reasons else "",
        }

    recs = edges.to_dict(orient="records")
    judged: list[dict] = []
    total = len(recs)
    t0 = time.time()
    print(
        f"[INFO] pre recheck start: total={total}, workers={workers}, votes={votes}, min_conf={min_conf}, pass_policy={pass_policy}, target_keep={target_keep}",
        flush=True,
    )

    if workers > 1:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_judge_one, r): r for r in recs}
            for fut in as_completed(futs):
                base = dict(futs[fut])
                try:
                    base.update(fut.result())
                except Exception as e:  # noqa: BLE001
                    base.update(
                        {
                            "llm_allow": False,
                            "llm_conf": 0.0,
                            "llm_vote": 0.0,
                            "llm_reason": f"future_error:{e}",
                        }
                    )
                judged.append(base)
                done += 1
                if done % progress_every == 0 or done == total:
                    dt = max(1e-8, time.time() - t0)
                    print(f"[INFO] progress: {done}/{total} ({done/total:.1%}), rate={done/dt:.2f}/s", flush=True)
    else:
        for i, r in enumerate(recs, start=1):
            row = dict(r)
            row.update(_judge_one(r))
            judged.append(row)
            if i % progress_every == 0 or i == total:
                dt = max(1e-8, time.time() - t0)
                print(f"[INFO] progress: {i}/{total} ({i/total:.1%}), rate={i/dt:.2f}/s", flush=True)

    out = pd.DataFrame(judged)
    before = len(out)

    if "conf" in out.columns:
        out["conf_rule"] = pd.to_numeric(out["conf"], errors="coerce").fillna(0.0)
    else:
        out["conf_rule"] = 0.0

    if target_keep > 0:
        ranked = out.sort_values(["llm_conf", "llm_vote", "conf_rule"], ascending=False).reset_index(drop=True)
        k = min(int(target_keep), len(ranked))
        out_yes = ranked[ranked["llm_vote"] > 0.0].copy()
        if len(out_yes) >= k:
            ys = out_yes.sort_values(["llm_conf", "conf_rule"], ascending=False).reset_index(drop=True)
            cutoff = float(ys.loc[k - 1, "llm_conf"])
            out = ys[ys["llm_conf"] >= cutoff].copy()
        else:
            need = k - len(out_yes)
            out_no = ranked[ranked["llm_vote"] <= 0.0].head(need).copy()
            out = pd.concat([out_yes, out_no], ignore_index=True)
        print(f"[INFO] target_keep active: target={target_keep}, actual={len(out)}", flush=True)
    else:
        out = out[out["llm_allow"] == True].copy()  # noqa: E712

    kept = len(out)
    out["conf"] = 0.7 * pd.to_numeric(out["conf_rule"], errors="coerce").fillna(0.0) + 0.3 * pd.to_numeric(
        out["llm_conf"], errors="coerce"
    ).fillna(0.0)

    out = out.sort_values(["conf", "llm_conf"], ascending=False).reset_index(drop=True)
    save_df(out, str(out_path))

    dt = max(1e-8, time.time() - t0)
    print(f"[OK] pre LLM recheck done: before={before}, kept={kept}, dropped={before-kept}, sec={dt:.1f}", flush=True)
    print(f"[OK] saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
