from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.common import build_llm_client, load_config


@dataclass
class DiffSpec:
    name: str
    count: int
    target: float
    chain_len: int
    beam: int
    chain_retries: int
    max_resample: int
    tag_retries: int
    stage_a_attempts: int


def _read_first_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").splitlines()
    for ln in text:
        ln = ln.strip()
        if not ln:
            continue
        return json.loads(ln)
    return None


def _extract_q_text(problem: str) -> str:
    s = str(problem or "").strip()
    m = re.search(r"<Q>(.*?)</Q>", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1).strip()
        if t:
            return t
    return s


def _independent_answer(llm: Any, problem_text: str) -> dict[str, Any]:
    system_prompt = (
        "你有丰富的数学竞赛经验，你需要为初学者撰写一份详细的教程。"
        "你是一位有经验的数学老师，你需要帮助初学者学习这些数学问题。"
        "有一些具有挑战性的数学题，深呼吸，然后解决它们。"
        "你是一个知道很多数学问题的 AI 助手。"
        "只返回 JSON。"
    )
    user_prompt = f"""
看看下面的数学题，帮我解决它：
{problem_text}

返回 JSON：
{{
  "answer": "最终答案（尽量简洁）",
  "final_statement": "一句话总结最终结论"
}}
""".strip()
    return llm.json_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
    )


def _judge_outline_answer_consistency(
    llm: Any,
    problem_text: str,
    solution_outline: Any,
    answer: str,
) -> dict[str, Any]:
    system_prompt = "你是严格的数学一致性审查器。只返回 JSON。"
    user_prompt = f"""
请判断下面三者是否一致：
1) 题目
2) solution_outline 的最终结论
3) 给定 answer

题目：
{problem_text}

solution_outline：
{solution_outline}

answer：
{answer}

返回 JSON：
{{
  "consistent": true,
  "reason": "简短理由"
}}
""".strip()
    return llm.json_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out


def _build_specs(args: argparse.Namespace) -> list[DiffSpec]:
    return [
        DiffSpec(
            name="easy",
            count=int(args.n_easy),
            target=0.30,
            chain_len=2,
            beam=10,
            chain_retries=12,
            max_resample=4,
            tag_retries=2,
            stage_a_attempts=int(args.stage_a_attempts_easy),
        ),
        DiffSpec(
            name="medium",
            count=int(args.n_medium),
            target=0.55,
            chain_len=3,
            beam=12,
            chain_retries=16,
            max_resample=6,
            tag_retries=2,
            stage_a_attempts=int(args.stage_a_attempts_medium),
        ),
        DiffSpec(
            name="hard",
            count=int(args.n_hard),
            target=0.80,
            chain_len=4,
            beam=16,
            chain_retries=22,
            max_resample=8,
            tag_retries=2,
            stage_a_attempts=int(args.stage_a_attempts_hard),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-easy", type=int, default=5)
    ap.add_argument("--n-medium", type=int, default=1)
    ap.add_argument("--n-hard", type=int, default=2)
    ap.add_argument("--stage-a-workers", type=int, default=6)
    ap.add_argument("--stage-b-workers", type=int, default=4)
    ap.add_argument("--stage-a-timeout-sec", type=int, default=600)
    ap.add_argument("--stage-a-attempts-easy", type=int, default=12)
    ap.add_argument("--stage-a-attempts-medium", type=int, default=14)
    ap.add_argument("--stage-a-attempts-hard", type=int, default=18)
    ap.add_argument("--stage-b-retries", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=20260310)
    ap.add_argument("--stage-a-only", type=int, default=0)
    ap.add_argument("--stage-b-only", type=int, default=0)
    args = ap.parse_args()

    if int(args.stage_a_only) == 1 and int(args.stage_b_only) == 1:
        raise RuntimeError("stage-a-only and stage-b-only cannot both be 1")

    cfg = load_config(args.config)
    root = Path(cfg["_abs_project_root"])
    out_dir = root / "data" / "outputs"
    tmp_dir = out_dir / ".batch_stage_a_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    gen_script = root / "scripts" / "10_generate_questions.py"
    config_abs = Path(args.config)
    if not config_abs.is_absolute():
        config_abs = (Path.cwd() / config_abs).resolve()

    specs = [s for s in _build_specs(args) if s.count > 0]
    out_map = {
        "easy": out_dir / "synth_tmp_easy.jsonl",
        "medium": out_dir / "synth_tmp_medium.jsonl",
        "hard": out_dir / "synth_tmp_hard.jsonl",
    }

    rows_by_diff: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}

    def run_one(spec: DiffSpec, idx: int) -> dict[str, Any]:
        base_seed = int(args.random_seed) + {"easy": 100000, "medium": 200000, "hard": 300000}[spec.name] + idx * 100
        last_err = "unknown"
        for attempt in range(1, spec.stage_a_attempts + 1):
            seed = base_seed + attempt
            tmp_out = tmp_dir / f"{spec.name}_{idx}_{attempt}.jsonl"
            cmd = [
                sys.executable,
                str(gen_script),
                "--config",
                str(config_abs),
                "--n-samples",
                "1",
                "--chain-len",
                str(spec.chain_len),
                "--beam",
                str(spec.beam),
                "--chain-retries",
                str(spec.chain_retries),
                "--target-difficulty",
                str(spec.target),
                "--seed-per-node",
                "1",
                "--max-resample",
                str(spec.max_resample),
                "--tag-retries",
                str(spec.tag_retries),
                "--enforce-judge",
                "0",
                "--random-seed",
                str(seed),
                "--no-answer",
                "--out",
                str(tmp_out),
            ]
            try:
                cp = subprocess.run(
                    cmd,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=int(args.stage_a_timeout_sec),
                    check=False,
                )
            except Exception as e:  # noqa: BLE001
                last_err = f"timeout_or_subprocess_error: {e}"
                continue

            if cp.returncode != 0:
                tail = (cp.stderr or cp.stdout or "").strip().splitlines()
                last_err = tail[-1] if tail else f"returncode={cp.returncode}"
                continue

            row = _read_first_row(tmp_out)
            if not row:
                last_err = "empty_row_after_success"
                continue

            row["target_difficulty"] = spec.target
            row["difficulty_label"] = spec.name
            row["batch_meta"] = {
                "stage_a_attempt": attempt,
                "seed": seed,
            }
            return {"ok": True, "spec": spec.name, "row": row, "idx": idx, "attempt": attempt}

        return {"ok": False, "spec": spec.name, "idx": idx, "err": last_err}

    if int(args.stage_b_only) == 0:
        print("[INFO] Stage-A start")
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.stage_a_workers))) as ex:
            for spec in specs:
                for i in range(spec.count):
                    futures.append(ex.submit(run_one, spec, i))
            for fu in as_completed(futures):
                ret = fu.result()
                if ret["ok"]:
                    name = str(ret["spec"])
                    rows_by_diff[name].append(ret["row"])
                    print(f"[OK] stageA {name} idx={ret['idx']} attempt={ret['attempt']}")
                else:
                    print(f"[FAIL] stageA {ret['spec']} idx={ret['idx']} err={ret['err']}")

        for spec in specs:
            name = spec.name
            got = len(rows_by_diff[name])
            if got < spec.count:
                print(f"[WARN] stageA {name} got={got} < need={spec.count}")
            rows_by_diff[name] = rows_by_diff[name][: spec.count]
    else:
        print("[INFO] Stage-B only: load existing synth_tmp files")
        for name, path in out_map.items():
            rows = _read_jsonl(path)
            rows_by_diff[name] = rows
            print(f"[INFO] loaded {name}: {len(rows)} rows from {path}")

    if int(args.stage_a_only) == 0:
        print("[INFO] Stage-B start (answer + consistency)")
        synth_cfg = cfg.get("synthesis", {})
        api_cfg = cfg.get("pre_edge", {}) or cfg.get("concept_extraction", {})
        base_url = api_cfg.get("base_url", synth_cfg.get("base_url", ""))
        api_key_env = api_cfg.get("api_key_env", synth_cfg.get("api_key_env", "OPENAI_API_KEY"))
        api_key = api_cfg.get("api_key", synth_cfg.get("api_key"))
        model = "gpt-4o"

        tls = threading.local()

        def get_llm() -> Any:
            if getattr(tls, "llm", None) is None:
                tls.llm = build_llm_client(
                    base_url=base_url,
                    model=model,
                    api_key_env=api_key_env,
                    api_key=api_key,
                )
            return tls.llm

        def solve_one(diff_name: str, ridx: int, row: dict[str, Any]) -> dict[str, Any]:
            q_text = _extract_q_text(str(row.get("problem", "")))
            last_err = "unknown"
            for attempt in range(1, int(args.stage_b_retries) + 1):
                try:
                    llm = get_llm()
                    solve_obj = _independent_answer(llm=llm, problem_text=q_text)
                    ans = str(solve_obj.get("answer", "")).strip()
                    if not ans:
                        raise RuntimeError("empty answer")
                    cons_obj = _judge_outline_answer_consistency(
                        llm=llm,
                        problem_text=q_text,
                        solution_outline=row.get("solution_outline", []),
                        answer=ans,
                    )
                    consistent = bool(cons_obj.get("consistent", False))
                    if not consistent:
                        raise RuntimeError(str(cons_obj.get("reason", "consistency false")))
                    out = dict(row)
                    out["answer"] = ans
                    out["answer_solver"] = solve_obj
                    out["answer_consistency"] = cons_obj
                    bm = dict(out.get("batch_meta", {}))
                    bm["stage_b_attempt"] = attempt
                    out["batch_meta"] = bm
                    return {"ok": True, "diff": diff_name, "ridx": ridx, "row": out, "attempt": attempt}
                except Exception as e:  # noqa: BLE001
                    last_err = str(e)
                    continue
            return {"ok": False, "diff": diff_name, "ridx": ridx, "err": last_err}

        for diff_name in ["easy", "medium", "hard"]:
            rows = rows_by_diff[diff_name]
            if not rows:
                continue
            solved: list[dict[str, Any] | None] = [None] * len(rows)
            with ThreadPoolExecutor(max_workers=max(1, int(args.stage_b_workers))) as ex:
                f2 = [ex.submit(solve_one, diff_name, i, r) for i, r in enumerate(rows)]
                for fu in as_completed(f2):
                    ret = fu.result()
                    if ret["ok"]:
                        solved[int(ret["ridx"])] = ret["row"]
                        print(f"[OK] stageB {diff_name} idx={ret['ridx']} attempt={ret['attempt']}")
                    else:
                        print(f"[FAIL] stageB {ret['diff']} idx={ret['ridx']} err={ret['err']}")

            rows_by_diff[diff_name] = [r for r in solved if r is not None]

    for name, path in out_map.items():
        _write_jsonl(path, rows_by_diff[name])
        print(f"[OK] saved {name}: {path} rows={len(rows_by_diff[name])}")

    print("[OK] done")


if __name__ == "__main__":
    main()
