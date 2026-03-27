from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sort_key(path: Path) -> tuple[str, int, str]:
    m = re.search(r"(\d+)", path.stem)
    idx = int(m.group(1)) if m else 10**9
    parent = path.parent.name
    return (parent, idx, path.name)


def _level_to_difficulty(level: Any) -> str:
    s = str(level or "").strip().lower()
    m = re.search(r"([1-5])", s)
    lv = int(m.group(1)) if m else 3
    if lv <= 2:
        return "easy"
    if lv == 3:
        return "medium"
    return "hard"


def _read_math_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"invalid json object: {path}")
    return obj


def _iter_math_rows(math_root: Path) -> list[dict[str, Any]]:
    paths = sorted(math_root.glob("*/*.json"), key=_sort_key)
    rows: list[dict[str, Any]] = []
    for path in paths:
        obj = _read_math_json(path)
        problem = str(obj.get("problem", "")).strip()
        if not problem:
            continue
        level = str(obj.get("level", "")).strip()
        subject = str(obj.get("type", path.parent.name)).strip()
        rows.append(
            {
                "problem": problem,
                "difficulty": _level_to_difficulty(level),
                "math_subject": subject,
                "math_level": level,
                "math_source_file": str(path),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_inputs(*, math_root: Path, input_dir: Path, shard_size: int = 100) -> list[Path]:
    rows = _iter_math_rows(math_root)
    if not rows:
        raise RuntimeError(f"no valid MATH rows found under {math_root}")
    if len(rows) % shard_size != 0:
        raise RuntimeError(f"row count {len(rows)} is not divisible by shard size {shard_size}")

    input_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for idx in range(0, len(rows), shard_size):
        shard_rows = rows[idx : idx + shard_size]
        shard_id = idx // shard_size + 1
        out_path = input_dir / f"math_train_{shard_id:02d}.jsonl"
        _write_jsonl(out_path, shard_rows)
        out_paths.append(out_path)
    return out_paths


def run_step11(
    *,
    project_root: Path,
    config_path: Path,
    input_dir: Path,
    out_prefix: Path,
    start_shard: int,
    end_shard: int,
    api_key: str,
    python_bin: str,
    easy_model: str,
    medium_model: str,
    hard_model: str,
    extra_args: list[str],
) -> None:
    input_paths = [input_dir / f"math_train_{i:02d}.jsonl" for i in range(start_shard, end_shard + 1)]
    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"missing prepared input shards: {missing[:10]}")

    cmd = [
        python_bin,
        str(project_root / "scripts" / "11_generate_answers.py"),
        "--config",
        str(config_path),
        "--out-prefix",
        str(out_prefix),
        "--start-index",
        str(start_shard),
        "--api-key",
        api_key,
        "--easy-model",
        easy_model,
        "--medium-model",
        medium_model,
        "--hard-model",
        hard_model,
        "--inputs",
    ]
    cmd.extend(str(p) for p in input_paths)
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True, cwd=str(project_root))


def main() -> None:
    default_project_root = Path("/home/dataset-local/usr/lh/zeg/data_synthesis")
    default_math_root = Path("/home/dataset-local/usr/lh/zeg/datasets/MATH/train")
    default_input_dir = Path("/home/dataset-local/usr/lh/zeg/datasets/MATH_step11_inputs")
    default_out_prefix = Path("/home/dataset-local/usr/lh/zeg/datasets/MATH_step11_answers/math_train")

    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(default_project_root))
    ap.add_argument("--math-root", default=str(default_math_root))
    ap.add_argument("--config", default=str(default_project_root / "configs" / "kg_math.yaml"))
    ap.add_argument("--input-dir", default=str(default_input_dir))
    ap.add_argument("--out-prefix", default=str(default_out_prefix))
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--run-only", action="store_true")
    ap.add_argument("--start-shard", type=int, default=1)
    ap.add_argument("--end-shard", type=int, default=75)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--easy-model", default="gemini-3-flash-preview-nothinking")
    ap.add_argument("--medium-model", default="gemini-3-flash-preview-nothinking")
    ap.add_argument("--hard-model", default="gemini-3-flash-preview-nothinking")
    ap.add_argument("--extra-step11-arg", action="append", default=[])
    args = ap.parse_args()

    if args.prepare_only and args.run_only:
        raise RuntimeError("cannot use --prepare-only and --run-only together")

    project_root = Path(args.project_root)
    math_root = Path(args.math_root)
    config_path = Path(args.config)
    input_dir = Path(args.input_dir)
    out_prefix = Path(args.out_prefix)

    if not args.run_only:
        paths = prepare_inputs(math_root=math_root, input_dir=input_dir)
        print(
            json.dumps(
                {
                    "prepared_shards": len(paths),
                    "input_dir": str(input_dir),
                    "first_shard": str(paths[0]),
                    "last_shard": str(paths[-1]),
                },
                ensure_ascii=False,
            )
        )

    if not args.prepare_only:
        if not str(args.api_key).strip():
            raise RuntimeError("missing --api-key for step11 run")
        run_step11(
            project_root=project_root,
            config_path=config_path,
            input_dir=input_dir,
            out_prefix=out_prefix,
            start_shard=int(args.start_shard),
            end_shard=int(args.end_shard),
            api_key=str(args.api_key).strip(),
            python_bin=str(args.python_bin),
            easy_model=str(args.easy_model),
            medium_model=str(args.medium_model),
            hard_model=str(args.hard_model),
            extra_args=list(args.extra_step11_arg),
        )


if __name__ == "__main__":
    main()
