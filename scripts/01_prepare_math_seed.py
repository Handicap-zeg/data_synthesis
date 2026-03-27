from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.common import extract_boxed_answer, load_config, math_level_to_y, save_df, set_seed


def load_from_local_jsonl(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["seed"]["local_jsonl"])
    if not path.exists():
        raise FileNotFoundError(f"Local seed jsonl not found: {path}")

    df = pd.read_json(path, lines=True)
    df = df.rename(columns={"type": "subject"})
    for col in ["problem", "solution", "level", "subject"]:
        if col not in df.columns:
            df[col] = ""
    return df[["problem", "solution", "level", "subject"]].copy()


def load_from_local_math_dir(cfg: dict) -> pd.DataFrame:
    seed_cfg = cfg["seed"]
    root = Path(seed_cfg["local_math_dir"])
    split = str(seed_cfg.get("local_split", "train"))

    if not root.exists():
        raise FileNotFoundError(f"MATH directory not found: {root}")

    split_root = root / split
    scan_root = split_root if split_root.exists() else root

    rows = []
    for fp in scan_root.rglob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        problem = obj.get("problem", "")
        solution = obj.get("solution", "")
        level = obj.get("level", "")
        subject = obj.get("type") or obj.get("subject") or fp.parent.name

        rows.append(
            {
                "problem": str(problem),
                "solution": str(solution),
                "level": level,
                "subject": str(subject),
            }
        )

    if not rows:
        raise RuntimeError(f"No json samples found under: {scan_root}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["runtime"]["random_seed"])

    source = cfg["seed"]["source"]
    if source == "local_jsonl":
        df = load_from_local_jsonl(cfg)
    elif source == "local_math_dir":
        df = load_from_local_math_dir(cfg)
    else:
        raise ValueError(f"Unsupported seed source: {source}. Use local_math_dir or local_jsonl.")

    if cfg["seed"].get("limit"):
        df = df.head(int(cfg["seed"]["limit"]))

    df = df.reset_index(drop=True)
    df["qid"] = [f"MATH_{i:07d}" for i in range(len(df))]
    df["source"] = "MATH"
    df["answer"] = df["solution"].map(extract_boxed_answer)
    df["y"] = df["level"].map(math_level_to_y)

    out_cols = ["qid", "source", "subject", "level", "y", "problem", "solution", "answer"]
    out = df[out_cols]

    out_path = str(Path(cfg["paths"]["interim_dir"]) / "math_seed.parquet")
    save_df(out, out_path)

    print(f"[OK] saved seed to: {out_path}")
    print(f"[INFO] rows={len(out)} subjects={out['subject'].nunique()}")


if __name__ == "__main__":
    main()
