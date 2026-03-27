from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd

from lib.common import load_config, save_df


L1_RE = re.compile(r"^(\d{2})-XX\s+(.+)$")
L2_RE = re.compile(r"^(\d{2}[A-Z]xx)\s+(.+)$")
FULL_RE = re.compile(r"^(\d{2}[A-Z]\d{2})\s+(.+)$")


def pdftotext(pdf_path: str) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Failed to run pdftotext; ensure poppler-utils is installed") from e
    return proc.stdout


def clean_desc(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_msc(text: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    l1_rows = []
    l2_rows = []
    full_rows = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m1 = L1_RE.match(line)
        if m1:
            l1_rows.append({"l1": m1.group(1), "desc": clean_desc(m1.group(2))})
            continue

        m2 = L2_RE.match(line)
        if m2:
            code = m2.group(1)
            l2_rows.append({"code": code, "l1": code[:2], "desc": clean_desc(m2.group(2))})
            continue

        m3 = FULL_RE.match(line)
        if m3:
            code = m3.group(1)
            full_rows.append({"code": code, "l1": code[:2], "l2": code[:3] + "xx", "desc": clean_desc(m3.group(2))})
            continue

    l1 = pd.DataFrame(l1_rows).drop_duplicates(subset=["l1"]).sort_values("l1").reset_index(drop=True)
    l2 = pd.DataFrame(l2_rows).drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)
    full = pd.DataFrame(full_rows).drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)

    if l1.empty or full.empty:
        raise RuntimeError("Parsed MSC catalog is empty; check PDF quality and format")

    l1["l1"] = l1["l1"].astype(str).str.zfill(2)
    l2["code"] = l2["code"].astype(str)
    l2["l1"] = l2["l1"].astype(str).str.zfill(2)
    full["code"] = full["code"].astype(str)
    full["l1"] = full["l1"].astype(str).str.zfill(2)
    full["l2"] = full["l2"].astype(str)

    l1_map = dict(zip(l1["l1"], l1["desc"]))
    l2_map = dict(zip(l2["code"], l2["desc"])) if not l2.empty else {}

    full["l1_desc"] = full["l1"].map(l1_map).fillna("")
    full["l2_desc"] = full["l2"].map(l2_map).fillna("")

    return l1, l2, full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    msc_cfg = cfg.get("msc", {})

    pdf_path = str(msc_cfg.get("pdf_path", "")).strip()
    if not pdf_path:
        raise RuntimeError("Missing msc.pdf_path in config")
    pdf = Path(pdf_path)
    if not pdf.is_absolute():
        pdf = (Path(cfg["_abs_project_root"]) / pdf).resolve()
    pdf_path = str(pdf)

    catalog_csv = msc_cfg.get("catalog_csv", "data/interim/msc2020_codes.csv")
    base_root = Path(cfg["_abs_project_root"])
    out_path = Path(catalog_csv)
    if not out_path.is_absolute():
        out_path = (base_root / out_path).resolve()

    text = pdftotext(pdf_path)
    l1, l2, full = parse_msc(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)

    # side outputs for debugging/reference
    l1_path = out_path.with_name("msc2020_l1.csv")
    l2_path = out_path.with_name("msc2020_l2.csv")
    l1.to_csv(l1_path, index=False)
    l2.to_csv(l2_path, index=False)

    print(f"[OK] MSC full codes: {len(full)} -> {out_path}")
    print(f"[OK] MSC l1: {len(l1)} -> {l1_path}")
    print(f"[OK] MSC l2: {len(l2)} -> {l2_path}")


if __name__ == "__main__":
    main()
