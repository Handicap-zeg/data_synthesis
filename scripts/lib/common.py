from __future__ import annotations

import json
import ast
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(config_path).resolve().parent.parent
    cfg["_abs_project_root"] = str(root)

    paths = cfg.get("paths", {})
    for key, rel in paths.items():
        p = (root / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = str(p)
    cfg["paths"] = paths
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[`*_]", "", text)
    return text


def extract_boxed_answer(solution: str) -> str:
    if not isinstance(solution, str):
        return ""
    matches = re.findall(r"\\boxed\{([^{}]+)\}", solution)
    if matches:
        return matches[-1].strip()
    return ""


def math_level_to_y(level: Any) -> float:
    s = str(level).strip().lower() if level is not None else str()
    if s:
        for ch in s:
            if ch in set('12345'):
                lv = int(ch)
                return (lv - 1) / 4.0
    try:
        lv = int(level)
        if 1 <= lv <= 5:
            return (lv - 1) / 4.0
    except Exception:
        pass
    return 0.50


@dataclass
class LLMClient:
    base_url: str
    model: str
    api_key: str
    timeout_sec: float = 20.0

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("openai package is required only when enable_llm=true") from e

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_sec)
        self.last_parse_mode = "unknown"
        self.last_request_mode = "unknown"

    def json_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_retries: int = 1,
        sleep_sec: float = 1.0,
    ) -> dict[str, Any]:
        def _strip_reasoning(text: str) -> str:
            s = str(text or "")
            # Drop explicit reasoning blocks if the gateway leaks them.
            s = re.sub(r"<think>.*?</think>", " ", s, flags=re.IGNORECASE | re.DOTALL)
            s = re.sub(r"<analysis>.*?</analysis>", " ", s, flags=re.IGNORECASE | re.DOTALL)
            return s.strip()

        def _to_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for x in content:
                    if isinstance(x, dict):
                        t = x.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                        else:
                            parts.append(str(x))
                    else:
                        parts.append(str(x))
                return "\n".join(parts)
            if isinstance(content, dict):
                t = content.get("text")
                if isinstance(t, str):
                    return t
            return str(content)

        def _extract_json_blobs(text: str) -> list[str]:
            blobs: list[str] = []
            depth = 0
            start = -1
            in_str = False
            esc = False
            for i, ch in enumerate(text):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start >= 0:
                            blobs.append(text[start : i + 1])
                            start = -1
            return blobs

        def _parse_json_content(content: Any) -> tuple[dict[str, Any], str]:
            if isinstance(content, dict):
                return content, "dict"
            text = _strip_reasoning(_to_text(content))
            if not text:
                raise ValueError("empty LLM content")
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj, "json"
            except Exception:
                pass

            # Try fenced JSON first.
            fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
            for blob in fenced:
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict):
                        return obj, "fenced_json"
                except Exception:
                    pass
                try:
                    obj = ast.literal_eval(blob)
                    if isinstance(obj, dict):
                        return obj, "fenced_literal_eval"
                except Exception:
                    pass

            # Then try all {...} blobs from cleaned text, prefer later blobs.
            blobs = _extract_json_blobs(text)
            for blob in reversed(blobs):
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict):
                        return obj, "blob_json"
                except Exception:
                    pass
                try:
                    obj = ast.literal_eval(blob)
                    if isinstance(obj, dict):
                        return obj, "blob_literal_eval"
                except Exception:
                    pass
            raise ValueError(f"cannot parse JSON object: {text[:240]}")

        last_err = None
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        messages=messages,
                    )
                    self.last_request_mode = "response_format"
                except Exception as e1:  # noqa: BLE001
                    # Some OpenAI-compatible gateways do not support response_format.
                    if "response_format" in str(e1).lower():
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            temperature=temperature,
                            messages=messages,
                        )
                        self.last_request_mode = "plain_chat"
                    else:
                        raise
                content = resp.choices[0].message.content
                obj, parse_mode = _parse_json_content(content)
                self.last_parse_mode = parse_mode
                return obj
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(sleep_sec * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last_err}")


def build_llm_client(
    base_url: str,
    model: str,
    api_key_env: str | None = None,
    api_key: str | None = None,
    timeout_sec: float = 20.0,
) -> LLMClient:
    if api_key is None or not str(api_key).strip():
        if not api_key_env:
            raise RuntimeError("Missing api_key and api_key_env for LLM client")
        api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {api_key_env}")
    return LLMClient(base_url=base_url, model=model, api_key=api_key, timeout_sec=timeout_sec)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_df(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output path: {path}")


def load_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input path: {path}")
