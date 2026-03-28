from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from lib.common import load_config, read_jsonl


ANSWER_SCHEMA = {
    "verdict": "ok|wrong_problem",
    "key_idea": "...",
    "solution_steps": [
        "Step 1: explain the setup and why this method is appropriate",
        "Step 2: carry out the derivation with enough detail to reproduce it"
    ],
    "check": "...",
    "final_answer": "...",
}


DEFAULT_LOCAL_MODEL_PATH = "/jizhicfs/zeg/models/QwQ-32B"
_LOCAL_HF_RUNTIME_CACHE: dict[tuple[str, str, str, str, bool], dict[str, Any]] = {}


class AnswerGenerationError(RuntimeError):
    def __init__(self, message: str, *, debug: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.debug = dict(debug or {})


class JsonlShardWriter:
    def __init__(self, out_prefix: Path, shard_size: int = 100, start_index: int = 1) -> None:
        self.shard_size = max(1, int(shard_size))
        self.out_dir = out_prefix.parent
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = out_prefix.stem
        self.shard_index = max(1, int(start_index))
        self.current_count = self._count_rows(self.current_path())
        self.total_appended = 0
        self.touched_shards: set[int] = set()

    def _count_rows(self, path: Path) -> int:
        if not path.exists():
            return 0
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def current_path(self) -> Path:
        return self.out_dir / f"{self.prefix}_{self.shard_index:02d}.jsonl"

    def _sync(self) -> None:
        while True:
            path = self.current_path()
            actual = self._count_rows(path)
            if actual != self.current_count:
                self.current_count = actual
            if self.current_count < self.shard_size:
                return
            self.shard_index += 1
            self.current_count = 0

    def append_row(self, row: dict[str, Any]) -> Path:
        self._sync()
        path = self.current_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.current_count += 1
        self.total_appended += 1
        self.touched_shards.add(self.shard_index)
        return path

    def touched_paths(self) -> list[str]:
        return [str(self.out_dir / f"{self.prefix}_{idx:02d}.jsonl") for idx in sorted(self.touched_shards)]


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_input_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(str(path)):
            if not isinstance(row, dict):
                continue
            row = dict(row)
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def _decode_escaped(s: str) -> str:
    txt = str(s or "")
    out: list[str] = []
    i = 0
    while i < len(txt):
        ch = txt[i]
        if ch != "\\" or i + 1 >= len(txt):
            out.append(ch)
            i += 1
            continue
        nxt = txt[i + 1]
        if nxt in {'\\', '"', "/"}:
            out.append(nxt)
            i += 2
            continue
        if nxt == "n":
            if i + 2 < len(txt) and txt[i + 2].isalpha():
                out.append("\\n")
                i += 2
                continue
            out.append("\n")
            i += 2
            continue
        if nxt == "r":
            if i + 2 < len(txt) and txt[i + 2].isalpha():
                out.append("\\r")
                i += 2
                continue
            out.append("\r")
            i += 2
            continue
        if nxt == "t":
            if i + 2 < len(txt) and txt[i + 2].isalpha():
                out.append("\\t")
                i += 2
                continue
            out.append("\t")
            i += 2
            continue
        if nxt == "b":
            if i + 2 < len(txt) and txt[i + 2].isalpha():
                out.append("\\b")
                i += 2
                continue
            out.append("\b")
            i += 2
            continue
        if nxt == "f":
            if i + 2 < len(txt) and txt[i + 2].isalpha():
                out.append("\\f")
                i += 2
                continue
            out.append("\f")
            i += 2
            continue
        if nxt == "u" and i + 5 < len(txt):
            code = txt[i + 2 : i + 6]
            if re.fullmatch(r"[0-9a-fA-F]{4}", code):
                out.append(chr(int(code, 16)))
                i += 6
                continue
        out.append("\\")
        out.append(nxt)
        i += 2
    return "".join(out).strip()


def _extract_string_value(src: str, key: str) -> str:
    marker = '"' + key + '"'
    k = src.find(marker)
    if k < 0:
        return ""
    colon = src.find(':', k + len(marker))
    if colon < 0:
        return ""
    q1 = src.find('"', colon + 1)
    if q1 < 0:
        return ""
    out: list[str] = []
    esc = False
    i = q1 + 1
    while i < len(src):
        ch = src[i]
        if esc:
            out.append("\\")
            out.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip()


def _extract_string_array(src: str, key: str) -> list[str]:
    marker = '"' + key + '"'
    k = src.find(marker)
    if k < 0:
        return []
    lb = src.find('[', k + len(marker))
    if lb < 0:
        return []
    items: list[str] = []
    i = lb + 1
    while i < len(src):
        ch = src[i]
        if ch == ']':
            break
        if ch in ' \t\r\n,':
            i += 1
            continue
        if ch != '"':
            break
        i += 1
        out: list[str] = []
        esc = False
        while i < len(src):
            ch = src[i]
            if esc:
                out.append("\\")
                out.append(ch)
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                break
            else:
                out.append(ch)
            i += 1
        items.append(_decode_escaped("".join(out)))
        i += 1
    return [x for x in items if x]


def _normalize_key_steps(value: Any) -> list[str]:
    def _clean_step_text(text: str) -> str:
        s = str(text or "").strip()
        s = re.sub(r"^(?:step|步骤)\s*\d+\s*[:.)-]?\s*", "", s, flags=re.IGNORECASE)
        return s.strip()

    if isinstance(value, list):
        out = [_clean_step_text(str(x)) for x in value if _clean_step_text(str(x))]
        return out
    if isinstance(value, str) and value.strip():
        parts = [_clean_step_text(x) for x in value.splitlines() if _clean_step_text(x)]
        return parts if parts else [_clean_step_text(value)]
    return []


def _normalize_short_text(value: Any) -> str:
    return str(value or "").strip()


def _strip_field_label(text: str, label: str) -> str:
    s = str(text or "").strip()
    return re.sub(rf"^{re.escape(label)}\s*:?\s*", "", s, flags=re.IGNORECASE).strip()


def _extract_json_object(text: str) -> tuple[dict[str, Any], str]:
    s = str(text or "").strip()
    if not s:
        raise RuntimeError("empty_response_text")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, "direct_json"
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.IGNORECASE | re.DOTALL)
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

    depth = 0
    start = -1
    in_str = False
    esc = False
    blobs: list[str] = []
    for i, ch in enumerate(s):
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
                    blobs.append(s[start : i + 1])
                    start = -1
    for blob in reversed(blobs):
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj, "balanced_json"
        except Exception:
            pass
        try:
            obj = ast.literal_eval(blob)
            if isinstance(obj, dict):
                return obj, "balanced_literal_eval"
        except Exception:
            pass

    final_answer_raw = _extract_string_value(s, "final_answer")
    key_steps_raw = _extract_string_array(s, "key_steps")
    solution_steps_raw = _extract_string_array(s, "solution_steps")
    verdict_raw = _extract_string_value(s, "verdict")
    key_idea_raw = _extract_string_value(s, "key_idea")
    check_raw = _extract_string_value(s, "check")
    if final_answer_raw:
        final_answer = _decode_escaped(final_answer_raw)
        key_steps = solution_steps_raw or key_steps_raw or ["Key steps truncated by gateway; final answer preserved."]
        out = {
            "verdict": _decode_escaped(verdict_raw) or ("wrong_problem" if final_answer == "wrong problem" else "ok"),
            "key_idea": _decode_escaped(key_idea_raw),
            "solution_steps": key_steps,
            "check": _decode_escaped(check_raw),
            "final_answer": final_answer,
        }
        return out, "string_fallback"
    raise RuntimeError(f"gateway_invalid_json: {s[:500]}")


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text", "")
                if txt:
                    parts.append(str(txt))
            elif item:
                parts.append(str(item))
        return "\n".join(x.strip() for x in parts if str(x).strip()).strip()
    return str(content or "").strip()


def _resolve_torch_dtype(name: str) -> Any:
    import torch

    key = str(name or "bfloat16").strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    if key == "auto":
        return "auto"
    raise RuntimeError(f"unsupported torch dtype: {name}")


def _safe_context_window(model: Any, tokenizer: Any) -> int:
    candidates: list[int] = []
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    for value in (
        getattr(config, "max_position_embeddings", None),
        getattr(config, "sliding_window", None),
        getattr(generation_config, "max_length", None),
        tokenizer_limit,
    ):
        try:
            iv = int(value)
        except Exception:
            continue
        if 1024 <= iv <= 10_000_000:
            candidates.append(iv)
    return min(candidates) if candidates else 0


def _local_hf_runtime(
    *,
    model_path: str,
    torch_dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    key = (
        str(model_path),
        str(torch_dtype),
        str(device_map),
        str(attn_implementation),
        bool(trust_remote_code),
    )
    cached = _LOCAL_HF_RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path_str = str(model_path).strip()
    if not model_path_str:
        raise RuntimeError("missing local model path")
    if not Path(model_path_str).exists():
        raise FileNotFoundError(f"local model path does not exist: {model_path_str}")

    dtype = _resolve_torch_dtype(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_str,
        trust_remote_code=bool(trust_remote_code),
        use_fast=True,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(trust_remote_code),
        "device_map": str(device_map).strip() or "auto",
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    attn_impl = str(attn_implementation or "").strip()
    if attn_impl and attn_impl.lower() != "none":
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_path_str, **model_kwargs)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    runtime = {
        "tokenizer": tokenizer,
        "model": model,
        "device": next(model.parameters()).device,
        "context_window": _safe_context_window(model, tokenizer),
    }
    _LOCAL_HF_RUNTIME_CACHE[key] = runtime
    return runtime


def _local_generate(
    *,
    model_path: str,
    model_alias: str,
    prompt: str,
    timeout_sec: float,
    max_tokens: int,
    torch_dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    _ = timeout_sec
    runtime = _local_hf_runtime(
        model_path=model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    import torch

    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]
    context_window = int(runtime.get("context_window", 0) or 0)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a rigorous math solution writer for SFT data. "
                "Solve carefully, then output a cleaned but sufficiently detailed reasoning record in strict JSON. "
                "The answer should teach the method, explain key transitions, and remain mathematically precise. "
                "Do not ramble, do not output markdown, and never output partial JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        text = (
            "System:\n"
            + str(messages[0]["content"])
            + "\n\nUser:\n"
            + str(messages[1]["content"])
            + "\n\nAssistant:\n"
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    prompt_tokens = int(input_ids.shape[-1])
    effective_max_tokens = max(1, int(max_tokens))
    if context_window > 0:
        effective_max_tokens = min(effective_max_tokens, max(1, context_window - prompt_tokens - 8))

    eos_token_id = tokenizer.eos_token_id
    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(effective_max_tokens),
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id,
    }
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)

    generated_ids = output_ids[0, input_ids.shape[-1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    completion_tokens = int(generated_ids.shape[-1])
    finish_reason = "stop"
    eos_ids: set[int] = set()
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids.update(int(x) for x in eos_token_id if isinstance(x, int))
    last_token = int(generated_ids[-1].item()) if completion_tokens > 0 else None
    if completion_tokens >= int(effective_max_tokens) and (last_token is None or last_token not in eos_ids):
        finish_reason = "length"

    if not raw_text:
        raise RuntimeError(f"local_model_empty_text: {model_alias}")
    try:
        parsed, parse_mode = _extract_json_object(raw_text)
    except Exception as e:
        raise AnswerGenerationError(
            str(e),
            debug={
                "finish_reason": finish_reason,
                "raw_text": raw_text,
                "raw_text_len": len(raw_text),
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        ) from e

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": prompt_tokens,
            "audio_tokens": 0,
            "image_tokens": 0,
        },
        "completion_tokens_details": {
            "text_tokens": completion_tokens,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
        },
    }
    return {
        "parsed": parsed,
        "raw_text": raw_text,
        "raw_text_len": len(raw_text),
        "finish_reason": finish_reason,
        "usage": usage,
        "parse_mode": parse_mode,
    }


def _gateway_generate(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_sec: float,
    max_tokens: int,
) -> dict[str, Any]:
    base = str(base_url).rstrip("/")
    url = base + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a rigorous math solution writer for SFT data. "
                    "Solve carefully, then output a cleaned but sufficiently detailed reasoning record in strict JSON. "
                    "The answer should teach the method, explain key transitions, and remain mathematically precise. "
                    "Do not ramble, do not output markdown, and never output partial JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"gateway_http_{e.code}: {body[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"gateway_request_error: {e}") from e

    choices = obj.get("choices", [])
    if not choices:
        raise RuntimeError(f"gateway_empty_choices: {obj}")
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = choice0.get("message", {}) if isinstance(choice0, dict) else {}
    text = _message_text(msg.get("content", ""))
    if not text:
        raise RuntimeError(f"gateway_empty_text: {obj}")
    finish_reason = str(choice0.get("finish_reason", "")).strip()
    usage = obj.get("usage", {})
    try:
        parsed, parse_mode = _extract_json_object(text)
    except Exception as e:
        raise AnswerGenerationError(
            str(e),
            debug={
                "finish_reason": finish_reason,
                "raw_text": text,
                "raw_text_len": len(text),
                "usage": usage,
            },
        ) from e
    return {
        "parsed": parsed,
        "raw_text": text,
        "raw_text_len": len(text),
        "finish_reason": finish_reason,
        "usage": usage,
        "parse_mode": parse_mode,
    }


def _row_difficulty(row: dict[str, Any]) -> str:
    return str(row.get("difficulty", "")).strip().lower()


def _row_model(
    row: dict[str, Any],
    *,
    easy_model: str,
    medium_model: str,
    hard_model: str,
) -> str:
    difficulty = _row_difficulty(row)
    if difficulty == "easy":
        return str(easy_model)
    if difficulty == "hard":
        return str(hard_model)
    return str(medium_model)


def _row_max_tokens(
    row: dict[str, Any],
    *,
    easy_max_tokens: int,
    medium_max_tokens: int,
    hard_max_tokens: int,
) -> int:
    difficulty = _row_difficulty(row)
    if difficulty == "easy":
        return int(easy_max_tokens)
    if difficulty == "hard":
        return int(hard_max_tokens)
    return int(medium_max_tokens)


def _fallback_nothinking_model(model: str) -> str:
    m = str(model or "").strip()
    return m


def _difficulty_answer_style(row: dict[str, Any]) -> tuple[str, int, int]:
    _ = row
    return ("standard", 5, 8)


def _build_answer_prompt(
    problem: str,
    row: dict[str, Any],
    *,
    compact: bool = False,
    target_min_tokens: int = 0,
    expansion_retry: bool = False,
) -> str:
    difficulty, min_steps, max_steps = _difficulty_answer_style(row)
    if compact:
        min_steps = 4
        max_steps = 6
    schema = json.dumps(ANSWER_SCHEMA, ensure_ascii=False)
    style_line = (
        "Use one coherent teaching-oriented solution: make the plan explicit, justify the main transformation, "
        "and keep enough intermediate detail that a learner could reproduce the argument."
    )
    min_visible_chars = max(1200, int(target_min_tokens) * 4)
    base_rules = f"""
- Return a cleaned final reasoning record, not raw scratch work.
- Solve internally first, then write a teachable derivation that makes the key bridges explicit.
- The solution is for SFT training: keep style stable, explicit, mathematically reproducible, and useful for learning.
- verdict must be "ok" or "wrong_problem".
- When verdict="ok", key_idea must be present and should be 1 to 3 concise sentences naming the core method and why it works here.
- solution_steps must contain between {min_steps} and {max_steps} steps when verdict="ok".
- Each solution step should usually contain 2 to 5 full sentences, and most steps should be multi-sentence unless the math is genuinely trivial.
- Each solution step must contain a concrete mathematical claim, equation, simplification, counting argument, or deduction that directly advances the solution.
- It is good to briefly explain why a chosen equation, substitution, factorization, invariant, or case split is the right move.
- It is allowed to include brief reflective guidance such as what to compute next, what constraint matters, or why an apparent ambiguity is resolved.
- Do not include full alternate solutions, fake uncertainty, long digressions, or repetitive paraphrase.
- Do not restate the entire problem, do not explain elementary definitions, and do not add unrelated background.
- Use one main method and include the key learning transitions: setup, reduction, decisive computation, and conclusion.
- check should usually be present and should be 1 to 2 short sentences verifying the final answer against the main condition.
- final_answer must contain only the final answer, with no explanation.
- {style_line}
- Do not reveal unrestricted chain-of-thought; provide only the cleaned reasoning that is pedagogically useful.
- If the problem is mathematically ill-posed, under-specified, self-contradictory, depends on an unresolved free parameter, or cannot determine a unique target, set verdict to "wrong_problem" and final_answer to exactly "wrong problem".
- If the problem is solvable under the natural mathematical reading and has a unique answer, do not output "wrong_problem" merely because the wording is terse or contains minor irrelevant noise.
- If the statement appears solvable at first but careful math shows that the answer is not unique, some required condition is missing, or no exact target value can be determined from the given information, output "wrong_problem".
- If you can only guess by numerical trial, obtain only an approximation when the question asks for an exact answer, or find that the claimed integer/rational/exact value does not actually exist from the given statement, output "wrong_problem".
- When verdict="wrong_problem", key_idea must be "", check must be "", and solution_steps must contain exactly one short reason.
- When verdict="ok", prefer enough visible detail that the written JSON answer is at least roughly {target_min_tokens} tokens of actual solution text if the math supports it.
- As a practical target, the visible written output should usually be at least about {min_visible_chars} characters, not counting hidden internal reasoning.
- If an algebraic or counting step has a non-obvious bridge, write that bridge explicitly instead of compressing it into a single clause.
- Do not satisfy the length target by repeating the same statement in different words.
- Never output partial JSON.
- Do not include markdown fences or extra prose.
- If there are multiple valid values, use compact standard mathematical notation in final_answer.
- Before output, do one brief consistency check between the final answer and the problem conditions.
- End immediately after the closing brace.
""".strip()
    if compact:
        base_rules += "\n- Compact retry mode: reduce wording, but keep the key teaching transitions and all essential derivation steps."
    if expansion_retry:
        base_rules += "\n- Expansion retry mode: the previous attempt was too terse. Keep the same structure but add the missing intermediate derivations and short teaching explanations."
    return f"""
Solve the following math problem.
Return strict JSON only.

Problem:
{problem}

Required JSON schema:
{schema}

Rules:
{base_rules}
""".strip()


def _extract_completion_tokens(usage: Any) -> int:
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            for key in ("text_tokens", "output_tokens"):
                value = details.get(key)
                if isinstance(value, (int, float)) and int(value) > 0:
                    return int(value)
                if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
                    return int(value.strip())
        for key in ("output_tokens", "completion_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
    return 0


def _answer_richness(resp_out: dict[str, Any], normalized_answer: dict[str, Any]) -> tuple[int, int, int]:
    completion_tokens = _extract_completion_tokens(resp_out.get("usage", {}))
    raw_len = int(resp_out.get("raw_text_len", 0) or 0)
    step_count = len(normalized_answer.get("solution_steps", []))
    return (raw_len, completion_tokens, step_count)


def _answer_too_short(
    resp_out: dict[str, Any],
    normalized_answer: dict[str, Any],
    *,
    min_completion_tokens: int,
) -> bool:
    if str(normalized_answer.get("verdict", "")) != "ok":
        return False
    completion_tokens = _extract_completion_tokens(resp_out.get("usage", {}))
    raw_len = int(resp_out.get("raw_text_len", 0) or 0)
    min_raw_len = max(1200, int(min_completion_tokens) * 4)
    if raw_len < min_raw_len:
        return True
    if completion_tokens > 0:
        return completion_tokens < max(1, int(min_completion_tokens))
    return False


def _normalize_answer_obj(obj: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    verdict = _normalize_short_text(obj.get("verdict", ""))
    final_answer = _normalize_short_text(obj.get("final_answer", ""))
    key_idea = _normalize_short_text(obj.get("key_idea", ""))
    solution_steps = _normalize_key_steps(obj.get("solution_steps", obj.get("key_steps", [])))
    check = _normalize_short_text(obj.get("check", ""))
    difficulty, min_steps, max_steps = _difficulty_answer_style(row)

    if final_answer == "wrong problem" and not verdict:
        verdict = "wrong_problem"
    if verdict not in {"ok", "wrong_problem"}:
        verdict = "wrong_problem" if final_answer == "wrong problem" else "ok"

    if verdict == "wrong_problem":
        reason = solution_steps[0] if solution_steps else "Problem statement is under-specified or inconsistent."
        return {
            "verdict": "wrong_problem",
            "key_idea": "",
            "solution_steps": [str(reason).strip()],
            "check": "",
            "final_answer": "wrong problem",
        }

    if not final_answer:
        raise RuntimeError("empty_final_answer")
    if final_answer == "wrong problem":
        return {
            "verdict": "wrong_problem",
            "key_idea": "",
            "solution_steps": [solution_steps[0] if solution_steps else "Problem statement is under-specified or inconsistent."],
            "check": "",
            "final_answer": "wrong problem",
        }
    if not solution_steps:
        raise RuntimeError("empty_solution_steps")
    if not key_idea:
        raise RuntimeError("empty_key_idea")
    if len(solution_steps) < min_steps:
        raise RuntimeError(f"too_few_solution_steps_for_{difficulty}")
    if len(solution_steps) > max_steps:
        solution_steps = solution_steps[:max_steps]
    return {
        "verdict": "ok",
        "key_idea": key_idea,
        "solution_steps": solution_steps,
        "check": check,
        "final_answer": final_answer,
    }


def _render_solution_text(ans: dict[str, Any]) -> str:
    if str(ans.get("verdict", "")) == "wrong_problem":
        steps = ans.get("solution_steps", [])
        reason = str(steps[0]).strip() if steps else "Problem statement is under-specified or inconsistent."
        return f"wrong problem\nReason: {reason}"
    lines: list[str] = []
    key_idea = _strip_field_label(str(ans.get("key_idea", "")), "Key idea")
    if key_idea:
        lines.append(f"Key idea: {key_idea}")
    lines.append("Solution:")
    for idx, step in enumerate(ans.get("solution_steps", []), 1):
        lines.append(f"{idx}. {str(step).strip()}")
    check = _strip_field_label(str(ans.get("check", "")), "Check")
    if check:
        lines.append(f"Check: {check}")
    lines.append(f"Final answer: {str(ans.get('final_answer', '')).strip()}")
    return "\n".join(lines)

def _answer_problem(
    *,
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
    model_path: str,
    problem: str,
    timeout_sec: float,
    max_tokens: int,
    retries: int,
    debug_store_raw: bool,
    debug_raw_char_cap: int,
    min_completion_tokens: int,
    row: dict[str, Any],
    torch_dtype: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    def _single_pass(*, prompt_mode: str, prompt: str, call_max_tokens: int, call_model: str) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(max(1, int(retries))):
            try:
                if backend == "local_hf":
                    resp = _local_generate(
                        model_path=model_path,
                        model_alias=call_model,
                        prompt=prompt,
                        timeout_sec=timeout_sec,
                        max_tokens=call_max_tokens,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        attn_implementation=attn_implementation,
                        trust_remote_code=trust_remote_code,
                    )
                else:
                    resp = _gateway_generate(
                        base_url=base_url,
                        api_key=api_key,
                        model=call_model,
                        prompt=prompt,
                        timeout_sec=timeout_sec,
                        max_tokens=call_max_tokens,
                    )
                obj = resp.get("parsed", {})
                if not isinstance(obj, dict):
                    raise RuntimeError("answer_response_not_dict")
                out = {
                    "obj": obj,
                    "finish_reason": str(resp.get("finish_reason", "")),
                    "raw_text_len": int(resp.get("raw_text_len", 0)),
                    "usage": resp.get("usage", {}),
                    "parse_mode": str(resp.get("parse_mode", "")),
                    "prompt_mode": prompt_mode,
                    "model": call_model,
                }
                if debug_store_raw:
                    raw_text = str(resp.get("raw_text", ""))
                    cap = max(0, int(debug_raw_char_cap))
                    out["raw_text"] = raw_text[:cap] if cap else raw_text
                    out["raw_text_truncated"] = bool(cap and len(raw_text) > cap)
                return out
            except Exception as e:
                last_err = e
                if attempt + 1 < max(1, int(retries)):
                    time.sleep(1.0)
        assert last_err is not None
        raise last_err

    def _materialize_answer(resp_out: dict[str, Any], *, normalized_answer: dict[str, Any]) -> dict[str, Any]:
        solution_steps = list(normalized_answer.get("solution_steps", []))
        solution = _render_solution_text(normalized_answer)
        out = {
            "verdict": normalized_answer.get("verdict", "ok"),
            "key_idea": normalized_answer.get("key_idea", ""),
            "final_answer": normalized_answer.get("final_answer", ""),
            "key_steps": solution_steps,
            "solution_steps": solution_steps,
            "check": normalized_answer.get("check", ""),
            "solution": solution,
            "finish_reason": resp_out.get("finish_reason", ""),
            "raw_text_len": resp_out.get("raw_text_len", 0),
            "usage": resp_out.get("usage", {}),
            "parse_mode": resp_out.get("parse_mode", ""),
            "prompt_mode": resp_out.get("prompt_mode", ""),
            "model": resp_out.get("model", model),
        }
        if debug_store_raw:
            out["raw_text"] = resp_out.get("raw_text", "")
            out["raw_text_truncated"] = bool(resp_out.get("raw_text_truncated", False))
        return out

    prompt = _build_answer_prompt(
        problem,
        row,
        compact=False,
        target_min_tokens=max(0, int(min_completion_tokens)),
    )
    compact_prompt = _build_answer_prompt(problem, row, compact=True)
    try:
        first_resp = _single_pass(
            prompt_mode="standard",
            prompt=prompt,
            call_max_tokens=max_tokens,
            call_model=model,
        )
    except Exception as e:
        if isinstance(e, AnswerGenerationError):
            finish_reason = str(e.debug.get("finish_reason", "")).strip().lower()
            raw_len = int(e.debug.get("raw_text_len", 0) or 0)
            fallback_model = _fallback_nothinking_model(model)
            if finish_reason == "length":
                retry_tokens = max(int(max_tokens), 900 if _row_difficulty(row) != "easy" else 450)
                if raw_len <= 120 and fallback_model and fallback_model != model:
                    first_resp = _single_pass(
                        prompt_mode="length_fallback_nothinking",
                        prompt=compact_prompt,
                        call_max_tokens=retry_tokens,
                        call_model=fallback_model,
                    )
                else:
                    first_resp = _single_pass(
                        prompt_mode="length_fallback_compact",
                        prompt=compact_prompt,
                        call_max_tokens=max(retry_tokens, int(max_tokens) + 250),
                        call_model=fallback_model or model,
                    )
            else:
                raise
        else:
            raise

    best_resp = first_resp
    best_obj = first_resp.get("obj", {})
    best_normalized = _normalize_answer_obj(best_obj, row)

    if _answer_too_short(
        best_resp,
        best_normalized,
        min_completion_tokens=int(min_completion_tokens),
    ):
        expansion_prompt = _build_answer_prompt(
            problem,
            row,
            compact=False,
            target_min_tokens=max(0, int(min_completion_tokens)),
            expansion_retry=True,
        )
        try:
            expansion_resp = _single_pass(
                prompt_mode="min_tokens_retry",
                prompt=expansion_prompt,
                call_max_tokens=max_tokens,
                call_model=model,
            )
            expansion_obj = expansion_resp.get("obj", {})
            expansion_normalized = _normalize_answer_obj(expansion_obj, row)
            if _answer_richness(expansion_resp, expansion_normalized) >= _answer_richness(best_resp, best_normalized):
                best_resp = expansion_resp
                best_normalized = expansion_normalized
        except Exception:
            pass

    final_answer = _materialize_answer(
        best_resp,
        normalized_answer=best_normalized,
    )
    final_answer["length_recovery_used"] = str(best_resp.get("prompt_mode", "")) != "standard"
    return final_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-prefix", default="data/outputs/answers/synth_answered")
    ap.add_argument("--backend", choices=["local_hf", "gateway"], default="local_hf")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--model-path", default=DEFAULT_LOCAL_MODEL_PATH)
    ap.add_argument("--easy-model", default="gemini-3-flash-preview")
    ap.add_argument("--medium-model", default="gemini-3-flash-preview")
    ap.add_argument("--hard-model", default="gemini-3-flash-preview")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from-prefix", action="append", default=[])
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--start-index", type=int, default=1)
    ap.add_argument("--easy-max-tokens", type=int, default=12172)
    ap.add_argument("--medium-max-tokens", type=int, default=12172)
    ap.add_argument("--hard-max-tokens", type=int, default=12172)
    ap.add_argument("--easy-min-completion-tokens", type=int, default=1024)
    ap.add_argument("--medium-min-completion-tokens", type=int, default=2048)
    ap.add_argument("--hard-min-completion-tokens", type=int, default=2048)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--debug-store-raw", action="store_true")
    ap.add_argument("--debug-raw-char-cap", type=int, default=4000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["_abs_project_root"])

    input_paths: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.is_dir():
            input_paths.extend(sorted(p.glob("*.jsonl")))
        else:
            input_paths.append(p)
    input_paths = [p for p in input_paths if p.exists()]
    if not input_paths:
        raise FileNotFoundError("no input jsonl files found")

    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = (root / out_prefix).resolve()
    status_path = out_prefix.parent / f"{out_prefix.stem}.status.json"
    resume_prefixes: list[Path] = []
    for raw in list(args.resume_from_prefix or []):
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()
        resume_prefixes.append(p)

    synth_cfg = cfg.get("synthesis", {})
    backend = str(args.backend).strip().lower()
    base_url = ""
    api_key = ""
    model_path = str(args.model_path or DEFAULT_LOCAL_MODEL_PATH).strip()
    if backend == "gateway":
        base_url = str(args.base_url or synth_cfg.get("base_url", "")).strip()
        if not base_url:
            raise RuntimeError("missing base_url: provide --base-url or synthesis.base_url in config")
        api_key = str(args.api_key or os.getenv(args.api_key_env, "")).strip()
        if not api_key:
            raise RuntimeError(f"missing API key: provide --api-key or env {args.api_key_env}")
    else:
        if not model_path:
            raise RuntimeError("missing local model path")
        model_path_resolved = Path(model_path)
        if not model_path_resolved.is_absolute():
            model_path_resolved = (root / model_path_resolved).resolve()
        model_path = str(model_path_resolved)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"local model path does not exist: {model_path}")

    override_model = str(args.model or "").strip()
    if backend == "local_hf":
        local_alias = override_model or Path(model_path).name
        easy_model = local_alias
        medium_model = local_alias
        hard_model = local_alias
    else:
        easy_model = override_model or str(args.easy_model).strip()
        medium_model = override_model or str(args.medium_model).strip()
        hard_model = override_model or str(args.hard_model).strip()

    rows = _iter_input_rows(input_paths)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    writer = JsonlShardWriter(out_prefix=out_prefix, shard_size=100, start_index=int(args.start_index))
    completed_keys: set[tuple[str, str]] = set()
    scan_prefixes: list[Path] = []
    if args.resume:
        scan_prefixes.append(out_prefix)
    scan_prefixes.extend(resume_prefixes)
    seen_resume_files: set[str] = set()
    for prefix in scan_prefixes:
        pat = f"{prefix.stem}_*.jsonl"
        for path in sorted(prefix.parent.glob(pat)):
            path_key = str(path.resolve())
            if path_key in seen_resume_files:
                continue
            seen_resume_files.add(path_key)
            for row in read_jsonl(str(path)):
                if not isinstance(row, dict):
                    continue
                key = (str(row.get("_source_file", "")), str(row.get("problem", "")))
                if key[0] and key[1]:
                    completed_keys.add(key)

    status: dict[str, Any] = {
        "state": "running",
        "backend": backend,
        "model": override_model,
        "easy_model": easy_model,
        "medium_model": medium_model,
        "hard_model": hard_model,
        "base_url": base_url,
        "model_path": model_path if backend == "local_hf" else "",
        "input_files": [str(p) for p in input_paths],
        "resume_from_prefixes": [str(p) for p in resume_prefixes],
        "requested_rows": len(rows),
        "completed_rows": 0,
        "failed_rows": 0,
        "current_shard": str(writer.current_path()),
        "current_shard_rows": writer.current_count,
        "touched_shards": writer.touched_paths(),
        "error": "",
        "recent_errors": [],
        "started_at": time.time(),
        "easy_max_tokens": int(args.easy_max_tokens),
        "medium_max_tokens": int(args.medium_max_tokens),
        "hard_max_tokens": int(args.hard_max_tokens),
        "easy_min_completion_tokens": int(args.easy_min_completion_tokens),
        "medium_min_completion_tokens": int(args.medium_min_completion_tokens),
        "hard_min_completion_tokens": int(args.hard_min_completion_tokens),
        "torch_dtype": str(args.torch_dtype),
        "device_map": str(args.device_map),
        "attn_implementation": str(args.attn_implementation),
        "trust_remote_code": bool(args.trust_remote_code),
        "retries": int(args.retries),
        "debug_store_raw": bool(args.debug_store_raw),
        "debug_raw_char_cap": int(args.debug_raw_char_cap),
    }
    _write_status(status_path, status)

    done = 0
    failed = 0
    for row in rows:
        source_file = str(row.get("_source_file", ""))
        problem = str(row.get("problem", "")).strip()
        if not problem:
            continue
        key = (source_file, problem)
        if key in completed_keys:
            done += 1
            status["completed_rows"] = done
            continue
        row_model = _row_model(
            row,
            easy_model=easy_model,
            medium_model=medium_model,
            hard_model=hard_model,
        )
        row_max_tokens = _row_max_tokens(
            row,
            easy_max_tokens=int(args.easy_max_tokens),
            medium_max_tokens=int(args.medium_max_tokens),
            hard_max_tokens=int(args.hard_max_tokens),
        )
        row_min_completion_tokens = _row_max_tokens(
            row,
            easy_max_tokens=int(args.easy_min_completion_tokens),
            medium_max_tokens=int(args.medium_min_completion_tokens),
            hard_max_tokens=int(args.hard_min_completion_tokens),
        )
        try:
            ans = _answer_problem(
                backend=backend,
                base_url=base_url,
                api_key=api_key,
                model=row_model,
                model_path=model_path,
                problem=problem,
                timeout_sec=float(args.timeout_sec),
                max_tokens=row_max_tokens,
                retries=int(args.retries),
                debug_store_raw=bool(args.debug_store_raw),
                debug_raw_char_cap=int(args.debug_raw_char_cap),
                min_completion_tokens=row_min_completion_tokens,
                row=row,
                torch_dtype=str(args.torch_dtype),
                device_map=str(args.device_map),
                attn_implementation=str(args.attn_implementation),
                trust_remote_code=bool(args.trust_remote_code),
            )
            out_row = dict(row)
            out_row["answer_model"] = row_model
            out_row["answer_verdict"] = ans.get("verdict", "ok")
            out_row["answer_key_idea"] = ans.get("key_idea", "")
            out_row["answer_check"] = ans.get("check", "")
            out_row["answer"] = ans["final_answer"]
            out_row["key_steps"] = ans["key_steps"]
            out_row["solution_steps"] = ans.get("solution_steps", ans["key_steps"])
            out_row["solution"] = ans["solution"]
            out_row["answer_max_tokens"] = row_max_tokens
            out_row["answer_min_completion_tokens"] = row_min_completion_tokens
            out_row["answer_finish_reason"] = ans.get("finish_reason", "")
            out_row["answer_raw_text_len"] = ans.get("raw_text_len", 0)
            out_row["answer_parse_mode"] = ans.get("parse_mode", "")
            out_row["answer_prompt_mode"] = ans.get("prompt_mode", "")
            out_row["answer_length_recovery_used"] = bool(ans.get("length_recovery_used", False))
            out_row["answer_usage"] = ans.get("usage", {})
            if bool(args.debug_store_raw):
                out_row["answer_raw_text"] = ans.get("raw_text", "")
                out_row["answer_raw_text_truncated"] = bool(ans.get("raw_text_truncated", False))
            writer.append_row(out_row)
            completed_keys.add(key)
            done += 1
            status["completed_rows"] = done
        except Exception as e:
            failed += 1
            status["failed_rows"] = failed
            err_item = {
                "source_file": source_file,
                "problem_preview": problem[:240],
                "model": row_model,
                "max_tokens": row_max_tokens,
                "min_completion_tokens": row_min_completion_tokens,
                "error": str(e),
            }
            if isinstance(e, AnswerGenerationError):
                err_item["error_debug"] = e.debug
            recent_errors = list(status.get("recent_errors", []))
            recent_errors.append(err_item)
            status["recent_errors"] = recent_errors[-20:]
        status["current_shard"] = str(writer.current_path())
        status["current_shard_rows"] = writer.current_count
        status["touched_shards"] = writer.touched_paths()
        _write_status(status_path, status)

    status["state"] = "ok_with_errors" if failed else "ok"
    status["finished_at"] = time.time()
    status["current_shard"] = str(writer.current_path())
    status["current_shard_rows"] = writer.current_count
    status["touched_shards"] = writer.touched_paths()
    _write_status(status_path, status)
    print(f"[OK] saved answer shards: {', '.join(writer.touched_paths()) if writer.touched_paths() else '(none)'}")


if __name__ == "__main__":
    main()
