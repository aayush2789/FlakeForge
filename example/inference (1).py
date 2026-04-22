#!/usr/bin/env python3
"""PII Redaction inference script with logging and timing."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .client import RedactionEnv
    from .models import ActionType, RedactionAction
except ImportError:
    from client import RedactionEnv
    from models import ActionType, RedactionAction

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
CONTAINER_BASE_URL = os.getenv("CONTAINER_BASE_URL", "http://localhost:7860")
USE_DOCKER_IMAGE = os.getenv("USE_DOCKER_IMAGE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
BENCHMARK = os.getenv("BENCHMARK", "pii-redaction-env")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
OPENAI_SEED = int(os.getenv("OPENAI_SEED", "42"))
INFERENCE_MAX_STEPS = int(os.getenv("INFERENCE_MAX_STEPS", "100"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "30"))
# Task-specific success thresholds
TASK_THRESHOLDS = {
    "gdpr_contract_easy": 0.90,
    "hipaa_medical_medium": 0.85,
    "security_logs_hard": 0.75,
}
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
RETRY_ON_TRANSIENT_ERRORS = os.getenv(
    "RETRY_ON_TRANSIENT_ERRORS", "0"
).strip().lower() in {"1", "true", "yes"}

PII_TASK_ID_OVERRIDE = os.getenv("PII_TASK_ID")
if PII_TASK_ID_OVERRIDE:
    TASKS = [PII_TASK_ID_OVERRIDE]
else:
    TASKS = ["gdpr_contract_easy", "hipaa_medical_medium", "security_logs_hard"]

# Setup outputs directory and logging
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"inference_{TIMESTAMP}.log"
SUMMARY_FILE = OUTPUT_DIR / f"summary_{TIMESTAMP}.json"

# Global state for tracking
_task_results: List[Dict[str, Any]] = []
_run_start_time = time.time()

# PII patterns
LABEL_PATTERNS: Dict[str, List[str]] = {
    "EMAIL": [r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"],
    "PHONE": [
        r"\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
        r"\b\d{3}[-.]\d{4}\b",
    ],
    "SSN": [r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"],
    "DOB": [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
    ],
    "NAME": [],
    "ADDRESS": [
        r"\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+){1,4}"
        r"(?:\s+(?:St|Ave|Blvd|Rd|Dr|Ln|Way|Ct|Terrace|Court|Place|"
        r"Drive|Road|Street|Lane|Boulevard|Avenue)\.?)?"
        r"(?:,\s*(?:Apt|Suite|Unit|Floor|Fl)\.?\s*[\w-]+)?",
    ],
}
PII_REGEX_PATTERNS = [p for patterns in LABEL_PATTERNS.values() for p in patterns]


# Logging utilities
def _write_log(message: str) -> None:
    """Write message to both console and log file."""
    print(message, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_start(task: str, env: str, model: str) -> None:
    msg = f"[START] task={task} env={env} model={model}"
    _write_log(msg)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    msg = (
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}"
    )
    _write_log(msg)


def log_end(
    success: bool, steps: int, score: float, rewards: List[float], task_time_s: float = 0.0
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    msg = (
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str} time={task_time_s:.2f}s"
    )
    _write_log(msg)


# Reward helpers
def _clamp_reward(raw: float) -> float:
    return round(max(0.0, min(1.0, (raw + 1.0) / 2.0)), 4)


def _extract_reward(reward: Any) -> float:
    if reward is None:
        return 0.0
    if hasattr(reward, "raw_total"):
        return _clamp_reward(float(reward.raw_total))
    if isinstance(reward, dict):
        return _clamp_reward(float(reward.get("raw_total", reward.get("total", 0.0))))
    if isinstance(reward, (int, float)):
        return _clamp_reward(float(reward))
    return 0.0


# LLM helpers
def _sanitize_api_key(value: Optional[str]) -> str:
    token = (value or "").strip()
    return token[7:].strip() if token.lower().startswith("bearer ") else token


def _chat_completion_kwargs(prompt: str, use_json_mode: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "seed": OPENAI_SEED,
        "timeout": REQUEST_TIMEOUT_S,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


# Prompt
def _observation_metadata(obs) -> Dict[str, Any]:
    meta = getattr(obs, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def _remaining_entities_hint(obs) -> str:
    meta = _observation_metadata(obs)
    for key in ("remaining_entities", "reward_remaining_entities"):
        value = meta.get(key)
        if isinstance(value, (int, float)):
            return str(max(0, int(value)))
    return "unknown"


def _last_reward_signal_hint(obs) -> str:
    meta = _observation_metadata(obs)
    components = meta.get("reward_components")
    if not isinstance(components, dict):
        return "unavailable"
    miss_penalty = float(components.get("miss_penalty", 0.0) or 0.0)
    skip_miss_penalty = float(components.get("skip_miss_penalty", 0.0) or 0.0)
    skip_stagnation_penalty = float(
        components.get("skip_stagnation_penalty", 0.0) or 0.0
    )
    return (
        f"miss_penalty={miss_penalty:.2f}, "
        f"skip_miss_penalty={skip_miss_penalty:.2f}, "
        f"skip_stagnation_penalty={skip_stagnation_penalty:.2f}"
    )


def _build_prompt(obs) -> str:
    return f"""
You are a GDPR/HIPAA compliance analyst performing PII redaction on a document.

Task: identify and redact ALL personally identifiable information using precise
absolute character indices. PII types: EMAIL, PHONE, SSN, NAME, ADDRESS, DOB.

Label guidance:
- a word or full name of a person -> NAME
- a date in any format -> DOB
- a street address -> ADDRESS
- an email address -> EMAIL
- a phone number -> PHONE
- a social security number -> SSN
- NAME detection: scan for full names near contextual anchors such as member,
  patient, customer, contractor, tenant, applicant, owner, user, caller, she,
  he, her, and his. Redact the NAME before associated contact details.

IMPORTANT - Obfuscated PII patterns to watch for:
- Emails: "john dot smith at gmail dot com", "j_doe at mail dot net"
- Phones: "five five five-0199", "area code four-one-five then 555-0187"
- SSNs: "three four five dash six seven dash eight nine zero one"
- DOBs: "nineteen eighty-five march twelve", "oh-three slash fifteen slash seventy-eight"

IMPORTANT - Span boundaries for obfuscated PII:
When PII is written out in words, the span MUST start at the FIRST word of the
obfuscated phrase and end at the LAST character, including all written-out connectors.

WRONG: "area code four-one-five, then 555-0187" -> start at "555"
RIGHT: "area code four-one-five, then 555-0187" -> start at "area"

DO NOT REDACT: Company names (Apple, Google, Amazon, Microsoft, Samsung, Meta),
product codes, project codenames, infrastructure terms, or organizational references.

Current window (chars {obs.cursor_position}-{obs.cursor_position + len(obs.visible_text)}):
---
{obs.visible_text}
---
Progress: {obs.progress_pct:.0%} | Already redacted: {obs.redacted_spans}
Previous actions: {obs.previous_actions}
Remaining entities (approx): {_remaining_entities_hint(obs)}
Last-step reward signals: {_last_reward_signal_hint(obs)}

Worked example: cursor_position=50, visible_text="Patient John Smith lived"
  "John" is at relative offset 8 -> absolute_start=58, absolute_end=62

ADDRESS example: visible_text="resides at 90 Birchwood Lane."
  Start at "90", end after "Lane", stop before punctuation.

Return strict JSON only, one of:
{{"action": "REDACT", "start": <abs_start>, "end": <abs_end>, "label": "<PII_TYPE>"}}
{{"action": "NEXT_CHUNK"}}
{{"action": "PREV_CHUNK"}}
{{"action": "SKIP"}}
{{"action": "FINISH"}}

Rules:
- REDACT: one PII span, absolute offsets, always include label.
- NEXT_CHUNK: after all PII in current window is handled.
- PREV_CHUNK: use as a recovery move if likely PII was missed in the prior window.
- SKIP: no PII in current window.
- FINISH: only when progress_pct >= 1.0 AND all PII types have been checked.
- Do NOT re-redact already_redacted spans.
- Prefer tight, precise spans over broad ones.

Navigation Strategy (VERY IMPORTANT):
- If you moved to NEXT_CHUNK but later realize PII was likely missed in the previous window, use PREV_CHUNK to go back and correct it.
- If previous_actions show repeated NEXT_CHUNK or SKIP without finding entities, reconsider earlier windows using PREV_CHUNK.
- If you see partial context of a PII span (for example only a first name or a truncated email), consider PREV_CHUNK to recover the full span.
- Avoid repeatedly moving forward if entities are being missed.
- Use PREV_CHUNK as a recovery mechanism after low-reward navigation steps.

Bad behavior to avoid:
- Repeated NEXT_CHUNK without fully processing the current window.
- Never going back to correct missed entities.

Critical navigation rule:
- SKIP does not move the cursor; if the window is clean, default to NEXT_CHUNK.
- Use SKIP only to re-check the same window when uncertain.
- Two consecutive SKIPs are a loop signal: move forward with NEXT_CHUNK.
- Use PREV_CHUNK when you likely missed PII in the prior window.
""".strip()


# Action parsing
def _action_to_string(action: RedactionAction) -> str:
    if action.action_type == ActionType.REDACT:
        return f"REDACT({action.start},{action.end})"
    return action.action_type.value


def _label_from_text(text: str) -> str:
    c = (text or "").strip()
    if re.fullmatch(LABEL_PATTERNS["EMAIL"][0], c, flags=re.IGNORECASE):
        return "EMAIL"
    if re.fullmatch(LABEL_PATTERNS["SSN"][0], c, flags=re.IGNORECASE):
        return "SSN"
    if re.fullmatch(LABEL_PATTERNS["PHONE"][0], c, flags=re.IGNORECASE):
        return "PHONE"
    if re.fullmatch(LABEL_PATTERNS["DOB"][0], c, flags=re.IGNORECASE):
        return "DOB"
    return "NAME"


def _snap_redact_span(
    obs, raw_start: int, raw_end: int, label: Optional[str] = None
) -> Tuple[int, int]:
    text = obs.visible_text or ""
    if not text:
        return raw_start, raw_end
    patterns = LABEL_PATTERNS.get(label, []) if label else []
    if not patterns:
        patterns = PII_REGEX_PATTERNS
    window_start = int(obs.cursor_position)
    window_end = window_start + len(text)
    rel_start = raw_start - window_start
    rel_end = raw_end - window_start
    scan_left = max(0, min(rel_start, rel_end) - 20)
    scan_right = min(len(text), max(rel_start, rel_end) + 20)
    if scan_right <= scan_left:
        return raw_start, raw_end
    scan_text = text[scan_left:scan_right]
    sc, ec = [], []
    for p in patterns:
        for m in re.finditer(p, scan_text, flags=re.IGNORECASE):
            sc.append(window_start + scan_left + m.start())
            ec.append(window_start + scan_left + m.end())
    ss = min(sc, key=lambda c: abs(c - raw_start)) if sc else raw_start
    se = min(ec, key=lambda c: abs(c - raw_end)) if ec else raw_end
    ss = max(0, min(ss, window_end - 1))
    se = max(ss + 1, min(se, window_end))
    return ss, se


def _extract_json_object(text: str) -> Optional[str]:
    c = (text or "").strip()
    if not c:
        return None
    if c.startswith("```"):
        c = re.sub(r"^```(?:json)?\s*", "", c, flags=re.IGNORECASE)
        c = re.sub(r"\s*```$", "", c, flags=re.IGNORECASE)
        c = c.strip()
    if c.startswith("{") and c.endswith("}"):
        return c
    start = c.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(c)):
        if c[i] == "{":
            depth += 1
        elif c[i] == "}":
            depth -= 1
            if depth == 0:
                return c[start : i + 1]
    return None


def _parse_action_payload(
    raw_text: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    json_text = _extract_json_object(raw_text)
    if not json_text:
        return None, "no_json_object"
    try:
        payload = json.loads(json_text)
        return (
            (payload, None) if isinstance(payload, dict) else (None, "json_not_object")
        )
    except Exception as exc:
        return None, str(exc)


def _coerce_action(
    payload: Dict[str, Any], obs
) -> Tuple[RedactionAction, Optional[str]]:
    if "action_type" not in payload and "action" in payload:
        payload = {**payload, "action_type": payload["action"]}
    if "action_type" not in payload:
        return RedactionAction(action_type=ActionType.NEXT_CHUNK), "missing_action_type"
    if str(payload.get("action_type", "")).upper() == ActionType.REDACT.value:
        try:
            raw_start = int(payload.get("start"))
            raw_end = int(payload.get("end"))
            label_value = payload.get("label")
            ss, se = _snap_redact_span(
                obs, raw_start, raw_end, label=str(label_value) if label_value else None
            )
            if not label_value:
                rs = max(0, ss - int(obs.cursor_position))
                re_ = max(rs + 1, se - int(obs.cursor_position))
                label_value = _label_from_text((obs.visible_text or "")[rs:re_])
            payload = {**payload, "label": label_value, "start": ss, "end": se}
        except Exception:
            pass
    try:
        return RedactionAction(**payload), None
    except Exception as exc:
        return RedactionAction(action_type=ActionType.NEXT_CHUNK), str(exc)


def _apply_navigation_recovery(
    action: RedactionAction, obs
) -> Tuple[RedactionAction, Optional[str]]:
    if action.action_type not in {ActionType.NEXT_CHUNK, ActionType.SKIP}:
        return action, None

    previous_actions = list(obs.previous_actions or [])
    trailing_nav = previous_actions[-3:]
    nav_stagnation = len(trailing_nav) == 3 and all(
        a in {ActionType.NEXT_CHUNK.value, ActionType.SKIP.value} for a in trailing_nav
    )

    trailing_skips = 0
    for prev_action in reversed(previous_actions):
        if prev_action == ActionType.SKIP.value:
            trailing_skips += 1
        else:
            break

    meta = _observation_metadata(obs)
    components = meta.get("reward_components") if isinstance(meta, dict) else None
    miss_penalty = 0.0
    skip_miss_penalty = 0.0
    skip_stagnation_penalty = 0.0
    if isinstance(components, dict):
        miss_penalty = float(components.get("miss_penalty", 0.0) or 0.0)
        skip_miss_penalty = float(components.get("skip_miss_penalty", 0.0) or 0.0)
        skip_stagnation_penalty = float(
            components.get("skip_stagnation_penalty", 0.0) or 0.0
        )

    should_recover = (
        miss_penalty < 0.0
        or skip_miss_penalty < 0.0
        or skip_stagnation_penalty < 0.0
        or nav_stagnation
    )

    if action.action_type == ActionType.SKIP and trailing_skips >= 2:
        return RedactionAction(action_type=ActionType.NEXT_CHUNK), "force_progress"

    if should_recover and int(getattr(obs, "cursor_position", 0)) > 0:
        return RedactionAction(action_type=ActionType.PREV_CHUNK), "nav_recovery_prev"
    return action, None


async def _next_action(client: OpenAI, obs) -> Tuple[RedactionAction, Optional[str]]:
    # Auto-finish if cursor is stuck at end
    trailing = 0
    for a in reversed(obs.previous_actions or []):
        if a == "NEXT_CHUNK":
            trailing += 1
        else:
            break
    if trailing >= 3 and obs.cursor_position >= obs.document_length - len(
        obs.visible_text
    ):
        return RedactionAction(action_type=ActionType.FINISH), "cursor_clamped"

    prompt = _build_prompt(obs)

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            **_chat_completion_kwargs(prompt, use_json_mode=True),
        )
        text = response.choices[0].message.content or ""
        payload, parse_error = _parse_action_payload(text)
        if payload is None:
            return RedactionAction(
                action_type=ActionType.NEXT_CHUNK
            ), f"parse_error:{parse_error}"
        action, action_error = _coerce_action(payload, obs)
        action, nav_error = _apply_navigation_recovery(action, obs)
        if action_error and nav_error:
            return action, f"{action_error};{nav_error}"
        return action, action_error or nav_error
    except Exception as exc:
        msg = str(exc)
        msg_low = msg.lower()
        retry = RETRY_ON_TRANSIENT_ERRORS and (
            "500" in msg
            or "internal server error" in msg_low
            or "timeout" in msg_low
            or "timed out" in msg_low
        )
        if retry:
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    **_chat_completion_kwargs(prompt, use_json_mode=False),
                )
                text = response.choices[0].message.content or ""
                payload, parse_error = _parse_action_payload(text)
                if payload is None:
                    return RedactionAction(
                        action_type=ActionType.NEXT_CHUNK
                    ), f"retry_parse_error:{parse_error}"
                action, action_error = _coerce_action(payload, obs)
                action, nav_error = _apply_navigation_recovery(action, obs)
                if action_error and nav_error:
                    return action, f"{action_error};{nav_error}"
                return action, action_error or nav_error
            except Exception as retry_exc:
                raise RuntimeError(str(retry_exc)) from retry_exc
        raise


# Task runner with timing and result collection
async def run_task(client: OpenAI, task_id: str, env: RedactionEnv) -> None:
    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False
    task_start_time = time.time()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = (await env.reset(task_id=task_id)).observation

        for step in range(1, INFERENCE_MAX_STEPS + 1):
            if obs.done:
                break

            action_error: Optional[str] = None
            try:
                action, action_error = await _next_action(client, obs)
            except Exception as exc:
                log_step(
                    step=step, action="ERROR", reward=0.0, done=True, error=str(exc)
                )
                steps = step
                break

            result = await env.step(action)
            obs = result.observation
            done = bool(result.done)

            clamped = _extract_reward(result.reward)
            if result.reward is None:
                clamped = _extract_reward(getattr(obs, "reward", None))
                if clamped == 0.0:
                    obs_meta = getattr(obs, "metadata", {}) or {}
                    clamped = _extract_reward(obs_meta.get("reward_raw_total"))
            rewards.append(clamped)
            steps = step

            log_step(
                step=step,
                action=_action_to_string(action),
                reward=clamped,
                done=done,
                error=action_error,
            )

            if done:
                break

        score = max(0.0, min(1.0, (sum(rewards) / len(rewards)) if rewards else 0.0))
        success = bool(score >= TASK_THRESHOLDS.get(task_id, 0.5))

    except Exception as exc:
        print(f"[DEBUG] run_task error: {exc}", flush=True)
        success = False

    finally:
        task_elapsed = time.time() - task_start_time
        log_end(success=success, steps=steps, score=score, rewards=rewards, task_time_s=task_elapsed)
        
        # Collect result for summary
        _task_results.append({
            "task_id": task_id,
            "success": success,
            "steps": steps,
            "score": round(score, 3),
            "rewards": [round(r, 2) for r in rewards],
            "time_seconds": round(task_elapsed, 2),
        })


def _save_summary() -> None:
    """Save inference summary to JSON file."""
    total_time = time.time() - _run_start_time
    summary = {
        "timestamp": TIMESTAMP,
        "model": MODEL_NAME,
        "benchmark": BENCHMARK,
        "total_tasks": len(_task_results),
        "total_time_seconds": round(total_time, 2),
        "tasks": _task_results,
        "total_passed": sum(1 for r in _task_results if r["success"]),
        "average_score": round(
            sum(r["score"] for r in _task_results) / len(_task_results), 3
        )
        if _task_results
        else 0.0,
    }
    
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    _write_log("\n" + "="*60)
    _write_log(f"Summary saved to: {SUMMARY_FILE}")
    _write_log(f"Log file saved to: {LOG_FILE}")
    _write_log("="*60 + "\n")


# Entry point
async def main() -> None:
    resolved_token = _sanitize_api_key(HF_TOKEN)
    if not resolved_token:
        raise RuntimeError("HF_TOKEN (or OPENAI_API_KEY) is required.")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=resolved_token,
        max_retries=OPENAI_MAX_RETRIES,
    )
    # Default mode: connect to already-running env endpoint (avoids spawning duplicate containers).
    # Opt-in image mode: set USE_DOCKER_IMAGE=1 and LOCAL_IMAGE_NAME=<image>.
    used_docker_image_mode = False
    if USE_DOCKER_IMAGE:
        if not IMAGE_NAME:
            raise RuntimeError(
                "USE_DOCKER_IMAGE=1 requires LOCAL_IMAGE_NAME to be set."
            )
        try:
            env = await RedactionEnv.from_docker_image(IMAGE_NAME)
            used_docker_image_mode = True
            async with env:
                for task_id in TASKS:
                    await run_task(client, task_id, env)
        except Exception as exc:
           for task_id in TASKS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_step(
                step=1,
                action="ERROR",
                reward=0.0,
                done=True,
                error=f"docker_error:{str(exc)}"
            )
            log_end(
                success=False,
                steps=1,
                score=0.0,
                rewards=[0.0]
            )
            return

    if not used_docker_image_mode:
        # Connect to already-running env endpoint.
        for task_id in TASKS:
            try:
                async with RedactionEnv(base_url=CONTAINER_BASE_URL) as env:
                    await run_task(client, task_id, env)
            except Exception as exc:
                print(f"[DEBUG] env error on task {task_id}: {exc}", flush=True)
    
    # Save summary after all tasks complete
    _save_summary()


if __name__ == "__main__":
    asyncio.run(main())
