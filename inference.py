#!/usr/bin/env python3
"""FlakeForge inference script with strict Analyzer -> Fixer execution flow."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from .agent.judge import FrozenJudge, JudgeLLMBackend
    from .agent.roles import AnalyzerRole, FixerRole, LoRAAdapterSpec, ModelBackend
    from .client import FlakeForgeEnv
    from .models import FlakeForgeAction, FlakeForgeObservation, Hypothesis
except ImportError:
    from agent.judge import FrozenJudge, JudgeLLMBackend
    from agent.roles import AnalyzerRole, FixerRole, LoRAAdapterSpec, ModelBackend
    from client import FlakeForgeEnv
    from models import FlakeForgeAction, FlakeForgeObservation, Hypothesis


if load_dotenv:
    load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.3-70b-instruct")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "minimaxai/minimax-m2.7")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
USE_DOCKER_IMAGE = os.getenv("USE_DOCKER_IMAGE", "0").strip().lower() in {"1", "true", "yes"}
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "flakeforge-env:latest")

MAX_STEPS = int(os.getenv("INFERENCE_MAX_STEPS", "14"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "900"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "45"))
ENV_MESSAGE_TIMEOUT_S = float(os.getenv("ENV_MESSAGE_TIMEOUT_S", "180"))

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"flakeforge_inference_{RUN_TS}.log"
SUMMARY_FILE = OUTPUT_DIR / f"flakeforge_summary_{RUN_TS}.json"


def _log(message: str) -> None:
    print(message, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


class OpenAIModelBackend(ModelBackend):
    """Text backend for Analyzer/Fixer calls through chat completions."""

    def __init__(self, model: str, base_url: str, api_key: Optional[str]) -> None:
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "")

    def generate(self, prompt: str, *, system_prompt: str, adapter_name: str) -> str:
        role_hint = (
            "You are in ANALYZER mode." if "analyzer" in adapter_name.lower() else "You are in FIXER mode."
        )
        messages = [
            {"role": "system", "content": f"{system_prompt}\n{role_hint}\nReturn only JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=0.95,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT_S,
            )
            return completion.choices[0].message.content or "{}"
        except Exception as exc:
            _log(f"[WARN] model backend failure adapter={adapter_name}: {exc}")
            return "{}"


class OpenAIJudgeBackend(JudgeLLMBackend):
    """Judge backend used by FrozenJudge for hypothesis/patch scoring."""

    def __init__(self, model: str, base_url: str, api_key: Optional[str]) -> None:
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "")

    def complete(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict senior engineer judge. Return only JSON "
                    "with keys: score (0-5) and reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=300,
                timeout=REQUEST_TIMEOUT_S,
            )
            return completion.choices[0].message.content or '{"score": 0, "reasoning": "empty"}'
        except Exception as exc:
            _log(f"[WARN] judge backend failure: {exc}")
            return '{"score": 0, "reasoning": "judge_call_failed"}'


def _extract_observation(reset_or_step_result: Any) -> FlakeForgeObservation:
    if isinstance(reset_or_step_result, FlakeForgeObservation):
        return reset_or_step_result
    return reset_or_step_result.observation


def _attach_hypothesis_to_action(action: FlakeForgeAction, hypothesis: Hypothesis) -> FlakeForgeAction:
    h_payload = {
        "root_cause_category": hypothesis.root_cause_category,
        "confidence": float(hypothesis.confidence),
        "evidence": list(hypothesis.evidence),
        "suggested_action": hypothesis.suggested_action,
    }
    return FlakeForgeAction(
        action_type=action.action_type,
        parameters=action.parameters,
        hypothesis=h_payload
    )


async def _build_env() -> FlakeForgeEnv:
    if USE_DOCKER_IMAGE:
        _log(f"[INIT] Using Docker image env: {LOCAL_IMAGE_NAME}")
        return await FlakeForgeEnv.from_docker_image(
            LOCAL_IMAGE_NAME,
            message_timeout_s=ENV_MESSAGE_TIMEOUT_S,
        )
    _log(f"[INIT] Using HTTP env: {ENV_BASE_URL}")
    return FlakeForgeEnv(base_url=ENV_BASE_URL, message_timeout_s=ENV_MESSAGE_TIMEOUT_S)


async def run_inference() -> Dict[str, Any]:
    if not NVIDIA_API_KEY:
        raise RuntimeError("Missing API key. Set NVIDIA_API_KEY or OPENAI_API_KEY.")

    model_backend = OpenAIModelBackend(MODEL_NAME, API_BASE_URL, NVIDIA_API_KEY)
    judge_backend = OpenAIJudgeBackend(JUDGE_MODEL, API_BASE_URL, NVIDIA_API_KEY)
    judge = FrozenJudge(backend=judge_backend)

    analyzer = AnalyzerRole(
        backend=model_backend,
        adapter=LoRAAdapterSpec(name="analyzer_lora", adapter_path="lora/analyzer"),
    )
    fixer = FixerRole(
        backend=model_backend,
        adapter=LoRAAdapterSpec(name="fixer_lora", adapter_path="lora/fixer"),
    )

    env = await _build_env()
    started_at = time.time()
    steps: List[Dict[str, Any]] = []

    try:
        reset_result = await env.reset()
        obs = _extract_observation(reset_result)

        _log(f"[START] episode={obs.episode_id} test={obs.test_identifier} max_steps={MAX_STEPS}")
        _log(f"[BASELINE] pass_rate={obs.baseline_pass_rate:.3f}")

        for step_idx in range(1, MAX_STEPS + 1):
            step_t0 = time.time()

            # Phase 1: Analysis (Analyzer role).
            hypothesis = analyzer.produce_hypothesis(obs)

            # Phase 2: Execution (Fixer role) using Analyzer context.
            proposed_action = fixer.produce_action(obs, hypothesis)
            action = _attach_hypothesis_to_action(proposed_action, hypothesis)

            try:
                step_result = await env.step(action)
            except Exception as exc:
                _log(f"[ERROR] env.step failed at step={step_idx}: {type(exc).__name__}: {exc}")
                break
            next_obs = _extract_observation(step_result)
            reward = float(getattr(step_result, "reward", next_obs.reward or 0.0))

            metadata = getattr(next_obs, "metadata", {}) or {}
            patch_diff = str(metadata.get("diff", ""))
            hypothesis_score = judge.score_hypothesis(obs, hypothesis)
            patch_score = judge.score_patch(obs, hypothesis, action, patch_diff)

            rec = {
                "step": step_idx,
                "hypothesis": {
                    "root_cause_category": hypothesis.root_cause_category,
                    "confidence": float(hypothesis.confidence),
                    "evidence": list(hypothesis.evidence),
                    "suggested_action": hypothesis.suggested_action,
                },
                "action": action.model_dump(),
                "reward": reward,
                "pass_rate": float(next_obs.current_pass_rate),
                "judge_hypothesis_score": int(hypothesis_score.get("score", 0)),
                "judge_patch_score": int(patch_score.get("score", 0)),
                "done": bool(next_obs.done),
                "duration_s": round(time.time() - step_t0, 3),
            }
            steps.append(rec)

            _log(
                "[STEP] "
                f"idx={step_idx} "
                f"analyze={hypothesis.root_cause_category}:{hypothesis.confidence:.2f} "
                f"execute={action.action_type} "
                f"reward={reward:.3f} "
                f"pass_rate={next_obs.current_pass_rate:.3f} "
                f"judge_h={rec['judge_hypothesis_score']} "
                f"judge_p={rec['judge_patch_score']} "
                f"done={str(bool(next_obs.done)).lower()}"
            )

            obs = next_obs
            if bool(next_obs.done):
                break

        elapsed_s = round(time.time() - started_at, 3)
        total_reward = round(sum(float(s["reward"]) for s in steps), 4)
        mean_h = round(sum(int(s["judge_hypothesis_score"]) for s in steps) / max(len(steps), 1), 3)
        mean_p = round(sum(int(s["judge_patch_score"]) for s in steps) / max(len(steps), 1), 3)
        final_obs = obs

        summary = {
            "episode_id": final_obs.episode_id,
            "test_identifier": final_obs.test_identifier,
            "model": MODEL_NAME,
            "judge_model": JUDGE_MODEL,
            "environment": "docker_image" if USE_DOCKER_IMAGE else ENV_BASE_URL,
            "max_steps": MAX_STEPS,
            "steps_executed": len(steps),
            "done": bool(final_obs.done),
            "baseline_pass_rate": float(final_obs.baseline_pass_rate),
            "final_pass_rate": float(final_obs.current_pass_rate),
            "improvement": round(float(final_obs.current_pass_rate - final_obs.baseline_pass_rate), 4),
            "total_reward": total_reward,
            "avg_judge_hypothesis_score": mean_h,
            "avg_judge_patch_score": mean_p,
            "elapsed_s": elapsed_s,
            "steps": steps,
        }
        SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _log(
            "[END] "
            f"steps={summary['steps_executed']} "
            f"baseline={summary['baseline_pass_rate']:.3f} "
            f"final={summary['final_pass_rate']:.3f} "
            f"improvement={summary['improvement']:+.3f} "
            f"total_reward={summary['total_reward']:.3f} "
            f"elapsed={summary['elapsed_s']:.2f}s"
        )
        _log(f"[OUTPUT] summary_file={SUMMARY_FILE}")
        _log(f"[OUTPUT] log_file={LOG_FILE}")

        return summary
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            maybe = close_fn()
            if asyncio.iscoroutine(maybe):
                await maybe


def main() -> None:
    summary = asyncio.run(run_inference())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
