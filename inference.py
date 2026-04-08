"""
Inference Script — Business Chat OpenEnv
=========================================
Runs an LLM agent against 3 tasks via the live HF Space HTTP API.

Environment variables required:
    API_BASE_URL   Azure/HF LLM endpoint
    MODEL_NAME     Model deployment name
    HF_TOKEN       API key
    ENV_URL        OpenEnv server URL (default: https://thevedp-business-chat-env.hf.space)
"""

import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error
import ssl

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("AZURE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "https://thevedp-business-chat-env.hf.space")
BENCHMARK = "business-chat-env"
MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 4096
SUCCESS_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# HTTP env client
# ---------------------------------------------------------------------------

def _http_post(url: str, data: dict, headers: dict, timeout: int = 30) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def env_reset(session_id: str, task_type: str, business_type: str) -> Dict[str, Any]:
    headers = {"X-Session-Id": session_id, "Content-Type": "application/json"}
    result = _http_post(f"{ENV_URL}/reset", {"task_type": task_type, "business_type": business_type}, headers, timeout=30)
    return result["observation"]


def env_step(session_id: str, action: dict) -> Dict[str, Any]:
    headers = {"X-Session-Id": session_id, "Content-Type": "application/json"}
    return _http_post(f"{ENV_URL}/step", {"action": action}, headers, timeout=60)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = textwrap.dedent("""
    You are an expert at classifying business chat messages.
    Given a list of messages and available labels, classify each message.
    Respond with valid JSON only in this exact format:
    {"classifications": {"msg_001": "label_name", "msg_002": "label_name", ...}}
    Use only the provided available_labels. Classify every message.
""").strip()

CLUSTER_SYSTEM = textwrap.dedent("""
    You are an expert at grouping business chat messages by topic.
    Given a list of messages (no labels provided), group them into 3-6 meaningful clusters.
    Respond with valid JSON only in this exact format:
    {
      "clusters": {"cluster_1": ["msg_001", "msg_002"], "cluster_2": ["msg_003"]},
      "cluster_labels": {"cluster_1": "descriptive topic label", "cluster_2": "another topic label"}
    }
    Every message must appear in exactly one cluster. Labels must be 2+ words and descriptive.
""").strip()

PROMPT_GEN_SYSTEM = textwrap.dedent("""
    You are an expert prompt engineer specializing in business communication.
    Given a business chat group context and sample messages, generate a highly tailored prompt template.
    Respond with valid JSON only in this exact format:
    {
      "prompt_template": "You are assisting [business]... [full detailed prompt, min 100 words]",
      "reasoning": "Explanation of design choices...",
      "identified_topics": ["topic1", "topic2", "topic3"]
    }
    The prompt_template must be specific to this business type, mention the business name, and cover all key topics found in the messages.
""").strip()


def build_classify_prompt(obs: Dict[str, Any]) -> str:
    ctx = obs["business_context"]
    msgs_text = "\n".join(f'{m["id"]}: [{m["sender"]}] {m["text"]}' for m in obs["messages"])
    return textwrap.dedent(f"""
        Business: {ctx["business_name"]} ({ctx["business_type"]})
        Group: {ctx["group_name"]}
        Available labels: {obs["available_labels"]}

        Messages to classify:
        {msgs_text}

        Classify every message. Return JSON only.
    """).strip()


def build_cluster_prompt(obs: Dict[str, Any]) -> str:
    ctx = obs["business_context"]
    msgs_text = "\n".join(f'{m["id"]}: [{m["sender"]}] {m["text"]}' for m in obs["messages"])
    return textwrap.dedent(f"""
        Business: {ctx["business_name"]} ({ctx["business_type"]})
        Group: {ctx["group_name"]}

        Messages to cluster (no labels given — discover topics yourself):
        {msgs_text}

        Group into 3-6 meaningful clusters with descriptive 2+ word labels. Return JSON only.
    """).strip()


def build_prompt_gen_prompt(obs: Dict[str, Any]) -> str:
    ctx = obs["business_context"]
    msgs_text = "\n".join(f'[{m["sender"]}]: {m["text"]}' for m in obs["messages"][:20])
    return textwrap.dedent(f"""
        Business: {ctx["business_name"]} ({ctx["business_type"]})
        Group: {ctx["group_name"]}
        Description: {ctx["description"]}

        Sample chat messages:
        {msgs_text}

        Generate a highly tailored prompt template for this specific business group. Return JSON only.
    """).strip()

# ---------------------------------------------------------------------------
# Agent — calls LLM and returns action dict
# ---------------------------------------------------------------------------

AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
IS_AZURE = "azure.com" in API_BASE_URL


def llm_chat(system: str, user: str) -> str:
    """Call OpenAI-compatible chat completions endpoint via raw HTTP."""
    if IS_AZURE:
        url = (
            f"{API_BASE_URL.rstrip('/')}/openai/deployments/{MODEL_NAME}"
            f"/chat/completions?api-version={AZURE_API_VERSION}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY,
        }
    else:
        url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_TOKENS,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"[VERBOSE] LLM API error {e.code}: {err_body[:500]}", flush=True)
        raise
    return (data["choices"][0]["message"]["content"] or "").strip()


def get_action(task_type: str, obs: Dict[str, Any]) -> tuple:
    if task_type == "classify":
        system, user = CLASSIFY_SYSTEM, build_classify_prompt(obs)
    elif task_type == "cluster":
        system, user = CLUSTER_SYSTEM, build_cluster_prompt(obs)
    else:
        system, user = PROMPT_GEN_SYSTEM, build_prompt_gen_prompt(obs)

    error = None
    action_str = ""
    try:
        raw = llm_chat(system, user)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        action_str = raw[:80].replace("\n", " ")

        if task_type == "classify":
            action = {"task_type": "classify", "classify_action": {"classifications": parsed["classifications"]}}
        elif task_type == "cluster":
            action = {"task_type": "cluster", "cluster_action": {"clusters": parsed["clusters"], "cluster_labels": parsed["cluster_labels"]}}
        else:
            action = {"task_type": "prompt_gen", "prompt_gen_action": {"prompt_template": parsed["prompt_template"], "reasoning": parsed["reasoning"], "identified_topics": parsed["identified_topics"]}}

    except Exception as exc:
        error = str(exc)[:100]
        action_str = "parse_error"
        msgs = obs.get("messages", [])
        if task_type == "classify":
            labels = obs.get("available_labels", ["general"])
            action = {"task_type": "classify", "classify_action": {"classifications": {m["id"]: labels[0] for m in msgs}}}
        elif task_type == "cluster":
            action = {"task_type": "cluster", "cluster_action": {"clusters": {"cluster_1": [m["id"] for m in msgs]}, "cluster_labels": {"cluster_1": "general discussion"}}}
        else:
            action = {"task_type": "prompt_gen", "prompt_gen_action": {"prompt_template": "You are a helpful assistant for this business group. Respond professionally to all queries.", "reasoning": "Fallback.", "identified_topics": ["general"]}}

    return action, action_str, error

# ---------------------------------------------------------------------------
# Run a single episode
# ---------------------------------------------------------------------------

def run_episode(task_type: str, business_type: str) -> float:
    task_name = f"{task_type}-{business_type}"
    session_id = f"{task_type}-{business_type}-{int(time.time())}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = env_reset(session_id, task_type, business_type)
        ctx = obs["business_context"]
        print(f"\n{'='*60}", flush=True)
        print(f"[VERBOSE] Business : {ctx['business_name']} ({ctx['business_type']})", flush=True)
        print(f"[VERBOSE] Task     : {task_type}", flush=True)
        print(f"[VERBOSE] Messages : {len(obs['messages'])} total", flush=True)
        print(f"{'='*60}\n", flush=True)

        for step in range(1, MAX_STEPS + 1):
            action, action_str, error = get_action(task_type, obs)
            result = env_step(session_id, action)

            reward = result.get("reward", {}).get("score", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", obs)

            rewards.append(reward)
            steps_taken = step

            print(f"[VERBOSE] Reward: {result.get('reward', {})}", flush=True)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = round(min(max(max(rewards) if rewards else 0.0, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[VERBOSE] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tasks = [
        ("classify", "electronics_retail"),
        ("cluster", "restaurant_chain"),
        ("prompt_gen", "real_estate"),
    ]

    all_scores = []
    for task_type, business_type in tasks:
        score = run_episode(task_type, business_type)
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    scores_str = ",".join(f"{s:.3f}" for s in all_scores)
    print(f"\n[SUMMARY] tasks={len(tasks)} avg_score={avg:.3f} scores={scores_str}", flush=True)


if __name__ == "__main__":
    main()
