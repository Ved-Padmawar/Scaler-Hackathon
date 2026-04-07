"""
test_agent.py — End-to-end agent test via HTTP endpoints
=========================================================
Runs an LLM agent against the live FastAPI server endpoints.
Tests all 9 task-business combinations.

Usage:
    poetry run python test_agent.py

Requires server to be running:
    poetry run python run.py
"""

import orjson
import os
import textwrap
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("AZURE_API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 2048
SESSION_ID = "test-agent-session"

HEADERS = {
    "Content-Type": "application/json",
    "X-Session-Id": SESSION_ID,
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def api_reset(task_type: Optional[str] = None, business_type: Optional[str] = None) -> Dict:
    payload = {}
    if task_type:
        payload["task_type"] = task_type
    if business_type:
        payload["business_type"] = business_type
    r = requests.post(f"{SERVER_URL}/reset", json=payload, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def api_step(action: Dict) -> Dict:
    r = requests.post(f"{SERVER_URL}/step", json={"action": action}, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def api_state() -> Dict:
    r = requests.get(f"{SERVER_URL}/state", headers=HEADERS)
    r.raise_for_status()
    return r.json()


def api_health() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = textwrap.dedent("""
    You are an expert at classifying business chat messages.
    Given a list of messages and available labels, classify each message.
    Respond with valid JSON only:
    {"classifications": {"msg_001": "label", "msg_002": "label", ...}}
    Use only the provided labels. Classify every message.
""").strip()

CLUSTER_SYSTEM = textwrap.dedent("""
    You are an expert at grouping business chat messages by topic.
    Group messages into 3-6 meaningful clusters.
    Respond with valid JSON only:
    {
      "clusters": {"cluster_1": ["msg_001", "msg_002"], "cluster_2": ["msg_003"]},
      "cluster_labels": {"cluster_1": "product inquiries", "cluster_2": "pricing questions"}
    }
    Every message must appear in exactly one cluster. Labels must be descriptive (2+ words).
""").strip()

PROMPT_GEN_SYSTEM = textwrap.dedent("""
    You are an expert prompt engineer for business communication.
    Given a business chat group's context and sample messages, generate a tailored prompt template.
    Respond with valid JSON only:
    {
      "prompt_template": "You are assisting [business]...",
      "reasoning": "Why this prompt works for this business...",
      "identified_topics": ["topic1", "topic2", "topic3"]
    }
    The prompt_template must be 100+ words, mention the business name/type, and cover all key topics.
""").strip()


def build_user_prompt(task_type: str, observation: Dict) -> str:
    ctx = observation["business_context"]
    messages = observation["messages"]

    if task_type == "classify":
        msgs_text = "\n".join(f'{m["id"]}: [{m["sender"]}] {m["text"]}' for m in messages)
        return textwrap.dedent(f"""
            Business: {ctx["business_name"]} ({ctx["business_type"]})
            Group: {ctx["group_name"]}
            Labels: {observation["available_labels"]}
            Messages:
            {msgs_text}
            Classify every message. JSON only.
        """).strip()

    elif task_type == "cluster":
        msgs_text = "\n".join(f'{m["id"]}: [{m["sender"]}] {m["text"]}' for m in messages)
        return textwrap.dedent(f"""
            Business: {ctx["business_name"]} ({ctx["business_type"]})
            Group: {ctx["group_name"]}
            Description: {ctx["description"]}
            Messages (no labels — discover clusters yourself):
            {msgs_text}
            Group into 3-6 clusters. JSON only.
        """).strip()

    else:  # prompt_gen
        msgs_text = "\n".join(f'[{m["sender"]}]: {m["text"]}' for m in messages[:20])
        return textwrap.dedent(f"""
            Business: {ctx["business_name"]} ({ctx["business_type"]})
            Group: {ctx["group_name"]}
            Description: {ctx["description"]}
            Sample messages:
            {msgs_text}
            Generate tailored prompt template. JSON only.
        """).strip()


# ---------------------------------------------------------------------------
# Agent action builder
# ---------------------------------------------------------------------------

def get_action(client: AzureOpenAI, task_type: str, observation: Dict) -> tuple[Dict, str, Optional[str]]:
    system = {
        "classify": CLASSIFY_SYSTEM,
        "cluster": CLUSTER_SYSTEM,
        "prompt_gen": PROMPT_GEN_SYSTEM,
    }[task_type]

    error = None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_prompt(task_type, observation)},
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = orjson.loads(raw)
        action_str = raw[:80].replace("\n", " ")

        if task_type == "classify":
            action = {"task_type": "classify", "classify_action": {"classifications": parsed["classifications"]}}
        elif task_type == "cluster":
            action = {"task_type": "cluster", "cluster_action": {"clusters": parsed["clusters"], "cluster_labels": parsed["cluster_labels"]}}
        else:
            action = {"task_type": "prompt_gen", "prompt_gen_action": {
                "prompt_template": parsed["prompt_template"],
                "reasoning": parsed["reasoning"],
                "identified_topics": parsed["identified_topics"],
            }}

    except Exception as exc:
        error = str(exc)
        action_str = "parse_error"
        msgs = observation.get("messages", [])
        if task_type == "classify":
            labels = observation.get("available_labels", ["general"])
            action = {"task_type": "classify", "classify_action": {"classifications": {m["id"]: labels[0] for m in msgs}}}
        elif task_type == "cluster":
            action = {"task_type": "cluster", "cluster_action": {"clusters": {"cluster_1": [m["id"] for m in msgs]}, "cluster_labels": {"cluster_1": "general discussion"}}}
        else:
            action = {"task_type": "prompt_gen", "prompt_gen_action": {
                "prompt_template": f"You are assisting {observation['business_context']['business_name']} group. Respond professionally to all queries.",
                "reasoning": "Fallback due to parse error.",
                "identified_topics": ["general"],
            }}

    return action, action_str, error


# ---------------------------------------------------------------------------
# Run one episode via HTTP
# ---------------------------------------------------------------------------

def run_episode(client: AzureOpenAI, task_type: str, business_type: str) -> float:
    task_name = f"{task_type}-{business_type}"
    print(f"\n[START] task={task_name} env=business-chat-env model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        reset_data = api_reset(task_type=task_type, business_type=business_type)
        observation = reset_data["observation"]

        ctx = observation["business_context"]
        print(f"[VERBOSE] Business : {ctx['business_name']}", flush=True)
        print(f"[VERBOSE] Task     : {task_type} | Messages: {len(observation['messages'])}", flush=True)

        for step in range(1, MAX_STEPS + 1):
            action, action_str, error = get_action(client, task_type, observation)
            result = api_step(action)

            reward = result["reward"]["score"]
            done = result["done"]
            feedback = result["reward"]["feedback"]
            breakdown = result["reward"]["breakdown"]

            rewards.append(reward)
            steps_taken = step
            observation = result["observation"]

            error_val = error if error else "null"
            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)
            print(f"[VERBOSE] Breakdown: {breakdown}", flush=True)
            print(f"[VERBOSE] Feedback : {feedback}", flush=True)

            # Show agent output details
            if task_type == "cluster" and "cluster_action" in action:
                for cid, label in action["cluster_action"]["cluster_labels"].items():
                    n = len(action["cluster_action"]["clusters"].get(cid, []))
                    print(f"[VERBOSE]   Cluster '{label}': {n} messages", flush=True)
            elif task_type == "prompt_gen" and "prompt_gen_action" in action:
                print(f"[VERBOSE] Topics: {action['prompt_gen_action']['identified_topics']}", flush=True)
                print(f"[VERBOSE] Prompt: {action['prompt_gen_action']['prompt_template'][:200]}...", flush=True)

            state = api_state()
            print(f"[VERBOSE] State: step={state['step']} best_score={state['cumulative_score']}", flush=True)

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= 0.9

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Checking server health...", flush=True)
    if not api_health():
        print(f"ERROR: Server not reachable at {SERVER_URL}. Run: poetry run python run.py")
        return

    print(f"Server is up at {SERVER_URL}", flush=True)

    if "azure.com" in API_BASE_URL:
        client = AzureOpenAI(
            azure_endpoint=API_BASE_URL,
            api_key=API_KEY,
            api_version="2025-04-01-preview",
        )
    else:
        client = OpenAI(base_url=API_BASE_URL or None, api_key=API_KEY)

    task_types = ["classify", "cluster", "prompt_gen"]
    # Set to None to run all 3 businesses, or a specific one e.g. "real_estate"
    target_business = os.getenv("TARGET_BUSINESS", None)
    business_types = [target_business] if target_business else ["electronics_retail", "restaurant_chain", "real_estate"]

    all_scores = []
    for task_type in task_types:
        for business_type in business_types:
            score = run_episode(client, task_type, business_type)
            all_scores.append(score)

    scores_str = ",".join(f"{s:.3f}" for s in all_scores)
    print(f"\n[SUMMARY] tasks={len(all_scores)} avg_score={sum(all_scores)/len(all_scores):.3f} scores={scores_str}", flush=True)


if __name__ == "__main__":
    main()
