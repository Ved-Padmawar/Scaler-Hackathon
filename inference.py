"""
Inference Script — Business Chat OpenEnv
=========================================
Runs an LLM agent against all 3 tasks (classify, cluster, prompt_gen)
across all 3 business types and emits mandatory [START]/[STEP]/[END] logs.

Environment variables required:
    API_BASE_URL   Azure/HF endpoint
    MODEL_NAME     Model deployment name
    HF_TOKEN       API key
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

from env.environment import BusinessChatEnv
from env.models import (
    Action,
    BusinessType,
    ClassifyAction,
    ClusterAction,
    PromptGenAction,
    TaskType,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("AZURE_API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "business-chat-env"
MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 2048
SUCCESS_THRESHOLD = 0.9  # fix #4: align with env's done threshold

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
      "cluster_labels": {"cluster_1": "product inquiries", "cluster_2": "pricing questions"}
    }
    Every message must be assigned to exactly one cluster. Labels must be descriptive.
""").strip()

PROMPT_GEN_SYSTEM = textwrap.dedent("""
    You are an expert prompt engineer specializing in business communication.
    Given a business chat group's context and sample messages, generate a highly tailored prompt template.
    The prompt will be used by an LLM to process future messages from this group.
    Respond with valid JSON only in this exact format:
    {
      "prompt_template": "You are assisting a [business] ... [full prompt here]",
      "reasoning": "Explanation of design choices...",
      "identified_topics": ["topic1", "topic2", "topic3"]
    }
    The prompt_template must be at least 100 words, specific to this business, and cover all key topics.
""").strip()


def build_classify_prompt(observation: Dict[str, Any]) -> str:
    ctx = observation["business_context"]
    messages = observation["messages"]
    labels = observation["available_labels"]
    msgs_text = "\n".join(f'{m["id"]}: [{m["sender"]}] {m["text"]}' for m in messages)
    return textwrap.dedent(f"""
        Business: {ctx["business_name"]} ({ctx["business_type"]})
        Group: {ctx["group_name"]}
        Available labels: {labels}

        Messages to classify:
        {msgs_text}

        Classify every message. Return JSON only.
    """).strip()


def build_cluster_prompt(observation: Dict[str, Any]) -> str:
    ctx = observation["business_context"]
    messages = observation["messages"]
    msgs_text = "\n".join(f'{m["id"]}: [{m["sender"]}] {m["text"]}' for m in messages)
    return textwrap.dedent(f"""
        Business: {ctx["business_name"]} ({ctx["business_type"]})
        Group: {ctx["group_name"]}
        Description: {ctx["description"]}

        Messages to cluster (no labels given — discover them yourself):
        {msgs_text}

        Group into 3-6 meaningful clusters. Return JSON only.
    """).strip()


def build_prompt_gen_prompt(observation: Dict[str, Any]) -> str:
    ctx = observation["business_context"]
    messages = observation["messages"][:15]
    msgs_text = "\n".join(f'[{m["sender"]}]: {m["text"]}' for m in messages)
    return textwrap.dedent(f"""
        Business: {ctx["business_name"]} ({ctx["business_type"]})
        Group: {ctx["group_name"]}
        Description: {ctx["description"]}

        Sample chat messages:
        {msgs_text}

        Generate a highly tailored prompt template for this specific business group. Return JSON only.
    """).strip()

# ---------------------------------------------------------------------------
# Agent — calls LLM and parses response into Action
# ---------------------------------------------------------------------------

def get_action(
    client: OpenAI,
    task_type: str,
    observation: Dict[str, Any],
) -> tuple[Action, str, Optional[str]]:
    if task_type == TaskType.CLASSIFY:
        system = CLASSIFY_SYSTEM
        user = build_classify_prompt(observation)
    elif task_type == TaskType.CLUSTER:
        system = CLUSTER_SYSTEM
        user = build_cluster_prompt(observation)
    else:
        system = PROMPT_GEN_SYSTEM
        user = build_prompt_gen_prompt(observation)

    error = None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        action_str = raw[:80].replace("\n", " ")

        if task_type == TaskType.CLASSIFY:
            action = Action(
                task_type=TaskType.CLASSIFY,
                classify_action=ClassifyAction(classifications=parsed["classifications"]),
            )
        elif task_type == TaskType.CLUSTER:
            action = Action(
                task_type=TaskType.CLUSTER,
                cluster_action=ClusterAction(
                    clusters=parsed["clusters"],
                    cluster_labels=parsed["cluster_labels"],
                ),
            )
        else:
            action = Action(
                task_type=TaskType.PROMPT_GEN,
                prompt_gen_action=PromptGenAction(
                    prompt_template=parsed["prompt_template"],
                    reasoning=parsed["reasoning"],
                    identified_topics=parsed["identified_topics"],
                ),
            )

    except Exception as exc:
        error = str(exc)
        action_str = "parse_error"
        # Fallback minimal action
        if task_type == TaskType.CLASSIFY:
            msgs = observation.get("messages", [])
            labels = observation.get("available_labels", ["general"])
            action = Action(
                task_type=TaskType.CLASSIFY,
                classify_action=ClassifyAction(
                    classifications={m["id"]: labels[0] for m in msgs}
                ),
            )
        elif task_type == TaskType.CLUSTER:
            msgs = observation.get("messages", [])
            action = Action(
                task_type=TaskType.CLUSTER,
                cluster_action=ClusterAction(
                    clusters={"cluster_1": [m["id"] for m in msgs]},
                    cluster_labels={"cluster_1": "general discussion"},
                ),
            )
        else:
            action = Action(
                task_type=TaskType.PROMPT_GEN,
                prompt_gen_action=PromptGenAction(
                    prompt_template="You are a helpful assistant for this business group. Respond professionally to all queries and provide accurate information based on the business context.",
                    reasoning="Fallback due to parse error.",
                    identified_topics=["general"],
                ),
            )

    return action, action_str, error


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------

def run_episode(
    client: OpenAI,
    env: BusinessChatEnv,
    task_type: TaskType,
    business_type: BusinessType,
) -> float:
    task_name = f"{task_type.value}-{business_type.value}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        reset_result = env.reset(task_type=task_type, business_type=business_type)
        observation = reset_result.observation.model_dump_for_agent()

        # Verbose: show what the agent sees
        ctx = observation["business_context"]
        print(f"\n{'='*60}", flush=True)
        print(f"[VERBOSE] Business : {ctx['business_name']} ({ctx['business_type']})", flush=True)
        print(f"[VERBOSE] Group    : {ctx['group_name']}", flush=True)
        print(f"[VERBOSE] Task     : {task_type.value}", flush=True)
        if observation.get("available_labels"):
            print(f"[VERBOSE] Labels   : {observation['available_labels']}", flush=True)
        print(f"[VERBOSE] Messages ({len(observation['messages'])} total, showing first 5):", flush=True)
        for m in observation["messages"][:5]:
            print(f"[VERBOSE]   [{m['id']}] {m['sender']}: {m['text'][:80]}", flush=True)
        print(f"{'='*60}\n", flush=True)

        for step in range(1, MAX_STEPS + 1):
            action, action_str, error = get_action(client, task_type, observation)
            result = env.step(action)

            reward = result.reward.score
            done = result.done

            rewards.append(reward)
            steps_taken = step
            observation = result.observation.model_dump_for_agent()

            # Verbose: show what the agent produced
            if task_type == TaskType.CLASSIFY and action.classify_action:
                sample = dict(list(action.classify_action.classifications.items())[:5])
                print(f"[VERBOSE] Classifications (first 5): {sample}", flush=True)
            elif task_type == TaskType.CLUSTER and action.cluster_action:
                for cid, label in action.cluster_action.cluster_labels.items():
                    msgs = action.cluster_action.clusters.get(cid, [])
                    print(f"[VERBOSE] Cluster '{label}': {len(msgs)} messages", flush=True)
            elif task_type == TaskType.PROMPT_GEN and action.prompt_gen_action:
                print(f"[VERBOSE] Topics: {action.prompt_gen_action.identified_topics}", flush=True)
                print(f"[VERBOSE] Prompt template:\n{action.prompt_gen_action.prompt_template}\n", flush=True)
                print(f"[VERBOSE] Reasoning: {action.prompt_gen_action.reasoning}", flush=True)

            print(f"[VERBOSE] Reward breakdown: {result.reward.breakdown}", flush=True)
            print(f"[VERBOSE] Feedback: {result.reward.feedback}\n", flush=True)

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if "azure.com" in API_BASE_URL:
        client = AzureOpenAI(
            azure_endpoint=API_BASE_URL,
            api_key=API_KEY,
            api_version="2025-04-01-preview",
        )
    else:
        client = OpenAI(base_url=API_BASE_URL or None, api_key=API_KEY)
    env = BusinessChatEnv()

    # 3 runs: one per task, using electronics_retail as the representative business
    tasks = [
        (TaskType.CLASSIFY, BusinessType.ELECTRONICS_RETAIL),
        (TaskType.CLUSTER, BusinessType.RESTAURANT_CHAIN),
        (TaskType.PROMPT_GEN, BusinessType.REAL_ESTATE),
    ]

    all_scores = []
    for task_type, business_type in tasks:
        score = run_episode(client, env, task_type, business_type)
        all_scores.append(score)

    scores_str = ",".join(f"{s:.3f}" for s in all_scores)
    print(
        f"\n[SUMMARY] tasks={len(tasks)} avg_score={sum(all_scores)/len(all_scores):.3f} scores={scores_str}",
        flush=True,
    )


if __name__ == "__main__":
    main()
