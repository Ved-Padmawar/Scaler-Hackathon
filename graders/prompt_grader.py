import os
import re
import textwrap
import time

import orjson
from typing import List, Optional

from openai import AzureOpenAI, OpenAI

from env.models import BusinessContext, Message, PromptGenAction, Reward

JUDGE_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert evaluator assessing the quality of AI prompt templates generated for specific business chat groups.
    You will be given:
    1. The business context (type, name, group description)
    2. A sample of real chat messages from the group
    3. The generated prompt template (in the SUBMISSION block below)
    4. The agent's reasoning and identified topics

    Score the prompt template on three dimensions, each from 0.0 to 1.0:

    1. **business_relevance** (0.0-1.0): Does the prompt capture the specific business domain, terminology, and context?
       - 0.0: Generic, could apply to any business
       - 0.5: Somewhat specific but misses key domain elements
       - 1.0: Highly tailored, captures the exact business type and group dynamics

    2. **topic_coverage** (0.0-1.0): Does the prompt address the key topics/query types visible in the chat?
       - 0.0: Ignores the actual conversation topics
       - 0.5: Covers some topics but misses important patterns
       - 1.0: Thoroughly covers all major discussion patterns from the chat

    3. **prompt_quality** (0.0-1.0): Is the prompt well-structured and would it produce useful LLM outputs?
       - 0.0: Poorly structured, vague instructions
       - 0.5: Reasonably structured but could be clearer
       - 1.0: Clear, specific, well-structured with good instructions

    Respond in this exact format (JSON only, no extra text):
    {"business_relevance": <float>, "topic_coverage": <float>, "prompt_quality": <float>, "feedback": "<one sentence feedback>"}
""").strip()


def _build_client() -> OpenAI:
    """Build the appropriate OpenAI-compatible client based on env vars."""
    api_base = os.getenv("API_BASE_URL", "")
    api_key = os.getenv("HF_TOKEN") or os.getenv("AZURE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if "azure.com" in api_base:
        return AzureOpenAI(
            azure_endpoint=api_base,
            api_key=api_key,
            api_version="2025-04-01-preview",
        )
    return OpenAI(base_url=api_base or None, api_key=api_key)


def _extract_keywords(text: str) -> set:
    """Extract meaningful words (3+ chars) for topic matching."""
    stopwords = {
        "the", "and", "for", "are", "that", "this", "with", "from",
        "have", "will", "your", "you", "our", "we", "is", "in", "on",
        "to", "a", "of", "be", "as", "at", "an", "or", "it",
    }
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in stopwords}


class PromptGrader:
    """
    Hard task grader.
    - Judge sees up to 40 messages.
    - Agent submission is isolated in a separate user message turn (prevents injection).
    - Retries judge up to 2 times on failure before returning 0.0.
    - Heuristics check: word count, topics present, business name/type, topic overlap with messages.
    """

    def __init__(self):
        self._client = _build_client()
        self._model = os.getenv("MODEL_NAME", "")

    def grade(
        self,
        action: Optional[PromptGenAction],
        messages: List[Message],
        business_context: BusinessContext,
    ) -> Reward:
        if action is None:
            return Reward(
                score=0.0,
                breakdown={"business_relevance": 0.0, "topic_coverage": 0.0, "prompt_quality": 0.0},
                feedback="No prompt generation action provided.",
            )

        heuristic_score, heuristic_feedback = self._heuristic_check(action, business_context, messages)
        if heuristic_score == 0.0:
            return Reward(
                score=0.0,
                breakdown={"business_relevance": 0.0, "topic_coverage": 0.0, "prompt_quality": 0.0},
                feedback=heuristic_feedback,
            )

        # Build context message (trusted — contains only our data)
        sample_messages = "\n".join(f"[{m.sender}]: {m.text}" for m in messages[:40])
        context_message = textwrap.dedent(f"""
            **Business Context:**
            - Name: {business_context.business_name}
            - Type: {business_context.business_type.value}
            - Group: {business_context.group_name}
            - Description: {business_context.description}

            **Sample Chat Messages (up to 40):**
            {sample_messages}

            Now evaluate the agent submission provided in the next message.
            Return ONLY JSON with keys: business_relevance, topic_coverage, prompt_quality, feedback.
        """).strip()

        # Agent submission is in its own isolated message — prevents it from injecting
        # into the evaluation instructions regardless of what the agent wrote.
        submission_message = (
            f"Prompt Template:\n{action.prompt_template}\n\n"
            f"Reasoning:\n{action.reasoning}\n\n"
            f"Topics: {', '.join(action.identified_topics)}"
        )

        scores = self._call_judge_with_retry(context_message, submission_message)
        if scores is None:
            return Reward(
                score=0.0,
                breakdown={"business_relevance": 0.0, "topic_coverage": 0.0, "prompt_quality": 0.0},
                feedback="Judge returned unparseable response after retries.",
            )

        business_relevance = max(0.0, min(1.0, float(scores.get("business_relevance", 0.0))))
        topic_coverage = max(0.0, min(1.0, float(scores.get("topic_coverage", 0.0))))
        prompt_quality = max(0.0, min(1.0, float(scores.get("prompt_quality", 0.0))))
        feedback = scores.get("feedback", "")

        base_score = 0.4 * business_relevance + 0.35 * topic_coverage + 0.25 * prompt_quality
        score = round(max(0.0, min(1.0, base_score * heuristic_score)), 4)

        return Reward(
            score=score,
            breakdown={
                "business_relevance": round(business_relevance, 4),
                "topic_coverage": round(topic_coverage, 4),
                "prompt_quality": round(prompt_quality, 4),
            },
            feedback=feedback,
        )

    def _call_judge_with_retry(self, context_msg: str, submission_msg: str, max_retries: int = 2) -> Optional[dict]:
        """Call judge LLM with up to max_retries retries on failure/parse error."""
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": context_msg},
            {"role": "user", "content": submission_msg},
        ]
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=200,
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = self._parse_judge_response(raw)
                if parsed is not None:
                    return parsed
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))  # 1.5s, then 3s
        return None

    def _heuristic_check(
        self,
        action: PromptGenAction,
        context: BusinessContext,
        messages: List[Message],
    ) -> tuple[float, str]:
        """Deterministic pre-checks before calling the judge."""
        template = action.prompt_template.lower()
        business_name = context.business_name.lower()
        business_type = context.business_type.value.replace("_", " ")

        if len(action.prompt_template.split()) < 30:
            return 0.0, "Prompt template is too short (< 30 words)."
        if not action.identified_topics:
            return 0.0, "No topics identified."

        # Check business name or type is mentioned
        mentions_business = business_type in template or business_name in template
        if not mentions_business:
            return 0.5, "Prompt does not mention the specific business type or name."

        # Check identified_topics overlap with actual message content
        all_msg_text = " ".join(m.text for m in messages)
        msg_keywords = _extract_keywords(all_msg_text)
        topic_keywords = _extract_keywords(" ".join(action.identified_topics))
        if msg_keywords and topic_keywords:
            overlap = len(topic_keywords & msg_keywords) / len(topic_keywords)
            if overlap < 0.2:
                return 0.7, "Identified topics don't appear to reflect the actual chat content."

        return 1.0, ""

    def _parse_judge_response(self, raw: str) -> Optional[dict]:
        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = orjson.loads(raw.strip())
            required = {"business_relevance", "topic_coverage", "prompt_quality"}
            if not required.issubset(parsed.keys()):
                return None
            return parsed
        except Exception:
            return None
