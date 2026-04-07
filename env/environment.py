import random
from pathlib import Path

import orjson
from typing import Optional

from env.models import (
    Action,
    BusinessContext,
    BusinessType,
    Message,
    Observation,
    Reward,
    ResetResult,
    StateResult,
    StepResult,
    TaskType,
)
from graders.classify_grader import ClassifyGrader
from graders.cluster_grader import ClusterGrader
from graders.prompt_grader import PromptGrader

DATA_DIR = Path(__file__).parent.parent / "data"

BUSINESS_DATA_FILES = {
    BusinessType.ELECTRONICS_RETAIL: DATA_DIR / "electronics_retail.json",
    BusinessType.RESTAURANT_CHAIN: DATA_DIR / "restaurant_chain.json",
    BusinessType.REAL_ESTATE: DATA_DIR / "real_estate.json",
}

MAX_STEPS = 3
DONE_SCORE_THRESHOLD = 0.9


class BusinessChatEnv:
    def __init__(self):
        self._task_type: TaskType = TaskType.CLASSIFY
        self._business_type: BusinessType = BusinessType.ELECTRONICS_RETAIL
        self._messages: list[Message] = []
        self._business_context: Optional[BusinessContext] = None
        self._available_labels: Optional[list[str]] = None
        self._step: int = 0
        self._done: bool = False
        self._last_score: float = 0.0          # fix #7: renamed from _cumulative_score
        self._best_score: float = 0.0          # track best score across steps
        self._reward_history: list[float] = []
        self._last_feedback: str = ""          # fix #6: feed last reward's feedback into next observation

        self._classify_grader = ClassifyGrader()
        self._cluster_grader = ClusterGrader()
        self._prompt_grader = PromptGrader()

    def reset(
        self,
        task_type: Optional[TaskType] = None,
        business_type: Optional[BusinessType] = None,
    ) -> ResetResult:
        self._task_type = task_type or random.choice(list(TaskType))
        self._business_type = business_type or random.choice(list(BusinessType))
        self._step = 0
        self._done = False
        self._last_score = 0.0
        self._best_score = 0.0
        self._reward_history = []
        self._last_feedback = ""

        data = self._load_business_data(self._business_type)
        self._business_context = BusinessContext(
            business_name=data["business_name"],
            business_type=self._business_type,
            group_name=data["group_name"],
            description=data["description"],
        )
        self._messages = [Message(**m) for m in data["messages"]]
        self._available_labels = data.get("available_labels")

        return ResetResult(observation=self._build_observation())

    def step(self, action: Action) -> StepResult:
        if self._business_context is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Fix #8: validate action matches current task
        self._validate_action(action)

        self._step += 1
        reward = self._grade(action)

        self._last_score = reward.score
        self._best_score = max(self._best_score, reward.score)
        self._reward_history.append(reward.score)
        self._last_feedback = reward.feedback  # fix #6: carry feedback into next obs

        self._done = self._step >= MAX_STEPS or reward.score >= DONE_SCORE_THRESHOLD

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=self._done,
            info={
                "step": self._step,
                "last_score": self._last_score,
                "best_score": self._best_score,
                "reward_history": self._reward_history,
            },
        )

    def state(self) -> StateResult:
        # Fix #10: handle pre-reset state gracefully
        if self._business_context is None:
            return StateResult(
                task_type=self._task_type,
                business_context=BusinessContext(
                    business_name="",
                    business_type=self._business_type,
                    group_name="",
                    description="Episode not initialized. Call /reset first.",
                ),
                step=0,
                done=False,
                best_score=0.0,
            )
        return StateResult(
            task_type=self._task_type,
            business_context=self._business_context,
            step=self._step,
            done=self._done,
            best_score=self._best_score,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_action(self, action: Action) -> None:
        """Fix #8: ensure action payload matches current task."""
        if action.task_type != self._task_type:
            raise ValueError(
                f"Action task_type '{action.task_type}' does not match "
                f"current task '{self._task_type}'."
            )
        if self._task_type == TaskType.CLASSIFY and action.classify_action is None:
            raise ValueError("classify_action is required for classify task.")
        if self._task_type == TaskType.CLUSTER and action.cluster_action is None:
            raise ValueError("cluster_action is required for cluster task.")
        if self._task_type == TaskType.PROMPT_GEN and action.prompt_gen_action is None:
            raise ValueError("prompt_gen_action is required for prompt_gen task.")

    def _build_observation(self) -> Observation:
        """Fix #6: include last_feedback so state evolves between steps."""
        instructions = self._get_instructions()
        if self._last_feedback and self._step > 0:
            instructions = f"[Previous feedback: {self._last_feedback}]\n\n{instructions}"
        return Observation(
            business_context=self._business_context,
            messages=self._messages,
            task_type=self._task_type,
            available_labels=self._available_labels if self._task_type == TaskType.CLASSIFY else None,
            step=self._step,
            instructions=instructions,
        )

    def _get_instructions(self) -> str:
        if self._task_type == TaskType.CLASSIFY:
            labels = ", ".join(self._available_labels or [])
            return (
                f"You are analyzing a {self._business_context.business_type.value} business chat group. "
                f"Classify each message into one of these labels: [{labels}]. "
                f"Return a ClassifyAction with message_id -> label mappings."
            )
        elif self._task_type == TaskType.CLUSTER:
            return (
                f"You are analyzing a {self._business_context.business_type.value} business chat group. "
                f"Group the messages into 3-6 meaningful clusters based on topic/intent. "
                f"Give each cluster a descriptive human-readable label. "
                f"Every message must be assigned to exactly one cluster. "
                f"Return a ClusterAction with cluster_id -> [message_ids] and cluster_id -> label."
            )
        else:
            return (
                f"You are analyzing a {self._business_context.business_type.value} business chat group "
                f"called '{self._business_context.group_name}'. "
                f"Study the conversation patterns, topics, and communication style. "
                f"Generate a highly tailored prompt template for this specific business group. "
                f"The prompt must mention the business name/type and cover all key topics. "
                f"Return a PromptGenAction with prompt_template, reasoning, and identified_topics."
            )

    def _grade(self, action: Action) -> Reward:
        if self._task_type == TaskType.CLASSIFY:
            return self._classify_grader.grade(action.classify_action, self._messages)
        elif self._task_type == TaskType.CLUSTER:
            return self._cluster_grader.grade(action.cluster_action, self._messages)
        else:
            return self._prompt_grader.grade(
                action.prompt_gen_action,
                self._messages,
                self._business_context,
            )

    def _load_business_data(self, business_type: BusinessType) -> dict:
        path = BUSINESS_DATA_FILES[business_type]
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}. Run data/generate_data.py first."
            )
        with open(path, "rb") as f:
            return orjson.loads(f.read())
