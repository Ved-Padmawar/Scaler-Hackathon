from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    CLASSIFY = "classify"
    CLUSTER = "cluster"
    PROMPT_GEN = "prompt_gen"


class BusinessType(str, Enum):
    ELECTRONICS_RETAIL = "electronics_retail"
    RESTAURANT_CHAIN = "restaurant_chain"
    REAL_ESTATE = "real_estate"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    id: str
    sender: str
    text: str
    timestamp: str
    ground_truth_label: Optional[str] = None  # hidden from agent, used by grader


class BusinessContext(BaseModel):
    business_name: str
    business_type: BusinessType
    group_name: str
    description: str


# ---------------------------------------------------------------------------
# Observation (what the agent sees)
# ---------------------------------------------------------------------------

# Action schemas shown to the agent in every observation response
_ACTION_SCHEMAS: Dict[str, Any] = {
    "classify": {
        "task_type": "classify",
        "classify_action": {
            "classifications": {
                "msg_001": "<one of available_labels>",
                "msg_002": "<one of available_labels>",
                "...": "classify every message"
            }
        }
    },
    "cluster": {
        "task_type": "cluster",
        "cluster_action": {
            "clusters": {
                "cluster_1": ["msg_001", "msg_002"],
                "cluster_2": ["msg_003", "msg_004"],
                "...": "3 to 6 clusters, every message in exactly one cluster"
            },
            "cluster_labels": {
                "cluster_1": "descriptive label (2+ words)",
                "cluster_2": "descriptive label (2+ words)"
            }
        }
    },
    "prompt_gen": {
        "task_type": "prompt_gen",
        "prompt_gen_action": {
            "prompt_template": "100+ word prompt tailored to this specific business and group",
            "reasoning": "why this prompt fits this business context",
            "identified_topics": ["topic1", "topic2", "topic3"]
        }
    }
}


class Observation(BaseModel):
    business_context: BusinessContext
    messages: List[Message]
    task_type: TaskType
    available_labels: Optional[List[str]] = None  # only for classify task
    step: int = Field(default=0, ge=0)
    instructions: str = ""

    def model_dump_for_agent(self) -> Dict[str, Any]:
        """Return observation without ground truth labels, with action schema."""
        data = self.model_dump()
        for msg in data["messages"]:
            msg.pop("ground_truth_label", None)
        data["action_schema"] = _ACTION_SCHEMAS[self.task_type]
        return data


# ---------------------------------------------------------------------------
# Actions (what the agent submits)
# ---------------------------------------------------------------------------

class ClassifyAction(BaseModel):
    """Easy task: assign a label to each message."""
    classifications: Dict[str, str] = Field(
        description="message_id -> label"
    )


class ClusterAction(BaseModel):
    """Medium task: group messages into clusters with human-readable labels."""
    clusters: Dict[str, List[str]] = Field(
        description="cluster_id -> list of message_ids"
    )
    cluster_labels: Dict[str, str] = Field(
        description="cluster_id -> human-readable label name"
    )


class PromptGenAction(BaseModel):
    """Hard task: generate a business-specific prompt template."""
    prompt_template: str = Field(
        min_length=50,
        description="The generated prompt template tailored to this business"
    )
    reasoning: str = Field(
        min_length=20,
        description="Reasoning behind the prompt design choices"
    )
    identified_topics: List[str] = Field(
        description="Key topics/patterns identified from the chat"
    )


class Action(BaseModel):
    task_type: TaskType
    classify_action: Optional[ClassifyAction] = None
    cluster_action: Optional[ClusterAction] = None
    prompt_gen_action: Optional[PromptGenAction] = None

    @field_validator("classify_action", "cluster_action", "prompt_gen_action", mode="before")
    @classmethod
    def check_action_present(cls, v, info):
        # Structural validation happens in env._validate_action at runtime
        return v


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    feedback: str

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)


# ---------------------------------------------------------------------------
# Step / Reset / State results (OpenEnv spec)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)  # fix #12: mutable default


class ResetResult(BaseModel):
    observation: Observation


class StateResult(BaseModel):
    task_type: TaskType
    business_context: BusinessContext
    step: int
    done: bool
    best_score: float = Field(ge=0.0, le=1.0)  # fix #7: renamed from cumulative_score


# ---------------------------------------------------------------------------
# API DTOs (server layer)
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)  # fix #12: mutable default


class ResetRequest(BaseModel):
    task_type: Optional[TaskType] = None
    business_type: Optional[BusinessType] = None


class ResetResponse(BaseModel):
    observation: Dict[str, Any]


class StateResponse(BaseModel):
    task_type: TaskType
    business_context: BusinessContext
    step: int
    done: bool
    cumulative_score: float
