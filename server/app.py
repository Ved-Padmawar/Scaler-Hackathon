from fastapi import FastAPI, HTTPException, Header
from typing import Optional
import uuid

from env.environment import BusinessChatEnv
from env.models import (
    BusinessType,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
    TaskType,
)

app = FastAPI(
    title="Business Chat OpenEnv",
    description="OpenEnv environment for business chat analysis and prompt generation.",
    version="1.0.0",
)

# Fix #9: per-session env instances keyed by session ID
_sessions: dict[str, BusinessChatEnv] = {}


def _get_env(session_id: str) -> BusinessChatEnv:
    if session_id not in _sessions:
        _sessions[session_id] = BusinessChatEnv()
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse)
async def reset(
    request: ResetRequest = ResetRequest(),
    x_session_id: Optional[str] = Header(default=None),
):
    # Fix session bug: always use a consistent session ID, return it in response
    session_id = x_session_id or "default"
    env = _get_env(session_id)
    try:
        task_type = TaskType(request.task_type) if request.task_type else None
        business_type = BusinessType(request.business_type) if request.business_type else None
        result = env.reset(task_type=task_type, business_type=business_type)
        obs = result.observation.model_dump_for_agent()
        obs["session_id"] = session_id  # return session_id so client can reuse it
        return ResetResponse(observation=obs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse)
async def step(
    request: StepRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    session_id = x_session_id or "default"
    env = _get_env(session_id)
    try:
        result = env.step(request.action)
        return StepResponse(
            observation=result.observation.model_dump_for_agent(),
            reward=result.reward,
            done=result.done,
            info=result.info,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=StateResponse)
async def state(x_session_id: Optional[str] = Header(default=None)):
    session_id = x_session_id or "default"
    env = _get_env(session_id)
    try:
        result = env.state()  # Fix #10: now returns safe pre-reset state
        return StateResponse(
            task_type=result.task_type,
            business_context=result.business_context,
            step=result.step,
            done=result.done,
            cumulative_score=result.best_score,  # best_score exposed as cumulative_score in API
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
