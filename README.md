---
title: Business Chat OpenEnv
emoji: 💬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Business Chat OpenEnv

An OpenEnv environment that trains and evaluates AI agents on real-world business chat group analysis. Agents learn to classify messages, discover topic clusters, and generate highly tailored prompt templates for specific business contexts.

## Motivation

Businesses using group chats (WhatsApp, Slack, Teams) generate large volumes of unstructured conversation data. Each business has a unique communication style, terminology, and set of recurring query types. This environment teaches agents to:

1. Understand the context of a business chat group
2. Identify patterns and topics in messages
3. Generate prompts that are deeply tailored to each specific business

---

## Environment Description

### Business Types
| Business | Group | Description |
|---|---|---|
| Electronics Retail | Seller Support | Product inquiries, pricing, orders, complaints |
| Restaurant Chain | Operations | Menu queries, reservations, inventory, staff updates |
| Real Estate | Agent Group | Property inquiries, site visits, pricing, legal queries |

### Tasks

| Task | Difficulty | Description |
|---|---|---|
| `classify` | Easy | Classify each message using provided label set |
| `cluster` | Medium | Discover topic clusters with no labels given |
| `prompt_gen` | Hard | Generate a business-specific LLM prompt template |

---

## Action & Observation Spaces

### Observation
```json
{
  "business_context": {
    "business_name": "TechZone Electronics",
    "business_type": "electronics_retail",
    "group_name": "Seller Support Group",
    "description": "..."
  },
  "messages": [
    {"id": "msg_001", "sender": "Aisha", "text": "...", "timestamp": "..."}
  ],
  "task_type": "classify",
  "available_labels": ["product_inquiry", "pricing", "complaint", "order_update", "general"],
  "instructions": "...",
  "step": 0
}
```

### Actions

**Classify (Easy)**
```json
{
  "task_type": "classify",
  "classify_action": {
    "classifications": {"msg_001": "product_inquiry", "msg_002": "pricing"}
  }
}
```

**Cluster (Medium)**
```json
{
  "task_type": "cluster",
  "cluster_action": {
    "clusters": {"cluster_1": ["msg_001", "msg_002"], "cluster_2": ["msg_003"]},
    "cluster_labels": {"cluster_1": "product inquiries", "cluster_2": "pricing discussions"}
  }
}
```

**Prompt Generation (Hard)**
```json
{
  "task_type": "prompt_gen",
  "prompt_gen_action": {
    "prompt_template": "You are assisting TechZone Electronics seller support group...",
    "reasoning": "The group frequently discusses product specs and order tracking...",
    "identified_topics": ["product_inquiry", "order_tracking", "pricing"]
  }
}
```

---

## Reward Function

| Task | Signal | Breakdown |
|---|---|---|
| classify | Accuracy + coverage | 70% accuracy, 30% coverage penalty |
| cluster | Coherence + coverage + label quality | 45% keyword coherence, 25% coverage, 15% label quality, 15% structure |
| prompt_gen | LLM-as-judge | 40% business relevance, 35% topic coverage, 25% prompt quality |

All rewards are in `[0.0, 1.0]`. Partial credit is given at every step.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Get current state |
| `GET` | `/health` | Health check |

---

## Setup & Usage

### Local (Poetry)
```bash
poetry install
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t business-chat-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=<your-endpoint> \
  -e MODEL_NAME=<your-model> \
  -e HF_TOKEN=<your-key> \
  business-chat-env
```

### Run Inference
```bash
export API_BASE_URL=<your-azure-endpoint>
export MODEL_NAME=<your-model>
export HF_TOKEN=<your-key>

poetry run python inference.py
```

### Environment Variables
| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint (Azure/HF) |
| `MODEL_NAME` | Model deployment name |
| `HF_TOKEN` | API key |

---

## Baseline Scores

| Task | Business | Score |
|---|---|---|
| classify | Electronics Retail | ~0.80 |
| cluster | Restaurant Chain | ~0.65 |
| prompt_gen | Real Estate | ~0.60 |

---

## Project Structure
```
├── env/
│   ├── environment.py     # Core OpenEnv (step/reset/state)
│   ├── models.py          # Pydantic models, enums, DTOs
│   └── tasks/             # Task definitions
├── graders/
│   ├── classify_grader.py # Exact match accuracy grader
│   ├── cluster_grader.py  # Cluster purity grader
│   └── prompt_grader.py   # LLM-as-judge grader
├── data/                  # Synthetic business chat JSON files
├── server/
│   └── app.py             # FastAPI server (thin controller)
├── inference.py           # Baseline inference script
├── openenv.yaml           # OpenEnv spec
└── Dockerfile
```
