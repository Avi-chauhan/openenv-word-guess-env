# Word Guess Environment

A Wordle-style reinforcement learning environment built on
[Meta's OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

Built as part of the
[Meta PyTorch OpenEnv Hackathon x SST 2026](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon).

## What is this?

An RL environment where an LLM agent learns to guess a secret 5-letter word in 6 attempts, similar to Wordle. The agent receives per-letter feedback and a shaped reward signal to guide learning.

## How it works
```
reset()  →  "Guess a 5-letter word. You have 6 attempts."
step("crane")  →  feedback: ["correct", "absent", "correct", "absent", "wrong_position"]
               →  reward: 0.4
step("grace")  →  reward: 1.0  →  done: True
```

## Reward structure

| Outcome | Reward |
|---|---|
| Correct word | 1.0 |
| Partial (n correct positions) | n/5 × 0.5 |
| Out of attempts | 0.0 |

## Quick start

### Run locally
```bash
git clone https://github.com/YOUR_USERNAME/word-guess-env.git
cd word-guess-env
uv venv && source .venv/bin/activate
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -f server/Dockerfile -t word-guess-env:latest .
docker run -p 8000:8000 word-guess-env:latest
```

### Use the environment
```python
from client import WordGuessEnv
from models import GuessAction

with WordGuessEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset()
    result = env.step(GuessAction(guess="crane"))
    print(result.observation.feedback)
    print(result.reward)
```

### Use from HuggingFace (coming soon)
```python
from client import WordGuessEnv
from models import GuessAction

with WordGuessEnv(base_url="https://YOUR_HF_USERNAME-word-guess-env.hf.space").sync() as env:
    obs = env.reset()
```

## Project structure
```
word-guess-env/
├── models.py          # Action + Observation types
├── client.py          # EnvClient for training frameworks
├── __init__.py
├── server/
│   ├── environment.py # Core logic: reset(), step(), grader
│   ├── app.py         # FastAPI server
│   ├── requirements.txt
│   └── Dockerfile
└── test.py            # Local test script
```

## Tech stack

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — RL environment framework by Meta & HuggingFace
- [FastAPI](https://fastapi.tiangolo.com/) — WebSocket server
- [Pydantic](https://docs.pydantic.dev/) — Type-safe actions and observations
- [Docker](https://www.docker.com/) — Containerised deployment
- [HuggingFace Spaces](https://huggingface.co/spaces) — Live deployment

## Compatible training frameworks

- [TRL](https://huggingface.co/docs/trl/openenv)
- [SkyRL](https://skyrl.readthedocs.io/en/latest/examples/openenv.html)
- [Unsloth](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb)
- [ART](https://art.openpipe.ai/integrations/openenv-integration)