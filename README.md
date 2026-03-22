---
title: word-guess-env
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Word Guess Environment

A Wordle-style reinforcement learning environment built on [Meta's OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

Built as part of the [Meta PyTorch OpenEnv Hackathon x SST 2026](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon).

🤗 **Live on HuggingFace Spaces:**
`https://avichauhan-word-guess-env.hf.space`

---

## What is this?

An RL environment where an LLM agent learns to guess a secret 5-letter word in 6 attempts, similar to Wordle. The agent receives per-letter feedback and a shaped reward signal to guide learning.

## How it works
```
reset()        →  "Guess a 5-letter word. You have 6 attempts."

step("crane")  →  feedback: ["absent", "correct", "correct", "absent", "correct"]
               →  reward: 0.3
               →  done: False

step("brave")  →  feedback: ["correct", "correct", "correct", "correct", "correct"]
               →  reward: 1.0
               →  done: True
               →  message: "Correct! The word was 'brave'"
```

## Reward structure

| Outcome | Reward |
|---|---|
| Correct word guessed | `1.0` |
| Partial — n letters in correct position | `n/5 × 0.5` |
| Out of attempts | `0.0` |

The partial reward gradient encourages the LLM to make progressively
better guesses rather than guessing randomly.

---

## Quick start

### Run locally
```bash
git clone https://github.com/avichauhan/word-guess-env.git
cd word-guess-env

# Setup venv
uv venv && source .venv/bin/activate
uv sync
# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker
```bash
docker build -f server/Dockerfile -t word-guess-env:latest .
docker run -p 7860:7860 word-guess-env:latest
```

### Test locally
```python
# test.py
from client import WordGuessEnv
from models import GuessAction

with WordGuessEnv(base_url="http://localhost:8000").sync() as env: #8000/7860 based on how one runs project.
    obs = env.reset()
    print(obs.observation.message)

    result = env.step(GuessAction(guess="crane"))
    print(result.observation.feedback)
    print(result.reward)
```
```bash
python test.py
```

---

## Use from HuggingFace (live)

No local setup needed — connect directly to the hosted Space:
```python
from client import WordGuessEnv
from models import GuessAction

with WordGuessEnv(
    base_url="https://avichauhan-word-guess-env.hf.space"
).sync() as env:

    obs = env.reset()
    print(obs.observation.message)

    guesses = ["crane", "apple", "brave", "delta", "honey", "flame"]
    for word in guesses:
        result = env.step(GuessAction(guess=word))
        print(f"Guess: {word}")
        print(f"Feedback: {result.observation.feedback}")
        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}\n")
        if result.done:
            break
```

---

## Project structure
```
word-guess-env/
├── models.py              # Action + Observation types (Pydantic)
├── client.py              # EnvClient for training frameworks
├── __init__.py
├── openenv.yaml           # OpenEnv environment manifest
├── pyproject.toml         # Package config
├── test.py                # Local + remote test script
└── server/
    ├── environment.py     # Core logic: reset(), step(), grader
    ├── app.py             # FastAPI + WebSocket server
    ├── requirements.txt   # Server dependencies
    └── Dockerfile         # Container definition
```

---

## Environment interface

### Action
```python
class GuessAction(Action):
    guess: str  # The word the agent is guessing
```

### Observation
```python
class GuessObservation(Observation):
    target_length: int        # Number of letters in secret word
    feedback: list[str]       # Per-letter: "correct" | "wrong_position" | "absent"
    attempts_remaining: int   # How many guesses left
    message: str              # Human-readable status
    reward: float             # Step reward
    done: bool                # Episode over?
```

---

## Development

Install all dependencies with uv:
```bash
uv venv && source .venv/bin/activate
uv sync
```

> **Note:** `pyproject.toml` includes `[tool.hatch.build.targets.wheel] packages = ["."]`
> because project files live at root level rather than in a `word_guess_env/`
> subdirectory. Without this, `uv sync` fails with a hatchling wheel build error.

---

## Compatible training frameworks

| Framework | Link |
|---|---|
| TRL | [docs](https://huggingface.co/docs/trl/openenv) |
| SkyRL | [example](https://skyrl.readthedocs.io/en/latest/examples/openenv.html) |
| Unsloth | [colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb) |
| ART | [example](https://art.openpipe.ai/integrations/openenv-integration) |

---

## Tech stack

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — RL environment framework by Meta & HuggingFace
- [FastAPI](https://fastapi.tiangolo.com/) — WebSocket server
- [Pydantic](https://docs.pydantic.dev/) — Type-safe actions and observations
- [Docker](https://www.docker.com/) — Containerised deployment
- [HuggingFace Spaces](https://huggingface.co/spaces) — Live public deployment

---

## License

MIT