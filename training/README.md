# Module 5 — Training with OpenEnv + TRL

This module trains a small LLM using **GRPO** on the
live WordGuessEnv environment.

## What is GRPO?

Group Relative Policy Optimisation — for each prompt, it:
1. Generates multiple completions (guesses)
2. Scores each with the reward function (your env's grader)
3. Updates the model to prefer higher-scoring responses

Over thousands of episodes, the LLM learns to guess words
better purely from reward signals — no labelled data needed.

## Architecture
```
Dataset prompt
     ↓
GRPOTrainer calls rollout_func()
     ↓
rollout_func() connects to live HF Space
     ↓
env.reset() → new secret word
     ↓
LLM generates guess → env.step(guess) → reward
     ↓  (repeat up to 6 turns)
Returns: prompt_ids, completion_ids, logprobs, env_reward
     ↓
reward_from_env() extracts env_reward
     ↓
GRPO updates model weights
```

## Run on Google Colab (free T4 GPU)
```python
# Cell 1 — Install
!pip install trl transformers torch datasets openenv-core

# Cell 2 — Clone repo
!git clone https://github.com/avichauhan/word-guess-env.git
%cd word-guess-env
!git checkout module-5-training

# Cell 3 — Train
!python training/train.py
```

## Requirements

- GPU: T4 or better (free Colab works)
- RAM: 8GB+
- The live HF Space must be running:
  https://huggingface.co/spaces/avichauhan/word-guess-env