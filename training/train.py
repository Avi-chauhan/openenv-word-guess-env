import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

"""
Module 5: Training an LLM with GRPO on WordGuessEnv
=====================================================
Uses your live HuggingFace Space as the RL environment.
Trains a small LLM (Qwen 0.5B) to guess words using reward signals.

Run on Colab (free T4 GPU):
    pip install -r training/requirements.txt
    python training/train.py
"""

import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from client import WordGuessEnv
from models import GuessAction

# ── GPU check ─────────────────────────────────────────────────────────────────
print(f"GPU available   : {torch.cuda.is_available()}")
print(f"GPU name        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU)'}")

has_gpu       = torch.cuda.is_available()
supports_bf16 = has_gpu and torch.cuda.is_bf16_supported()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_URL     = "https://avichauhan-word-guess-env.hf.space"
MAX_TURNS   = 6
NUM_SAMPLES = 64

SYSTEM_PROMPT = """You are playing a word guessing game similar to Wordle.
You must guess a secret 5-letter word.
After each guess you receive per-letter feedback:
  - correct        = right letter, right position
  - wrong_position = right letter, wrong position
  - absent         = letter not in word

Rules:
- Reply with ONLY a single 5-letter word, lowercase, no punctuation.
- Use the feedback to make smarter guesses.

Example response: crane"""

# ── Environment client ────────────────────────────────────────────────────────
print(f"Connecting to environment: {ENV_URL}")
env_client = WordGuessEnv(base_url=ENV_URL)

# ── Rollout function ──────────────────────────────────────────────────────────
def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict:
    tokenizer = trainer.processing_class
    all_prompt_ids     = []
    all_completion_ids = []
    all_logprobs       = []
    all_rewards        = []

    for base_prompt in prompts:
        with env_client.sync() as env:
            obs = env.reset()
            episode_reward     = 0.0
            episode_prompt_ids = []
            episode_comp_ids   = []
            episode_logprobs   = []

            for turn in range(MAX_TURNS):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"{base_prompt}\n\n"
                        f"Word length: {obs.observation.target_length} letters\n"
                        f"Attempts remaining: {obs.observation.attempts_remaining}\n"
                        f"Last feedback: {obs.observation.feedback}\n"
                        f"Message: {obs.observation.message}\n\n"
                        f"Your guess:"
                    )},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )

                outputs = generate_rollout_completions(trainer, [prompt_text])[0]
                completion_text = tokenizer.decode(
                    outputs["completion_ids"], skip_special_tokens=True
                ).strip().split()[0][:5].lower()

                episode_prompt_ids.extend(outputs["prompt_ids"])
                episode_comp_ids.extend(outputs["completion_ids"])
                episode_logprobs.extend(outputs["logprobs"])

                obs = env.step(GuessAction(guess=completion_text))
                episode_reward = float(obs.reward or 0.0)

                if obs.done:
                    break

            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_comp_ids)
            all_logprobs.append(episode_logprobs)
            all_rewards.append(episode_reward)

    return {
        "prompt_ids":     all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs":       all_logprobs,
        "env_reward":     all_rewards,
    }

# ── Reward function ───────────────────────────────────────────────────────────
def reward_from_env(completions, **kwargs):
    env_rewards = kwargs.get("env_reward", [])
    return [float(r) for r in env_rewards] if env_rewards else [0.0] * len(completions)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = Dataset.from_dict({
    "prompt": ["Guess the secret 5-letter word."] * NUM_SAMPLES
})

# ── Trainer ───────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="eager")

grpo_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    num_train_epochs=1,
    num_generations=2,
    max_completion_length=16,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    output_dir="./outputs/word-guess-grpo",
    logging_steps=1,
    report_to="none",
    bf16=supports_bf16,
    fp16=has_gpu and not supports_bf16,
    no_cuda=not has_gpu,
    gradient_checkpointing=True,
    vllm_gpu_memory_utilization=0.3,
    dataloader_pin_memory=False,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_from_env,
    train_dataset=dataset,
    rollout_func=rollout_func,
    args=grpo_args,
)

if __name__ == "__main__":
    print("Starting GRPO training on WordGuessEnv...")
    print(f"Model      : {MODEL_ID}")
    print(f"Environment: {ENV_URL}")
    print(f"Episodes   : {NUM_SAMPLES}")
    print(f"bf16       : {supports_bf16}")
    print(f"fp16       : {has_gpu and not supports_bf16}")
    trainer.train()
    print("Training complete! Model saved to ./outputs/word-guess-grpo")