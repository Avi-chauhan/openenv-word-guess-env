# test.py
import sys
sys.path.insert(0, '.')

from client import WordGuessEnv
from models import GuessAction

# A list of guesses to try in order
guesses = ["crane", "apple", "brave", "delta", "honey", "flame"]

with WordGuessEnv(base_url="https://avichauhan-word-guess-env.hf.space").sync() as env:

    obs = env.reset()
    print("=== Game Started ===")
    print(f"Message  : {obs.observation.message}")
    print(f"Attempts : {obs.observation.attempts_remaining}\n")

    for word in guesses:
        result = env.step(GuessAction(guess=word))
        print(f"=== Guess: {word} ===")
        print(f"Feedback : {result.observation.feedback}")
        print(f"Reward   : {result.reward}")
        print(f"Done     : {result.done}")
        print(f"Message  : {result.observation.message}\n")

        if result.done:
            break