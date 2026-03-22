# server/environment.py
import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment  # ← correct import
from openenv.core.env_server.types import State             # ← correct import

try:
    from ..models import GuessAction, GuessObservation
except ImportError:
    from models import GuessAction, GuessObservation

WORD_LIST = ["apple", "brave", "crane", "delta", "eager",
             "flame", "grace", "honey", "ivory", "joint"]

class WordGuessEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.secret = ""
        self.attempts = 6

    def reset(self, seed=None, episode_id=None, **kwargs) -> GuessObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.secret = random.choice(WORD_LIST)
        self.attempts = 6
        return GuessObservation(
            target_length=len(self.secret),
            feedback=[],
            attempts_remaining=self.attempts,
            message=f"Guess a {len(self.secret)}-letter word. You have {self.attempts} attempts.",
            done=False,
            reward=0.0
        )

    def step(self, action: GuessAction, timeout_s=None, **kwargs) -> GuessObservation:
        guess = action.guess.lower().strip()
        self._state.step_count += 1
        self.attempts -= 1

        feedback = self._grade(guess)
        correct = guess == self.secret
        done = correct or self.attempts == 0

        if correct:
            reward = 1.0
            msg = f"Correct! The word was '{self.secret}'"
        elif done:
            reward = 0.0
            msg = f"Out of attempts. The word was '{self.secret}'"
        else:
            reward = feedback.count("correct") / len(self.secret) * 0.5
            msg = f"{self.attempts} attempts remaining."

        return GuessObservation(
            target_length=len(self.secret),
            feedback=feedback,
            attempts_remaining=self.attempts,
            message=msg,
            done=done,
            reward=reward
        )

    def _grade(self, guess: str) -> list[str]:
        result = []
        for i, ch in enumerate(guess):
            if i >= len(self.secret):
                result.append("absent")
            elif ch == self.secret[i]:
                result.append("correct")
            elif ch in self.secret:
                result.append("wrong_position")
            else:
                result.append("absent")
        return result

    @property
    def state(self) -> State:
        return self._state