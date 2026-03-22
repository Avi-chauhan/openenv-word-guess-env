# client.py
from typing import Dict, Any
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import GuessAction, GuessObservation


class WordGuessEnv(EnvClient[GuessAction, GuessObservation, State]):

    def _step_payload(self, action: GuessAction) -> Dict[str, Any]:
        """Convert GuessAction → JSON to send to server."""
        return {"guess": action.guess}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GuessObservation]:
        """Convert server JSON response → StepResult."""
        obs_data = payload.get("observation", {})
        observation = GuessObservation(
            target_length=obs_data.get("target_length", 5),
            feedback=obs_data.get("feedback", []),
            attempts_remaining=obs_data.get("attempts_remaining", 6),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Convert server JSON response → State."""
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )