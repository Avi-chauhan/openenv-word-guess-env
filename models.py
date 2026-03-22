from openenv.core.env_server.types import Action, Observation 
from pydantic import Field

class GuessAction(Action):
    guess: str = Field(..., description="The word being guessed")

class GuessObservation(Observation):
    target_length: int = Field(default=5)
    feedback: list[str] = Field(default=[])
    attempts_remaining: int = Field(default=6)
    message: str = Field(default="")