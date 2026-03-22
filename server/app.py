# server/app.py
from openenv.core.env_server.http_server import create_app

try:
    from ..models import GuessAction, GuessObservation
    from .environment import WordGuessEnvironment
except ImportError:
    from models import GuessAction, GuessObservation
    from server.environment import WordGuessEnvironment

app = create_app(
    WordGuessEnvironment,
    GuessAction,
    GuessObservation,
    env_name="word_guess",
    max_concurrent_envs=10,
)