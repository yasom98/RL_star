import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rl_star.rl_grid import GridEnv, QLearner  # noqa: E402


def test_training_runs():
    env = GridEnv(size=4)
    agent = QLearner(env)
    agent.train(episodes=10)
    assert agent.q_table.shape == (env.n_states, env.n_actions)
    assert np.any(agent.q_table != 0)
