from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from loguru import logger
from snoop import snoop
from icecream import ic
from tqdm import trange

# Environment -----------------------------------------------------------------

@dataclass
class GridEnv:
    size: int = 4
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (3, 3)

    def reset(self) -> Tuple[int, int]:
        self.position = self.start
        logger.debug("Environment reset to {}", self.position)
        return self.position

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        x, y = self.position
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.size - 1, y + 1)
        else:
            raise ValueError("Invalid action")

        self.position = (x, y)
        reward = 1.0 if self.position == self.goal else -0.01
        done = self.position == self.goal
        logger.debug("Moved to {}, reward={}, done={}", self.position, reward, done)
        return self.position, reward, done

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


# Q-Learning ------------------------------------------------------------------

@dataclass
class QLearner:
    env: GridEnv
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        self.q_table = np.zeros((self.env.n_states, self.env.n_actions), dtype=np.float32)
        logger.info("Q-table initialized with shape {}", self.q_table.shape)

    def choose_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.env.n_actions)
            ic("random action", action)
            return action
        action = int(np.argmax(self.q_table[state]))
        ic("greedy action", action)
        return action

    def state_index(self, position: Tuple[int, int]) -> int:
        return position[0] * self.env.size + position[1]

    @snoop
    def train(self, episodes: int = 100) -> None:
        for _ in trange(episodes, desc="Training"):
            state = self.state_index(self.env.reset())
            done = False
            while not done:
                action = self.choose_action(state)
                pos, reward, done = self.env.step(action)
                next_state = self.state_index(pos)
                best_next = np.max(self.q_table[next_state])
                target = reward + self.gamma * best_next
                self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
                state = next_state

    def save(self, path: str) -> None:
        temp_fd, temp_path = tempfile.mkstemp()
        os.close(temp_fd)
        try:
            np.save(temp_path, self.q_table)
            os.replace(temp_path, path)  # atomic write
            logger.info("Q-table saved to {}", path)
        except Exception as exc:
            logger.exception("Failed to save Q-table: {}", exc)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def load(self, path: str) -> None:
        try:
            self.q_table = np.load(path)
            logger.info("Q-table loaded from {}", path)
        except Exception as exc:
            logger.exception("Failed to load Q-table: {}", exc)
            raise

if __name__ == "__main__":
    env = GridEnv(size=4)
    agent = QLearner(env)
    try:
        agent.train(episodes=100)
        agent.save("q_table.npy")
    except Exception as exc:
        logger.exception("Training failed: {}", exc)
