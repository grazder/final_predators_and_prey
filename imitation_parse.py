import json
import sys
import os

import numpy as np
from imitation.features import generate_features_prey, generate_features_predator
from imitation.agents import PredatorAgent, PreyAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from tqdm import tqdm


class Buffer:
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))

        self.filled_i = 0

    def clear(self):
        self.states = np.zeros((self.buffer_size, self.state_dim))
        self.actions = np.zeros((self.buffer_size, self.action_dim))
        self.filled_i = 0

    def push(self, state: np.ndarray, action: np.ndarray):
        if self.filled_i >= self.buffer_size:
            return

        self.states[self.filled_i] = state
        self.actions[self.filled_i] = action
        self.filled_i += 1

    def full(self) -> bool:
        return self.filled_i == self.buffer_size

    def save(self, name: str, dir: str):
        np.save(os.path.join(dir, f'{name}:actions.npy'), self.actions)
        np.save(os.path.join(dir, f'{name}:states.npy'), self.states)

SAVE_SIZE = 100_000
PREY_FEATURE_SIZE = 210
PREDATOR_FEATURE_SIZE = 104

if __name__ == '__main__':
    env = PredatorsAndPreysEnv(render=False)
    prey_buffer = Buffer(SAVE_SIZE, PREY_FEATURE_SIZE, 1)
    predator_buffer = Buffer(SAVE_SIZE, PREDATOR_FEATURE_SIZE, 1)

    predator_agent = PredatorAgent()
    prey_agent = PreyAgent()

    done = True
    transitions = 0
    while True:
        if done:
            state_dict = env.reset()

        predator_action = predator_agent.act(state_dict)
        prey_action = prey_agent.act(state_dict)

        if transitions % 5 == 0:
            for i, prey in enumerate(state_dict["preys"]):
                prey_buffer.push(generate_features_prey(state_dict), prey_action[i])

            for j, predator in enumerate(state_dict["predators"]):
                predator_buffer.push(generate_features_predator(state_dict), predator_action[j])

        state_dict, reward, done = env.step(predator_action, prey_action)
        transitions += 1

        if predator_buffer.full():
            predator_buffer.save(f"{transitions}", "imitation/predator")
            predator_buffer.clear()
            print(f"Save predator buffer at transition {transitions}")

        if prey_buffer.full():
            prey_buffer.save(f"{transitions}", "imitation/prey")
            prey_buffer.clear()
            print(f"Save prey buffer at transition {transitions}")

