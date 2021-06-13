import numpy as np
import numpy as np
import torch
from torch import nn, Tensor
import math
import os

DEVICE = 'cpu'


def generate_features_prey(state_dict):
    features = []

    for prey in state_dict['preys']:
        x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                    prey['radius'], prey['speed'], prey['is_alive']

        features += [x_prey, y_prey, alive, r_prey]

        pred_list = []

        for predator in state_dict['predators']:
            x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], predator['radius'], predator[
                'speed']

            angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
            distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

            pred_list += [[angle, distance, int(alive), r_prey]]

        pred_list = sorted(pred_list, key=lambda x: x[1])
        pred_list = [item for sublist in pred_list for item in sublist]
        features += pred_list

        obs_list = []

        for obs in state_dict['obstacles']:
            x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
            angle = np.arctan2(y_obs - y_prey, x_obs - x_prey) / np.pi
            distance = np.sqrt((y_obs - y_prey) ** 2 + (x_obs - x_prey) ** 2)

            obs_list += [[angle, distance, r_obs]]

        obs_list = sorted(obs_list, key=lambda x: x[1])
        obs_list = [item for sublist in obs_list for item in sublist]
        features += obs_list

    return np.array(features, dtype=np.float32)


def calc_distance(first, second):
    return ((first["x_pos"] - second["x_pos"]) ** 2 + (first["y_pos"] - second["y_pos"]) ** 2) ** 0.5


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 64, norm_in: bool = True):
        super().__init__()
        action_dim = action_dim * 2
        if norm_in:
            self.in_fn = nn.BatchNorm1d(state_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.nonlin = nn.ReLU()
        self.out_fn = nn.Tanh()

    def forward(self, states: Tensor) -> Tensor:
        batch_size, _ = states.shape
        h1 = self.nonlin(self.fc1(states))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))

        norms = np.zeros((out.shape[0], out.shape[1] // 2))
        for i in range(0, out.shape[1], 2):
            angle = torch.atan2(*out[:, i:i + 2].T)
            normalized = (angle / math.pi).view(batch_size, -1).cpu().detach().numpy().reshape(1, -1)
            norms[:, i // 2] = normalized

        norms = torch.Tensor(norms)
        return norms

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ELU(),
#             nn.Linear(256, 256),
#             nn.ELU(),
#             nn.Linear(256, action_dim),
#             nn.Tanh()
#         )
#
#     def forward(self, state):
#         return self.model(state)

def distance(first, second):
    return ((first["x_pos"] - second["x_pos"]) ** 2 + (first["y_pos"] - second["y_pos"]) ** 2) ** 0.5

class PreyAgent:
    def __init__(self):
        self.model = Actor(210, 5).to(torch.device(DEVICE))
        state_dict = torch.load('agent.pkl')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            features = generate_features_prey(state)
            features = torch.tensor(np.array([features]), dtype=torch.float, device=DEVICE)
            action = self.model(features).cpu().numpy()[0]

            return action

EPS = 0.01
DELTA = 10
BACK_DELTA = 10

class PredatorAgent:
    def __init__(self):
        self.priv_state = None
        self.random_delay = [-1] * 3
        self.random_angle = [[-1] * BACK_DELTA + [-1] * DELTA] * 3

    def act(self, state_dict):
        action = []

        for j, predator in enumerate(state_dict["predators"]):
            closest_prey = None
            angle_value = 0.

            if self.random_delay[j] < 0:
                for i, prey in enumerate(state_dict["preys"]):
                    if not prey["is_alive"]:
                        continue
                    if closest_prey is None:
                        closest_prey = prey
                    else:
                        if distance(closest_prey, predator) > distance(prey, predator):
                            closest_prey = prey

                if closest_prey is not None:
                    angle_value = np.arctan2(closest_prey["y_pos"] - predator["y_pos"],
                                             closest_prey["x_pos"] - predator["x_pos"]) / np.pi

                if self.priv_state is not None:
                    priv_predator = self.priv_state["predators"][j]

                    if abs(priv_predator["y_pos"] - predator["y_pos"]) + abs(
                            priv_predator["x_pos"] - predator["x_pos"]) < EPS:
                        self.random_delay[j] = DELTA + BACK_DELTA
                        self.random_angle[j] = [1 + angle_value] * BACK_DELTA + [np.random.uniform(-1, 1)] * DELTA
            else:
                self.random_delay[j] -= 1
                angle_value = self.random_angle[j][self.random_delay[j]]

            action.append(angle_value)

        self.priv_state = state_dict
        return action
