import numpy as np

def generate_features_predator(state_dict, j):
    features = []
    predator = state_dict['predators'][j]

    x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], predator['radius'], predator['speed']

    features += [x_pred, y_pred]

    prey_list = []

    for prey in state_dict['preys']:
        x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                    prey['radius'], prey['speed'], prey['is_alive']
        angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
        distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

        prey_list += [[angle, distance, int(alive), r_prey]]

    prey_list = sorted(prey_list, key=lambda x: x[1])
    prey_list = [item for sublist in prey_list for item in sublist]
    features += prey_list

    obs_list = []

    for obs in state_dict['obstacles']:
        x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
        angle = np.arctan2(y_obs - y_pred, x_obs - x_pred) / np.pi
        distance = np.sqrt((y_obs - y_pred) ** 2 + (x_obs - x_pred) ** 2)

        obs_list += [[angle, distance, r_obs]]

    obs_list = sorted(obs_list, key=lambda x: x[1])
    obs_list = [item for sublist in obs_list for item in sublist]
    features += obs_list

    return np.array(features, dtype=np.float32)

def generate_features_prey(state_dict, i):
    features = []
    prey = state_dict['preys'][i]

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
