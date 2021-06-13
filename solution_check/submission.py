import numpy as np

class ClosestPath:
    def __init__(self, state, number=100):
        x_limit = 9
        y_limit = 9

        predator_radius = 1

        x_step = 2 * x_limit / number
        y_step = 2 * y_limit / number

        x_coords = np.arange(-x_limit + x_step, x_limit, x_step)
        y_coords = np.arange(-y_limit + y_step, y_limit, y_step)

        self.field_grid = np.array(np.meshgrid(x_coords, y_coords)).T
        self.field_obsticle = np.zeros(self.field_grid.shape[:2])

        ### Bounding Check

        for i in range(self.field_grid.shape[0]):
            for j in range(self.field_grid.shape[1]):
                x, y = self.field_grid[i][j]

                if x + predator_radius > x_limit or x - predator_radius < -x_limit or \
                        y + predator_radius > y_limit or y - predator_radius < -y_limit:
                    self.field_obsticle[i][j] = 1

                for obs in state['obstacles']:
                    x_obs, y_obs, rad_obs = obs['x_pos'], obs['y_pos'], obs['radius']

                    if np.sqrt((x - x_obs) ** 2 + (y - y_obs) ** 2) < predator_radius + rad_obs:
                        self.field_obsticle[i][j] = 1

        ### Adjacency Matrix

        self.height = self.field_grid.shape[0]
        self.width = self.field_grid.shape[1]
        self.N = self.height * self.width

        self.adj_matrix = []

        def get_distance(i, j, vert):
            if i >= 0 and i < self.height and \
                    j >= 0 and j < self.width:

                if 1 - self.field_obsticle[i][j]:
                    self.adj_matrix[vert] += [i * self.width + j]

        for vert in range(self.N):
            self.adj_matrix.append([])
            i, j = vert // self.height, vert % self.height

            if self.field_obsticle[i][j] == 0:
                get_distance(i + 1, j, vert)
                get_distance(i - 1, j, vert)
                get_distance(i, j + 1, vert)
                get_distance(i, j - 1, vert)

    def find_path(self, start, end, path=[]):
        path = path + [start]

        if start == end:
            return path

        if len(path) > 10:
            return None

        for node in self.adj_matrix[start]:
            if node not in path:
                newpath = self.find_path(node, end, path)

                if newpath:
                    return newpath

        return None

    def find_closest_vertex(self, x, y):
        min_dist = 10000
        min_dot = None

        for i in range(self.field_grid.shape[0]):
            for j in range(self.field_grid.shape[1]):
                x_grid, y_grid = self.field_grid[i][j]
                distance = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)

                if distance < min_dist and (1 - self.field_obsticle[i][j]):
                    min_dist = distance
                    min_dot = (i, j)

        return min_dot[0] * self.width + min_dot[1]

    def find_predator_paths(self, state, predator):
        # predator_paths = []
        # for predator in state['predators']:
        x, y = predator['x_pos'], predator['y_pos']

        pred_vert = self.find_closest_vertex(x, y)

        min_distance = 1000000
        min_path = None
        for prey in state['preys']:
            x, y = prey['x_pos'], prey['y_pos']

            if prey['is_alive']:
                prey_vert = self.find_closest_vertex(x, y)
                path = self.find_path(pred_vert, prey_vert)

                if path is not None:
                    path = [self.field_grid[x // self.width][x % self.width] for x in path]
                    if len(path) < min_distance:
                        min_distance = len(path)
                        min_path = path

        return min_path
        # predator_paths.append(min_path)

        # return predator_paths
        
def distance(first, second):
    return ((first["x_pos"] - second["x_pos"]) ** 2 + (first["y_pos"] - second["y_pos"]) ** 2) ** 0.5


def usual_distance(x_1, y_1, x_2, y_2):
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


EPS = 0.02
BACK_DELTA = 10
RANDOM_DELTA = 20
DELTA = 30
SPLIT_NUMBER = 10


class PredatorAgent:
    def __init__(self):
        self.priv_state = None
        self.random_delay = [-1] * 2
        self.pather = None
        self.paths = [None] * 2
        self.random_angle = [[-1] * BACK_DELTA + [-1] * RANDOM_DELTA] * 2

    def find_angle(self, x, y, j):
        while len(self.paths[j]) > 0 and usual_distance(self.paths[j][0][0], self.paths[j][0][1], x, y) < 1:
            self.paths[j].pop(0)

        if len(self.paths[j]) == 0:
            print('zeroing')
            return 0

        angle_value = np.arctan2(self.paths[j][0][1] - y, self.paths[j][0][0] - x) / np.pi
        return angle_value

    def act(self, state_dict):
        action = []

        if self.pather is None:
            self.pather = ClosestPath(state_dict, number=SPLIT_NUMBER)

        for j, predator in enumerate(state_dict["predators"]):
            closest_prey = None
            angle_value = 0.

            if self.random_delay[j] < 0:
                self.paths[j] = None

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

                        self.paths[j] = self.pather.find_predator_paths(state_dict, predator)
                        if self.paths[j] is not None:
                            print('yeah found')
                            self.random_delay[j] = DELTA
                            angle_value = self.find_angle(predator['x_pos'], predator['y_pos'], j)
                            print(j, angle_value)
                        else:
                            print('fck go back')
                            self.random_delay[j] = RANDOM_DELTA + BACK_DELTA
                            self.random_angle[j] = [1 + angle_value] * BACK_DELTA + [np.random.uniform(-1, 1)] * RANDOM_DELTA

            else:
                self.random_delay[j] -= 1
                if self.paths[j] is not None:
                    angle_value = self.find_angle(predator['x_pos'], predator['y_pos'], j)
                    print(j, angle_value)
                else:
                    angle_value = self.random_angle[j][self.random_delay[j]]

            action.append(angle_value)

        self.priv_state = state_dict
        return action


class PreyAgent:
    def act(self, state_dict):
        action = []
        for prey in state_dict["preys"]:
            closest_predator = None
            for predator in state_dict["predators"]:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if distance(closest_predator, prey) > distance(prey, predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.)
            else:
                action.append(1 + np.arctan2(closest_predator["y_pos"] - prey["y_pos"],
                                             closest_predator["x_pos"] - prey["x_pos"]) / np.pi)
        return action
