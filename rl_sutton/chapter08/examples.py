#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


ACTIONS = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}


class Environment:
    def __init__(self, grid, block=1, start=2, goal=3):
        self.grid = grid
        self.block = block
        self.start_pos, self.goal_pos = self.find_terminal(start, goal)

    def get_grid_shape(self):
        return self.grid.shape

    def find_terminal(self, start, goal):
        m, n = self.grid.shape
        start_pos, goal_pos = None, None
        for i in range(m):
            for j in range(n):
                if self.grid[i, j] == start:
                    start_pos = (i, j)
                elif self.grid[i, j] == goal:
                    goal_pos = (i, j)
        return (start_pos, goal_pos)

    def is_non_state(self, x, y):
        if self.grid[x, y] == self.block or (x, y) == self.start_pos or self.is_goal(x, y):
            return True
        return False

    def is_goal(self, x, y):
        if x == self.goal_pos[0] and y == self.goal_pos[1]:
            return True
        return False

    def sample(self, x, y, a):
        m, n = self.grid.shape
        ax, ay = ACTIONS[a]
        new_x, new_y = x+ax, y+ay
        if new_x < 0 or new_y < 0 or new_x >= m or new_y >= n or self.grid[new_x, new_y] == self.block:
            return (x, y, 0)
        rwd = 1 if self.is_goal(new_x, new_y) else 0
        return (new_x, new_y, rwd)


class Agent:
    def __init__(self, env):
        self.env = env
        self.num_of_actions = len(ACTIONS)
        self.all_actions = list(range(self.num_of_actions))
        self.policy = None
        self.q_table = None
        self.model = None

    def hash(self, x, y, a):
        m, n = self.env.get_grid_shape()
        return n*x + y + m*n*a

    def hash_to_id(self, h):
        m, n = self.env.get_grid_shape()
        a = h//(m*n)
        newh = h%(m*n)
        x, y = newh//n, newh%n
        assert h == self.hash(x, y, a)
        return (x, y, a)

    def initialize_model(self):
        model = {}
        m, n = self.env.get_grid_shape()
        for a in range(self.num_of_actions):
            for i in range(m):
                for j in range(n):
                    if self.env.is_non_state(i, j):
                        continue
                    model[self.hash(i, j, a)] = (i, j, 0, 0)
        return model

    def reset(self, is_model_init=False):
        m, n = self.env.get_grid_shape()
        self.policy = np.zeros((m ,n)).astype(int)
        self.q_table = np.zeros((self.num_of_actions, m, n))
        self.model = self.initialize_model() if is_model_init else {}

    def ep_greedy(self, x, y, ep):
        if ep > np.random.rand():
            return np.random.randint(self.num_of_actions)
        return self.policy[x, y]

    def update_policy(self, x, y):
        greedy_a = self.policy[x, y]
        greedy_q = self.q_table[greedy_a, x, y]
        np.random.shuffle(self.all_actions)
        for a in self.all_actions:
            if self.q_table[a, x, y] >= greedy_q:
                greedy_q = self.q_table[a, x, y]
                greedy_a = a
        self.policy[x, y] = greedy_a

    def update_model(self, h, nx, ny, rwd, time_passed):
        self.model[h] = (nx, ny, rwd, time_passed)

    def dyna_q_planning(self, planning_steps, gama, alpha):
        ids = list(self.model.keys())
        for i in range(planning_steps):
            h = ids[np.random.randint(len(ids))]
            x, y, a = self.hash_to_id(h)
            nx, ny, rwd, _ = self.model[h]
            nqmax = self.q_table[self.policy[nx, ny], nx, ny]
            self.q_table[a, x, y] += alpha*(rwd + gama*nqmax - self.q_table[a, x, y])
            self.update_policy(x, y)

    def dyna_q_plus_planning(self, planning_steps, time_passed, gama, alpha, k):
        ids = list(self.model.keys())
        for i in range(planning_steps):
            h = ids[np.random.randint(len(ids))]
            x, y, a = self.hash_to_id(h)
            nx, ny, rwd, time = self.model[h]
            nqmax = self.q_table[self.policy[nx, ny], nx, ny]
            # in dyna_q+ exploratory reward is used
            assert time_passed - time >= 0
            exploration_rwd = rwd + k*np.sqrt(time_passed - time)
            G = exploration_rwd + gama*nqmax
            self.q_table[a, x, y] += alpha*(G - self.q_table[a, x, y])
            self.update_policy(x, y)

    def planning_learning(self, planning_steps, episodes, gama, alpha, ep, k=0.1, is_dyna_q_plus=False):
        len_of_episode = []
        time_passed = 0
        for e in range(1, episodes+1):
            episode_time = 0
            cx, cy = self.env.start_pos
            while not self.env.is_goal(cx, cy):
                episode_time += 1; time_passed += 1
                ca = self.ep_greedy(cx, cy, ep)
                nx, ny, rwd = self.env.sample(cx, cy, ca)
                nqmax = self.q_table[self.policy[nx, ny], nx, ny]
                self.q_table[ca, cx, cy] += alpha*(rwd + gama*nqmax - self.q_table[ca, cx, cy])
                self.update_policy(cx, cy)
                self.update_model(self.hash(cx, cy, ca), nx, ny, rwd, time_passed)
                cx, cy = nx, ny
                # planning step
                if is_dyna_q_plus:
                    self.dyna_q_plus_planning(planning_steps, time_passed, gama, alpha, k)
                else:
                    self.dyna_q_planning(planning_steps, gama, alpha)
            len_of_episode.append(episode_time)
        return len_of_episode

    def dyna_q(self, planning_steps, episodes, gama=0.95, alpha=0.1, ep=0.1):
        self.reset()
        len_of_episode = self.planning_learning(planning_steps, episodes, gama, alpha, ep)
        return len_of_episode

    def dyna_q_plus(self, planning_steps, episodes, gama=0.95, alpha=0.1, ep=0.1, k=0.0001):
        self.reset(is_model_init=True)
        len_of_episode = self.planning_learning(planning_steps, episodes, gama, alpha, ep, k, True)
        return len_of_episode


def example8_4():
    grid = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 3],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0],
                     [2, 0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    env = Environment(grid)
    agent = Agent(env)
    print(agent.dyna_q_plus(5, 50))


def length(agent, planning_steps, episodes, runs):
    length = np.zeros(episodes)
    for run in range(1, runs+1):
        np.random.seed(run)
        length += (agent.dyna_q(planning_steps, episodes) - length) / run
    return length


def example8_1():
    grid = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 3],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0],
                     [2, 0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    env = Environment(grid)
    agent = Agent(env)

    runs = 30
    #print(agent.dyna_q(5, 50))
    len_of_episode_0_planning = length(agent, 0, 50, 30)
    len_of_episode_5_planning = length(agent, 5, 50, 30)
    len_of_episode_50_planning = length(agent, 50, 50, 30)
    episodes = list(range(1, 50+1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([14, 700])
    ax.plot(episodes, len_of_episode_0_planning, 'b-', label="0 planning steps")
    ax.plot(episodes, len_of_episode_5_planning, 'g-', label="5 planning steps")
    ax.plot(episodes, len_of_episode_50_planning, 'r-', label="50 planning steps")
    fig.suptitle("DynaQ agents")
    plt.xlabel("Episode")
    plt.ylabel("Steps per episode")
    plt.legend()
    plt.savefig('figure8_1.png')


if __name__ == '__main__':
    example8_4()

