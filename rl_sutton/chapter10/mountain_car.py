#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm


ACTIONS = [1, -1, 0]


class Environment:
    def __init__(self):
        self.x_range = [-1.2, 0.5]
        self.x_dot_range = [-0.07, 0.07]

    def get_start_pos(self):
        low, high = -0.6, -0.4-0.00001
        return np.random.uniform(low, high), 0

    def is_goal(self, x, x_dot):
        return x == self.x_range[1]

    def is_extreme(self, x, x_dot):
        return x == self.x_range[0] or x == self.x_range[1]

    def next(self, x, x_dot, a):
        nx_dot = x_dot + 0.001*ACTIONS[a] - 0.0025*np.cos(3*x)
        nx_dot = min(max(nx_dot, self.x_dot_range[0]), self.x_dot_range[1])
        nx = min(max(x + nx_dot, self.x_range[0]), self.x_range[1])
        rwd = -1
        if self.is_extreme(nx, nx_dot):
            nx_dot = 0
        return (nx, nx_dot, rwd)


class TileCoding:
    def __init__(self, limits, tiles, tilings, num_actions, offset=lambda n: 2*np.arange(n) + 1):
        self.tiles = tiles + 1
        self.tilings = tilings
        self.limits = np.array(limits)
        self.num_actions = num_actions

        dim = len(limits)
        tile_offset = offset(dim) * np.repeat([np.arange(tilings)], dim, axis=0).T
        self.tile_offset = (tile_offset / tiles) % 1
        
        self.tile_per_len = tiles / (self.limits[:, 1] - self.limits[:, 0])

        self.base_state = np.arange(tilings) * (self.tiles * self.tiles)
        self.base_action = np.arange(num_actions) * (self.num_state_tiles())

    def to_id(self, x, a):
        x = np.array(x)
        to_tiles = ((x - self.limits[:, 0]) * self.tile_per_len + self.tile_offset).astype(int)
        pos = to_tiles[:, 0] + to_tiles[:, 1]*self.tiles + self.base_state + self.base_action[a]
        return pos

    def num_state_tiles(self):
        return self.tiles * self.tiles * self.tilings

    def num_tiles(self):
        return self.num_state_tiles() * self.num_actions


class TDn:
    def __init__(self, env, tile_code):
        self.env = env
        self.tile_code = tile_code
        self.weights = None
        self.alpha = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def reset(self):
        num_weights = self.tile_code.num_tiles()
        self.weights = np.zeros(num_weights)

    def q_estimate(self, x, x_dot, a):
        features = self.tile_code.to_id((x, x_dot), a)
        return np.sum(self.weights[features])

    def update(self, x, x_dot, a, U):
        features = self.tile_code.to_id((x, x_dot), a)
        self.weights[features] += self.alpha * (U - self.q_estimate(x, x_dot, a))

    def argmax(self, value):
        max_v = np.max(value)
        return np.random.choice([a for a, v in enumerate(value) if v == max_v])

    def ep_greedy(self, x, x_dot, ep=0.1):
        if ep > np.random.rand():
            return np.random.randint(len(ACTIONS))
        qestimates = [self.q_estimate(x, x_dot, a) for a in range(len(ACTIONS))]
        return self.argmax(qestimates)

    def n_step(self, n):
        env = self.env
        steps = 0
        history = []
        x, x_dot = env.get_start_pos()
        a = self.ep_greedy(x, x_dot)
        history.append(((x, x_dot, a), 0))
        T = 100_000
        t, tau = 0, 0
        while tau < T:
            if t < T:
                x, x_dot, rwd = env.next(x, x_dot, a)
                a = self.ep_greedy(x, x_dot)
                history.append(((x, x_dot, a), rwd))
                steps += 1
                if env.is_goal(x, x_dot):
                    T = t+1
            tau = t+1-n
            if tau >= 0:
                G = 0
                if t+1 < T:
                    s, _ = history[t+1]
                    G = self.q_estimate(*s)
                for i in range(min(t+1, T), tau, -1):
                    _, r = history[i]
                    G += r
                s, _ = history[tau]
                self.update(*s, G)
            t += 1
        return steps

    def state_value(self, xx, yy):
        m, n = xx.shape
        v = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                x, x_dot = xx[i, j], yy[i, j]
                q = np.max([self.q_estimate(x, x_dot, a) for a in range(len(ACTIONS))])
                v[i, j] = -q
        return v


def figure10_1():
    fig = plt.figure(figsize=(9,10), dpi=80)
    def plot(X, Y, Z, id, episode):
        ax = fig.add_subplot(220+id, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_title(f"{episode} episode")
        ax.set_xlabel("position")
        ax.set_ylabel("velocity")

    tiles, tilings = 8, 8
    alpha = 0.5/tilings
    num_actions = len(ACTIONS)
    n = 1

    pos, vel = (-1.199, 0.499), (-0.07, 0.07)
    env = Environment()
    tile_code = TileCoding((pos, vel), tiles, tilings, num_actions)
    tdn = TDn(env, tile_code)

    tdn.set_alpha(alpha)
    tdn.reset()

    X, Y = np.meshgrid(np.linspace(*pos, 10), np.linspace(*vel, 10))
    plot_episodes = [12, 104, 1000, 9000]
    id, episodes = 0, 9000
    for e in tqdm(range(1, episodes+1)):
        tdn.n_step(n)
        if e in plot_episodes:
            Z = tdn.state_value(X, Y)
            id += 1
            plot(X, Y, Z, id, e)

    plt.tight_layout(pad=0.4, h_pad=3.0)
    plt.savefig("figure10_1.png")


def figure10_2():
    tiles, tilings = 8, 8
    num_actions = len(ACTIONS)
    alphas = np.array([0.1, 0.2, 0.5])
    runs, episodes = 10, 500

    pos, vel = (-1.199, 0.499), (-0.07, 0.07)
    env = Environment()
    tile_code = TileCoding((pos, vel), tiles, tilings, num_actions)
    tdn = TDn(env, tile_code)

    eps = list(range(1, episodes+1))
    steps = np.zeros((len(alphas), episodes))
    for apos in range(len(alphas)):
        tdn.set_alpha(alphas[apos] / tilings)
        print(f"Runs for alpha: {alphas[apos]}")
        for r in tqdm(range(1, runs+1)):
            tdn.reset()
            for e in eps:
                s = tdn.n_step(n=1)
                steps[apos][e-1] += (s - steps[apos][e-1]) / r

    fig = plt.figure()
    ax = fig.add_subplot()
    color = ['b-', 'g-', 'r-']
    for p in range(len(alphas)):
        ax.plot(eps, steps[p], color[p], label=f"alpha={alphas[p]}/{tilings}")
    ax.set_ylim([80, 1000])
    ax.set_yscale('log')
    fig.suptitle("Mountain Car problem")
    plt.xlabel("Episode")
    plt.ylabel(f"Steps per episode avg over {runs} runs")
    plt.legend()
    plt.savefig("figure10_2.png")


def figure10_3():
    tiles, tilings = 8, 8
    num_actions = len(ACTIONS)
    runs, episodes = 50, 500
    ns = [1, 4]
    alphas = [0.5, 0.3]

    pos, vel = (-1.199, 0.499), (-0.07, 0.07)
    env = Environment()
    tile_code = TileCoding((pos, vel), tiles, tilings, num_actions)
    tdn = TDn(env, tile_code)

    eps = list(range(1, episodes+1))
    steps = np.zeros((len(ns), episodes))
    for p in range(len(ns)):
        tdn.set_alpha(alphas[p] / tilings)
        np.random.seed(0)
        print(f"Runs for n={ns[p]}")
        for r in tqdm(range(1, runs+1)):
            tdn.reset()
            for e in eps:
                s = tdn.n_step(n=ns[p])
                steps[p][e-1] += (s - steps[p][e-1]) / r

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(eps, steps[0], 'r-', label=f"n={ns[0]}")
    ax.plot(eps, steps[1], 'g-', label=f"n={ns[1]}")
    ax.set_ylim([80, 1000])
    ax.set_yscale('log')
    fig.suptitle(f"Mountain Car problem n={ns[1]} vs n={ns[0]} td")
    plt.xlabel("Episode")
    plt.ylabel(f"Steps per episode avg over {runs} runs")
    plt.legend()
    plt.savefig("figure10_3_1_4.png")


def figure10_4():
    tiles, tilings = 8, 8
    num_actions = len(ACTIONS)
    runs, episodes = 50, 50
    ns = [1, 2, 4, 8]
    alphas = np.linspace(0.1, 0.9, 9)

    pos, vel = (-1.199, 0.499), (-0.07, 0.07)
    env = Environment()
    tile_code = TileCoding((pos, vel), tiles, tilings, num_actions)
    tdn = TDn(env, tile_code)

    steps = np.zeros((len(ns), len(alphas)))

    for p in range(len(ns)):
        print(f"Runs for n={ns[p]}")
        for ap in tqdm(range(len(alphas))):
            tdn.set_alpha(alphas[ap] / tilings)
            np.random.seed(0)
            for r in range(1, runs+1):
                tdn.reset()
                s = 0
                for e in range(1, episodes+1):
                    s += (tdn.n_step(n=ns[p]) - s) / e
                steps[p][ap] += (s - steps[p][ap]) / r

    fig = plt.figure()
    ax = fig.add_subplot()

    color = ['r-', 'g-', 'b-', 'k-']
    for i in range(len(ns)):
        ax.plot(alphas, steps[i], color[i], label=f"n={ns[i]}")

    ax.set_ylim([200, 500])
    fig.suptitle(f"Mountain Car problem comparing alphas")
    plt.xlabel(f"alpha x number of tilings ({tilings})")
    plt.ylabel(f"Steps per episode avg over {runs} runs")
    plt.legend()
    plt.savefig("figure10_4.png")


if __name__ == '__main__':
    figure10_4()


