#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


TOTAL_STATES = 1000

ACTIONS = [-1, 1]
ACTION_STEP_IN_ONE_DIR = 100


class Environment:
    def __init__(self):
        self.start_state = 500
        self.l_terminal, self.r_terminal = 0, TOTAL_STATES + 1

    def is_terminal(self, state):
        if state == self.l_terminal or state == self.r_terminal:
            return True
        return False

    def sample(self, state, action):
        new_state = max(min(state + action, self.r_terminal), self.l_terminal)
        rwd = 0
        if new_state == self.r_terminal:
            rwd = 1
        elif new_state == self.l_terminal:
            rwd = -1
        return new_state, rwd


class LinearApprox:
    def __init__(self, order, env):
        self.order = order
        self.weight = None
        self.env = env

    def reset(self):
        self.weight = np.random.rand(self.order)

    def estimate(self, X):
        raise NotImplementedError

    def update(self, G, X):
        raise NotImplementedError

    def state_estimates(self):
        return [self.estimate(s) for s in range(1, TOTAL_STATES+1)]

    def policy(self, state):
        action = -1 if np.random.rand() < 0.5 else 1
        step = action * np.random.randint(1, ACTION_STEP_IN_ONE_DIR+1)
        return step

    def state_distribution(self, episodes=100_000):
        env = self.env
        distribution = np.zeros(TOTAL_STATES)
        for e in tqdm(range(episodes)):
            s = env.start_state
            while not env.is_terminal(s):
                distribution[s-1] += 1
                a = self.policy(s)
                s, _ = env.sample(s, a)
        distribution /= np.sum(distribution)
        return distribution

    def monte_carlo(self):
        env = self.env
        s_his = []
        s, r = env.start_state, 0
        while not env.is_terminal(s):
            s_his.append(s)
            a = self.policy(s)
            s, r = env.sample(s, a)
        for s in reversed(s_his):
            G = r
            self.update(G, s)

    def tdn(self, n):
        env = self.env
        s_his = []
        s, r = env.start_state, 0
        s_his.append((s, r))
        T = 100_000
        t, tau = 0, 0
        while tau < T-1:
            if t < T:
                a = self.policy(s)
                s, r = env.sample(s, a)
                if env.is_terminal(s):
                    T = t+1
                s_his.append((s, r))
            tau = t+1-n
            if tau >= 0:
                G = 0 if t+1 >= T else self.estimate(s_his[t+1][0])
                for i in range(min(t+1, T), tau, -1):
                    G += s_his[i][1]
                self.update(G, s_his[tau][0])
            t += 1


class StateAggregation(LinearApprox):
    def __init__(self, groupby, env, alpha=2*1e-5):
        self.alpha = alpha
        self.groupby = groupby
        num_of_weights = TOTAL_STATES // groupby
        super().__init__(num_of_weights, env)

    def estimate(self, X):
        return self.weight[(X-1) // self.groupby]

    def update(self, G, X):
        indx = (X-1) // self.groupby
        # gradient is 1
        self.weight[indx] += self.alpha * (G - self.estimate(X))


class FeatureSpace(LinearApprox):
    def __init__(self, env, order, alpha):
        super().__init__(order, env)
        self.alpha = alpha
        self.feature = None

    def set_order(self, order):
        self.order = order

    def get_feature_func(self):
        raise NotImplementedError

    def reset(self):
        self.weight = np.zeros(self.order)
        to_feature = self.get_feature_func()
        self.feature = np.array([to_feature(s) for s in range(1, TOTAL_STATES+1)])

    def estimate(self, s):
        X = self.feature[s-1]
        return np.dot(self.weight, X)

    def update(self, G, s):
        X = self.feature[s-1]
        self.weight += self.alpha * (G - self.estimate(s)) * X


class Polynomial(FeatureSpace):
    def __init__(self, env, order=0, alpha=1e-4):
        super().__init__(env, order, alpha)

    def get_feature_func(self):
        return lambda s: [(s/TOTAL_STATES)**i for i in range(self.order+1)]


class Fourier(FeatureSpace):
    def __init__(self, env, order=0, alpha=5*1e-5):
        super().__init__(env, order, alpha)

    def get_feature_func(self):
        return lambda s: [np.cos(np.pi * i * (s/TOTAL_STATES)) for i in range(self.order+1)]


class TileCoding(FeatureSpace):
    def __init__(self, env, tiles, tilings, offset=4, alpha=0.0001):
        self.tiles, self.tilings = tiles + 1, tilings
        self.width = TOTAL_STATES // tiles
        self.offset = offset
        self.alpha = alpha
        num_of_weights = self.tiles * self.tilings
        super().__init__(env, num_of_weights, alpha/tilings)
   
    def get_feature_func(self):
        def func(s):
            width, offset = self.width, self.offset 
            X = []
            for t in range(self.tilings):
                f = (s-1 + offset*t) // width
                X.append(f)
            return X
        return func

    def estimate(self, s):
        X = self.feature[s-1]
        return np.sum(self.weight[X])

    def update(self, G, s):
        X = self.feature[s-1]
        self.weight[X] += self.alpha * (G - self.estimate(s))


def true_value(env, theta=1e-2):
    # l_terminal v = -1 and r_terminal v = 1
    # initialize v
    v = np.linspace(-1, 1, TOTAL_STATES+2)
    v[0] = 0; v[-1] = 0
    delta = 1
    while delta > theta:
        delta = 0
        for s in range(1, TOTAL_STATES+1):
            old = v[s]; v[s] = 0
            for a in ACTIONS:
                for step in range(1, ACTION_STEP_IN_ONE_DIR+1):
                    ns, r = env.sample(s, a*step)
                    # equally likely policy
                    v[s] += 1/(2*ACTION_STEP_IN_ONE_DIR) * (r + v[ns])
            delta = max(delta, abs(v[s]-old))
    return v[1:-1]


def rms_ve(v_true, v_esti, s_distribution):
    err = (v_true - v_esti) ** 2
    return np.sqrt(np.sum(err * s_distribution))


def figure9_1():
    env = Environment()

    states = list(range(1, TOTAL_STATES+1))
    v_true = true_value(env)

    episodes = 100_000
    lapx = StateAggregation(100, env)
    lapx.reset()
    for e in tqdm(range(episodes)):
        lapx.monte_carlo()
    v_monte = lapx.state_estimates()
    distribution = lapx.state_distribution()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(states, v_true, label="true value estimate")
    ax[0].plot(states, v_monte, label="monte carlo estimate")
    ax[0].set_title("Value estimate using state aggregation")
    ax[0].set_xlabel("State")
    ax[0].set_ylabel("Value scale")

    ax[1].plot(states, s_distribution, label="state distribution")
    ax[1].set_title("state distribution")
    ax[1].set_xlabel("State")
    ax[1].set_ylabel("probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure9_1.png")


def figure9_2():
    env = Environment()

    states = list(range(1, TOTAL_STATES+1))
    v_true = true_value(env)

    episodes = 100_000
    lapx = StateAggregation(100, env, alpha=2*1e-3)
    lapx.reset()
    for e in tqdm(range(episodes)):
        lapx.tdn(1)
    v_td0 = lapx.state_estimates()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(states, v_true, label="true value estimate")
    ax.plot(states, v_td0, label="td0 estimate")
    fig.suptitle("Asymptotic Value Estimate using state aggregation")
    plt.xlabel("State")
    plt.ylabel("Value scale")
    plt.legend()
    plt.savefig("figure9_2.png")

def figure9_5():
    # 5th order has 6 values [0, 1, 2, 3, 4, 5]
    orders = [5+1]
    runs, episodes = 30, 5_000

    env = Environment()
    v_true = true_value(env)

    papx, fapx = Polynomial(env), Fourier(env)
    distribution = papx.state_distribution(episodes=100_000)

    eps = list(range(1, episodes+1))
    pve, fve = np.zeros((len(orders), episodes)), np.zeros((len(orders), episodes))
    for o in range(len(orders)):
        papx.set_order(orders[o])
        fapx.set_order(orders[o])
        for r in tqdm(range(1, runs+1)):
            papx.reset()
            fapx.reset()
            for e in eps:
                # Polynomial Approximation
                papx.monte_carlo()
                pv_monte = papx.state_estimates()
                perr = rms_ve(v_true, pv_monte, distribution)
                pve[o][e-1] += (perr - pve[o][e-1]) / r
                # Fourier Approximation
                fapx.monte_carlo()
                fv_monte = fapx.state_estimates()
                ferr = rms_ve(v_true, fv_monte, distribution)
                fve[o][e-1] += (ferr - fve[o][e-1]) / r

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(orders)):
        ax.plot(eps, pve[i], label=f"Polynomial {orders[i]}th order")
    for i in range(len(orders)):
        ax.plot(eps, fve[i], label=f"Fourier {orders[i]}th order")
    plt.xlabel("Episode")
    plt.ylabel("sqrt of ve")
    plt.legend()
    plt.savefig("figure9_5.png")


def figure9_10():
    tiles = 5
    runs, episodes = 30, 5_000

    env = Environment()
    v_true = true_value(env)

    tapx1 = TileCoding(env, tiles, 1)
    distribution = tapx1.state_distribution(episodes=100_000)

    eps = list(range(1, episodes+1))
    tve1 = np.zeros(episodes)
    for r in tqdm(range(1, runs+1)):
        tapx1.reset()
        for e in eps:
            tapx1.monte_carlo()
            v_monte1 = tapx1.state_estimates()
            err1 = rms_ve(v_true, v_monte1, distribution)
            tve1[e-1] += (err1 - tve1[e-1]) / r

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(eps, tve1, label=f"TileCoding 1 tile")
    plt.xlabel("Episode")
    plt.ylabel("sqrt of ve")
    plt.legend()
    plt.savefig("figure9_10.png")


if __name__ == '__main__':
    figure9_10()


