#!/usr/bin/env python3

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class DP:
    def __init__(self, nodes, policy):
        self.nodes = nodes
        self.policy = policy
        self.prob, self.rwd = self.initialize()

    def is_terminal(self, node):
        if node == 0 or node == self.nodes-1:
            return True
        return False

    def initialize(self):
        # nodes number of states, 2 actions
        prob = np.zeros((2, self.nodes, self.nodes))
        rwd = np.ones((2, self.nodes))
        for s in range(1, self.nodes-1):
            prob[1][s][s+1] = 1
            prob[0][s][s-1] = 1
            if self.is_terminal(s-1):
                rwd[0][s] = 0
        return prob, rwd

    def value_evaluation(self, ep=0.0001):
        v = np.zeros(self.nodes)
        delta = 1
        while delta > ep:
            delta = 0
            for s in range(1, self.nodes-1):
                old = v[s]
                tmp = self.policy[s] * (self.rwd[1][s]-1) + (1-self.policy[s]) * (self.rwd[0][s]-1)
                tmp += self.policy[s] * np.dot(self.prob[1][s], v)
                tmp += (1-self.policy[s]) * np.dot(self.prob[0][s], v)
                v[s] = tmp
                delta = max(delta, abs(tmp-old))
        return v


class Environment:
    def __init__(self, nodes, success_rwd, fail_rwd):
        self.nodes = nodes
        self.success_rwd = success_rwd
        self.fail_rwd = fail_rwd

    def is_terminal(self, curr):
        if curr == 0 or curr == self.nodes-1:
            return True
        return False

    def sample(self, curr, policy):
        # policy list of probability of going towards right
        rwd = 0
        if np.random.rand() < policy[curr]:
            curr += 1
            rwd = self.success_rwd if self.is_terminal(curr) else 0
        else:
            curr -= 1
            rwd = self.fail_rwd if self.is_terminal(curr) else 0
        return curr, rwd


class NStepTD:
    def __init__(self, env):
        self.env = env
        self.nodes = env.nodes

    def estimate(self, policy, n, alpha, vtrue, gama=1, episodes=10):
        v = np.zeros(self.nodes)
        err = 0
        for e in range(1, episodes+1):
            curr = self.nodes//2
            T = 100000 # max value
            t = 0; update_t = 0
            state_history = [curr]; rwd_history = [0]
            while update_t < T-1:
                if t < T:
                    n_curr, rwd = self.env.sample(curr, policy)
                    rwd_history.append(rwd)
                    state_history.append(n_curr)
                    if self.env.is_terminal(n_curr):
                        T = t+1
                update_t = t - n + 1
                if update_t >= 0:
                    G = 0
                    for i in range(min(t+1, T), update_t, -1):
                        G = gama*G + rwd_history[i]
                    if t+1 < T:
                        G += math.pow(gama, n) * v[state_history[t+1]]
                    v[state_history[update_t]] += alpha * (G - v[state_history[update_t]])
                t += 1; curr = n_curr
            rms_err = np.sqrt(np.sum((v-vtrue)**2) / (self.nodes-2))
            err += (rms_err - err) / e
        return err


def main():
    policy = [0] + [1/2]*19 + [0]
    nodes = len(policy)
    dp = DP(nodes, policy)
    v_true = dp.value_evaluation()

    env = Environment(nodes, 0, -1)
    n_steps = [1, 2, 4, 8, 16]
    alphas = np.linspace(0, 1, 21)
    values = np.zeros((len(n_steps), len(alphas)))
    ntd = NStepTD(env)
    for apos in range(len(alphas)):
        for npos in range(len(n_steps)):
            err = 0
            for run in range(1, 100 + 1):
                rms_err = ntd.estimate(policy, n_steps[npos], alphas[apos], v_true)
                err += (rms_err - err) / run
            values[npos, apos] = err
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, values[0], 'r-', alphas, values[1], 'g-', alphas, values[2], 'b-', alphas, values[3], 'k-', alphas, values[4], 'c-')
    plt.savefig('figure7_2_2.png')


if __name__ == '__main__':
    main()


