#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


class DP:
    def __init__(self, nodes, policy):
        self.nodes = nodes
        self.policy = policy
        self.prob, self.rwd = self.initialize()

    def is_terminal(self, curr):
        if curr == 0 or curr == self.nodes-1:
            return True
        return False

    def initialize(self):
        prob = np.zeros((2, self.nodes, self.nodes))
        rwd = np.zeros((2, self.nodes))
        for s in range(1, self.nodes-1):
            prob[1][s][s+1] = 1
            prob[0][s][s-1] = 1
            if self.is_terminal(s+1):
                rwd[1][s] = 1
        return prob, rwd

    def value_prediction(self, ep=0.001):
        v = np.zeros(self.nodes)
        delta = 1
        while delta > ep:
            delta = 0
            for s in range(self.nodes):
                old = v[s]
                tmp = self.policy[s] * self.rwd[1][s] + (1-self.policy[s]) * self.rwd[0][s]
                tmp += self.policy[s] * np.dot(self.prob[1][s], v)
                tmp += (1-self.policy[s]) * np.dot(self.prob[0][s], v)
                v[s] = tmp
                delta = max(delta, abs(old-v[s]))
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

    def choose_action(self, curr, policy):
        if np.random.rand() < policy[curr]:
            return 1
        return -1

    def sample(self, curr, a):
        rwd = 0; curr += a
        if curr == 0:
            rwd = self.fail_rwd
        elif curr == self.nodes-1:
            rwd = self.success_rwd
        return curr, rwd


class OffPolicyMethod:
    def __init__(self, env, offpolicy):
        self.env = env
        self.offpolicy = offpolicy
        self.nodes = env.nodes

    def estimate_using_per_instance(self, policy, n, alpha, gama=1, episodes=10):
        v = np.zeros(self.nodes)
        for ep in range(1, episodes+1):
            T = 10000 # max value
            t = 0; update_t = 0
            curr = self.nodes//2; off_action = self.env.choose_action(curr, self.offpolicy)
            state_history = [curr]; action_history = [off_action]; rwd_history = [0]
            while update_t < T:
                if t < T:
                    next_state, next_rwd = self.env.sample(curr, off_action)
                    state_history.append(next_state); rwd_history.append(next_rwd)
                    if self.env.is_terminal(next_state):
                        T = t+1
                    else:
                        next_action = self.env.choose_action(next_state, self.offpolicy)
                        action_history.append(next_action)
                update_t = t-n+1
                if update_t >= 0:
                    G = v[state_history[t+1]] if t+1 < T else 0
                    for i in range(min(t+1, T), update_t, -1):
                        rwd = rwd_history[i]
                        a = action_history[i-1]; s = state_history[i-1]
                        if a == -1:
                            r = (1-policy[s]) / (1-self.offpolicy[s])
                        else: # a == 1
                            assert a == 1
                            r = policy[s] / self.offpolicy[s]
                        G = r*(rwd + gama*G) + (1-r) * v[s]
                    v[state_history[update_t]] += alpha*(G - v[state_history[update_t]])
                curr = next_state; off_action = next_action
                t += 1
        return v

    def estimate_using_simple_sampling(self, policy, n, alpha, gama=1, episodes=100):
        v = np.zeros(self.nodes)
        for ep in range(1, episodes+1):
            T = 10000 # max value
            t = 0; update_t = 0
            curr = self.nodes//2; off_action = self.env.choose_action(curr, self.offpolicy)
            state_history = [curr]; action_history = [off_action]; rwd_history = [0]
            while update_t < T:
                if t < T:
                    next_state, next_rwd = self.env.sample(curr, off_action)
                    state_history.append(next_state); rwd_history.append(next_rwd)
                    if self.env.is_terminal(next_state):
                        T = t+1
                    else:
                        next_action = self.env.choose_action(curr, self.offpolicy)
                        action_history.append(next_action)
                update_t = t-n+1
                if update_t >= 0:
                    G = v[state_history[t+1]] if t+1 < T else 0
                    r = 1
                    for i in range(min(t+1, T), update_t, -1):
                        a = action_history[i-1]; s = state_history[i-1]
                        if a == -1:
                            r *= (1-policy[s]) / (1 - self.offpolicy[s])
                        else: # a == 1
                            assert a == 1
                            r *= policy[s] / self.offpolicy[s]
                        G = gama*G + rwd_history[i]
                    v[state_history[update_t]] += alpha*(r*G - v[state_history[update_t]])
                    val = v[state_history[update_t]]
                    v[state_history[update_t]] = min(val, 1e5)
                curr = next_state; off_action = next_action
                t += 1
        return v


def save_figure(x, n_steps, v1, v2):
    lines = ['-', '--', '-.', 'o-', 'v-']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([0, 2])
    for i in range(len(n_steps)):
        ax.plot(x, v1[i], 'r'+lines[i], label=f"per_inst-{n_steps[i]}")
        ax.plot(x, v2[i], 'b'+lines[i], label=f"sample-{n_steps[i]}")
    fig.suptitle('Off Policy: PerInstance Vs Simple Sampling Method')
    plt.xlabel('alpha')
    plt.ylabel('Avg RMS error over 19 states and first 10 episodes')
    plt.legend()
    plt.savefig('figure7_10.png')


def main():
    nodes = 5
    offpolicy = [0] + [1/2]*nodes + [0]
    policy = [0] + [1]*nodes + [0]
    total_nodes = len(policy)

    dp = DP(total_nodes, policy)
    vtrue = dp.value_prediction()

    alphas = np.linspace(0, 0.5, 6)
    n_steps = [1, 2, 4, 8, 16]
    v_ss = np.zeros((len(n_steps), len(alphas)))
    v_pi = np.zeros((len(n_steps), len(alphas)))

    env = Environment(total_nodes, 1, 0)
    off_method = OffPolicyMethod(env, offpolicy)

    for npos in range(0, len(n_steps)):
        for apos in range(0, len(alphas)):
            step, alpha = n_steps[npos], alphas[apos]
            err_per_instance, err_simple_sample = 0, 0
            for run in range(1, 100+1):
                v_per_instance = off_method.estimate_using_per_instance(policy, step, alpha)
                v_simple_sample = off_method.estimate_using_simple_sampling(policy, step, alpha)
                rms_per_instance = np.sqrt(np.sum((vtrue-v_per_instance)**2) / nodes)
                rms_simple_sample = np.sqrt(np.sum((vtrue-v_simple_sample)**2) / nodes)
                err_per_instance += (rms_per_instance - err_per_instance) / run
                err_simple_sample += (rms_simple_sample - err_simple_sample) / run
            v_pi[npos, apos] = err_per_instance
            v_ss[npos, apos] = err_simple_sample


    save_figure(alphas, n_steps, v_pi, v_ss)


if __name__ == '__main__':
    main()






