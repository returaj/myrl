#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm


PRIORITY = [1, 2, 4, 8]

ACTIONS = [0, 1] # Reject: 0 and Accept: 1


class Environment:
    def __init__(self):
        self.num_servers = 10
        self.prob_free = 0.06

    def start_state(self):
        p = np.random.randint(len(PRIORITY))
        return p, self.num_servers

    def next(self, priority, free_servers, action):
        busy_servers = self.num_servers - free_servers
        for b in range(busy_servers):
            if self.prob_free > np.random.rand():
                free_servers += 1
        rwd = 0
        if free_servers > 0 and action == 1:
            rwd = PRIORITY[priority]
            free_servers -= 1
        p = np.random.randint(len(PRIORITY))
        return p, free_servers, rwd


class Sarsa:
    def __init__(self, env, alpha=0.01, beta=0.01, ep=0.1):
        self.env = env

        num_priority, num_server_states = len(PRIORITY), env.num_servers + 1
        num_actions = len(ACTIONS)
        self.Q = np.zeros((num_actions, num_priority, num_server_states))
        self.R = 0

        self.alpha, self.beta = alpha, beta
        self.ep = ep

    def update(self, p, f, A, U):
        delta = U - self.Q[A, p, f]
        self.R += self.beta * delta
        self.Q[A, p, f] += self.alpha * delta

    def argmax(self, value):
        max_v = np.max(value)
        return np.random.choice([a for a, v in enumerate(value) if v == max_v])

    def ep_greedy(self, p, f):
        if self.ep > np.random.rand():
            return np.random.choice(ACTIONS)
        estimates = [self.Q[a, p, f] for a in ACTIONS]
        return self.argmax(estimates)

    def semi_gradient(self, time_length):
        env = self.env
        p, f = env.start_state()
        a = self.ep_greedy(p, f)
        for t in tqdm(range(time_length)):
            np, nf, r = env.next(p, f, a)
            na = self.ep_greedy(np, nf)
            G = r - self.R + self.Q[na, np, nf]
            self.update(p, f, a, G)
            p, f, a = np, nf, na

    def get_policy_value(self):
        num_priority, num_server_states = len(PRIORITY), self.env.num_servers + 1
        policy = np.zeros((num_priority, num_server_states), dtype=int)
        value = np.zeros((num_priority, num_server_states))
        for p in range(num_priority):
            for f in range(num_server_states):
                estimates = [self.Q[a, p, f] for a in ACTIONS]
                policy[p, f] = self.argmax(estimates)
                value[p, f] = np.max(estimates)
        return policy, value


def figure10_5():
    env = Environment()
    sarsa = Sarsa(env)
 
    time_length = 2_000_000
    sarsa.semi_gradient(time_length)
    policy, value = sarsa.get_policy_value()

    print(policy)

    servers = list(range(0, env.num_servers+1))

    fig = plt.figure()
    ax = fig.add_subplot()

    color = ['r-', 'g-', 'b-', 'k-']
    for i in range(len(PRIORITY)):
        ax.plot(servers, value[i], color[i], label=f"priority {PRIORITY[i]}")

    plt.xlabel("number of free servers")
    plt.ylabel("differential value of best action")
    plt.legend()
    plt.savefig("figure10_5.png")


if __name__ == '__main__':
    figure10_5()


