#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from multiprocessing import Pool
import sys


# branching factor
B = 3

# number of states
NUM_OF_STATES = 1000

# compute time
COMPUTE_TIME = 5000

# evaluate time
EVAL_TIME = 10

# runs
RUNS = 30


class Environment:
    def __init__(self, b, num_of_states):
        self.num_of_states = num_of_states
        self.b = b

        self.num_of_actions = 2
        self.start_state, self.end_state = 0, num_of_states

        self.end_prob = 0.1

        # all equally likely transition to non-terminal state with prob = 0.9
        self.state_transition = np.random.randint(num_of_states, size=(num_of_states, self.num_of_actions, b))
        # expected rwd
        self.rwd_transition = np.random.randn(num_of_states, self.num_of_actions)

    def is_end(self, state):
        return state == self.end_state

    def sample(self, state, action):
        if self.is_end(state) or np.random.rand() < self.end_prob:
            return self.end_state, 0
        rwd = self.rwd_transition[state, action]
        next = np.random.randint(self.b)
        return self.state_transition[state, action, next], rwd


class DistributeUpdates:
    def __init__(self, env):
        self.env = env

    def argmax(self, value):
        max_v = np.max(value)
        return np.random.choice([a for a, v in enumerate(value) if v == max_v])

    def get_start_value(self, q):
        runs = 100
        start_state = self.env.start_state
        v = 0
        for run in range(1, runs):
            s = start_state
            ret = 0
            while not self.env.is_end(s):
                a = self.argmax(q[s, :])
                s, rwd = self.env.sample(s, a)
                ret += rwd
            v += (ret - v) / run
        return v

    @staticmethod
    def name():
        return "DistributeUpdates"

    def evaluate(self, compute_time, eval_time):
        raise NotImplementedError


class UniformUpdate(DistributeUpdates):
    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def name():
        return "UniformUpdate"

    def evaluate(self, compute_time, eval_time=10):
        num_of_states = self.env.num_of_states
        num_of_actions = self.env.num_of_actions
        non_terminal_prob = 1 - self.env.end_prob
        state_transition = self.env.state_transition
        rwd_transition = self.env.rwd_transition

        q = np.zeros((num_of_states, num_of_actions))
        total_pair = num_of_states * num_of_actions

        start_values = [0]

        time = 0; sa = 0
        while time < compute_time:
            s, a = sa % num_of_states, sa // num_of_states
            next_states = state_transition[s, a]
            q[s, a] = rwd_transition[s, a] + non_terminal_prob * np.mean(np.max(q[next_states, :], axis=1))
            sa = (sa+1) % total_pair; time += 1
            if time % eval_time == 0:
                start_values.append(self.get_start_value(q))
        return start_values


class TrajectorySampling(DistributeUpdates):
    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def name():
        return "TrajectorySampling"

    def ep_greedy_action(self, value, ep=0.1):
        if np.random.rand() < ep:
            return np.random.randint(len(value))
        return self.argmax(value)

    def evaluate(self, compute_time, eval_time=10):      
        num_of_states = self.env.num_of_states
        num_of_actions = self.env.num_of_actions
        non_terminal_prob = 1 - self.env.end_prob
        start_state = self.env.start_state
        state_transition = self.env.state_transition
        rwd_transition = self.env.rwd_transition

        q = np.zeros((num_of_states, num_of_actions))

        start_values = [0]
        time = 0
        while time < compute_time:
            s = start_state
            while (not self.env.is_end(s)) and (time < compute_time):
                a = self.ep_greedy_action(q[s, :])
                next_states = state_transition[s, a]
                q[s, a] = rwd_transition[s, a] + non_terminal_prob * np.mean(np.max(q[next_states, :], axis=1))
                s, _ = self.env.sample(s, a)
                time += 1
                if time % eval_time == 0:
                    start_values.append(self.get_start_value(q))
        return start_values


def run_in_process(algo_cls):
    env = Environment(B, NUM_OF_STATES)
    algo = algo_cls(env)

    return algo.evaluate(COMPUTE_TIME, EVAL_TIME)


def cal_avgs(algo_cls):
    avg_values = []
    args = [algo_cls] * RUNS
    with Pool(4) as pool:
        for i, values in enumerate(pool.imap_unordered(run_in_process, args)):
            avg_values.append(values)
            sys.stderr.write(f"\r{algo_cls.name()}: Runs completed: {i+1}")
    print()
    return np.mean(avg_values, axis=0)


def main():
    uniform_start_state = cal_avgs(UniformUpdate)
    trajectory_start_state = cal_avgs(TrajectorySampling)
    time = list(range(0, COMPUTE_TIME+1, EVAL_TIME))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time, uniform_start_state, label=f"uniform {B} branch")
    ax.plot(time, trajectory_start_state, label=f"trajectory {B} branch")
    fig.suptitle(f"Compare Trajectory Vs Uniform Sampling for {NUM_OF_STATES} states")
    plt.xlabel("Computation time, in expected updates")
    plt.ylabel("Values of start state under greedy policy")
    plt.legend()
    plt.savefig('figure8_8_3.png')


if __name__ == '__main__':
    main()





