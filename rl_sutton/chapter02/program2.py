#! /usr/bin/env python3

from multiprocessing import Pool
import numpy as np
import math
from time import time


class NonStationaryTestBed:
    def __init__(self, k):
        self.k = k
        self.arms = None

    def number_of_arms(self):
        return self.k

    def set_arms(self):
        self.arms = [[np.random.normal(0, 1), 1] for i in range(self.k)]

    def find_optimal_arm(self):
        opti = 0
        for a in range(self.k):
            if self.arms[a][0] > self.arms[opti][0]:
                opti = a
        return opti

    def update_arms(self, sigma=0.01):
        for a in self.arms:
            a[0] += np.random.normal(0, sigma)

    def sample_reward(self, a):
        self.update_arms()
        mu, sigma = self.arms[a]
        return np.random.normal(mu, sigma)


class Runner:
    def __init__(self, horizon, runs=2000):
        self.horizon = horizon
        self.runs = runs
        self.parameters = [math.pow(2, i) for i in range(-7, 3)]
        self.param_rwd_pair = None

    def execute(self, cls, param, k):
        testbed = NonStationaryTestBed(k)
        algo = cls(testbed, horizon=self.horizon)
        reward = algo.run_instance(param)
        return reward

    def run_with_params(self, cls, k=10):
        start = time()
        self.param_rwd_pair = []
        with Pool(4) as pool:
            for p in self.parameters:
                fun_args = [(cls, p, k)] * self.runs
                rewards = pool.starmap(self.execute, fun_args)
                assert len(rewards) == self.runs
                avg_rwd = sum(rewards) / self.runs
                self.param_rwd_pair.append((p, avg_rwd))
        end = time()
        print(f"Total time taken : {end-start} sec")

    def save(self, filepath):
        with open(filepath, 'w') as fp:
            for k, v in self.param_rwd_pair:
                fp.write(f"{k}, {v}\n")


class BanditAlgo:
    def __init__(self, testbed, horizon=1000):
        self.testbed = testbed
        self.horizon = horizon

    def reset(self, param):
        raise NotImplementedError

    def select_arm(self, param):
        raise NotImplementedError

    def update_epectation(self, param, arm, reward):
        raise NotImplementedError

    def run_instance(self, param):
        self.reset(param)
        inst_rwd = 0
        for h in range(1, self.horizon+1):
            a = self.select_arm(param)
            rwd = self.testbed.sample_reward(a)
            self.update_expectation(param, a, rwd)
            inst_rwd += (rwd - inst_rwd) / h
        return inst_rwd


class EpGreedySampleAvg(BanditAlgo):
    def __init__(self, testbed, horizon=1000):
        super().__init__(testbed, horizon)
        self.k = testbed.number_of_arms()
        self.expect = None

    def reset(self, param):
        self.testbed.set_arms()
        self.expect = [[0, 0, i] for i in range(self.k)]

    def select_arm(self, param):
        if param > np.random.rand():
            return np.random.randint(self.k)
        greedy_arm = 0
        for a in range(self.k):
            if self.expect[a][1] > self.expect[greedy_arm][1]: 
                greedy_arm = a
        return greedy_arm

    def update_expectation(self, param, arm, rwd):
        e = self.expect[arm]
        e[0] += 1
        e[1] += (rwd - e[1]) / e[0]


class EpGreedyConstantStep(BanditAlgo):
    def __init__(self, testbed, alpha=0.1, horizon=1000):
        super().__init__(testbed, horizon)
        self.k = testbed.number_of_arms()
        self.alpha = alpha
        self.expect = None

    def reset(self, param):
        self.testbed.set_arms()
        self.expect = [[0, i] for i in range(self.k)]

    def select_arm(self, param):
        if param > np.random.rand():
            return np.random.randint(self.k)
        greedy_arm = 0
        for a in range(self.k):
            if self.expect[a][0] > self.expect[greedy_arm][0]:
                greedy_arm = a
        return greedy_arm

    def update_expectation(self, param, arm, rwd):
        self.expect[arm][0] += self.alpha * (rwd - self.expect[arm][0])


class UCB(BanditAlgo):
    def __init__(self, testbed, horizon=1000):
        super().__init__(testbed, horizon)
        self.k = testbed.number_of_arms()
        self.expect = None
        self.total_time = 0

    def reset(self, param):
        self.testbed.set_arms()
        self.expect = [[0, 0, float('inf'), i] for i in range(self.k)]
        self.total_time = 0

    def select_arm(self, param):
        greedy_arm = 0
        for a in range(self.k):
            if self.expect[a][2] > self.expect[greedy_arm][2]:
                greedy_arm = a
        return greedy_arm

    def update_expectation(self, param, arm, rwd):
        self.total_time += 1
        e = self.expect[arm]
        e[0] += 1
        e[1] += (rwd - e[1]) / e[0]
        for e in self.expect:
            if e[0] == 0:
                continue
            e[2] = e[1] + param * np.sqrt(np.log(self.total_time) / e[0])


class OptimisticGreedy(BanditAlgo):
    def __init__(self, testbed, alpha=0.1, horizon=1000):
        super().__init__(testbed, horizon)
        self.alpha = alpha
        self.k = testbed.number_of_arms()
        self.expect = None

    def reset(self, param):
        self.testbed.set_arms()
        self.expect = [[param, i] for i in range(self.k)]

    def select_arm(self, param):
        greedy_arm = 0
        for a in range(self.k):
            if self.expect[a][0] > self.expect[greedy_arm][0]:
                greedy_arm = a
        return greedy_arm

    def update_expectation(self, param, arm, rwd):
        e = self.expect[arm]
        e[0] += self.alpha* (rwd - e[0])


class GradientBandit(BanditAlgo):
    def __init__(self, testbed, horizon=1000):
        super().__init__(testbed, horizon)
        self.k = testbed.number_of_arms()
        self.preference = None
        self.avg = 0
        self.cnt = 0

    def reset(self, param):
        self.testbed.set_arms()
        self.preference = np.zeros(self.k)
        self.avg = 0
        self.cnt = 0

    def select_arm(self, param):
        exp = np.exp(self.preference)
        prob = exp / np.sum(exp)
        rand = np.random.rand()
        cumsum = 0
        for a in range(self.k):
            cumsum += prob[a]
            if rand < cumsum:
                return a
        return 0

    def update_expectation(self, param, arm, rwd):
        self.cnt += 1
        self.avg += (rwd - self.avg) / self.cnt
        rwd_diff = param * (rwd - self.avg)
        exp = np.exp(self.preference)
        prob = exp / np.sum(exp)
        self.preference -= rwd_diff * prob
        self.preference[arm] += rwd_diff



def main():
    runner = Runner(horizon=5000, runs=2000)

    runner.run_with_params(EpGreedySampleAvg)
    runner.save('./non_stationary/ep_greedy_smpl.csv')

    runner.run_with_params(EpGreedyConstantStep)
    runner.save('./non_stationary/ep_greedy_const.csv')

    runner.run_with_params(GradientBandit)
    runner.save('./non_stationary/gradient_bandit.csv')

    runner.run_with_params(UCB)
    runner.save('./non_stationary/ucb.csv')

    runner.run_with_params(OptimisticGreedy)
    runner.save('./non_stationary/opti_greedy.csv')


if __name__ == '__main__':
    main()



