# Verifing the results of 10 armed testbed problem from Sutton Bartto Page No:28 Section: 2.3

import numpy as np
import copy


class StationaryNArmedTestBed:
    def __init__(self, k):
        self.k = k
        self.arms = [None]*k

    def set_arms(self):
        for i in range(self.k):
            # mean reward of an arm is a sample from normal distribution N(0, 1)
            # variance of an arm is set to 1
            self.arms[i] = [np.random.normal(0, 1), 1]

    def sample_reward(self, a):
        max_mean = max(self.arms, key=lambda x: x[0])[0]
        mu, sigma = self.arms[a]
        # reward from an arm is a sample from a gaussian distribution N(mu, sigma)
        return np.random.normal(mu, sigma), mu==max_mean


class NonStationaryNArmedTestBed(StationaryNArmedTestBed):
    def __init__(self, k):
        super().__init__(k)

    def sample_reward(self, a):
        for i in range(self.k):
            self.arms[i][0] += np.random.normal(0, 0.01)
        max_mean = max(self.arms, key=lambda x: x[0])[0]
        mu, sigma = self.arms[a]
        # arm's mean reward changes with time slghtly
        return np.random.normal(mu, sigma), mu==max_mean


class EpsilonGreedySampleAvg:
    def __init__(self, testbed, k, epsilon, alpha=0.1, horizon=1000, runs=2000):
        self.epsilon = epsilon
        self.testbed = testbed
        self.k = k
        self.alpha = alpha
        self.runs = runs
        self.horizon = horizon
        self.estimates = None

    def reset(self):
        self.estimates = [[0, 0, i] for i in range(self.k)]
        self.testbed.set_arms()

    def update_and_return_reward(self, arm):
        r, is_optimal = self.testbed.sample_reward(arm)
        e = self.estimates[arm]
        e[1] += 1
        e[0] += (r-e[0])/e[1]
        return r, is_optimal

    def run(self, explore=True, print_after_n=500):
        rewards = np.zeros(self.horizon)
        optimals = np.zeros(self.horizon)
        for r in range(1, self.runs + 1):
            self.reset()
            rlist, olist = [], []
            for t in range(1, self.horizon+1):
                a = None
                if explore and self.epsilon > np.random.rand():
                    a = np.random.randint(self.k)
                else:
                    tmp = copy.deepcopy(self.estimates)
                    np.random.shuffle(tmp)
                    tmp.sort(reverse=True, key=lambda x: x[0])
                    a = tmp[0][2]
                rw, is_optimal = self.update_and_return_reward(a)
                rlist.append(rw)
                olist.append(is_optimal)
            rewards += rlist
            optimals += olist
            print(f"Runs Completed: {r} \r", end="")
        rewards /= self.runs
        optimals = [x/(i+1) for i, x in enumerate(np.cumsum(optimals/self.runs))]
        return rewards, optimals


class EpsilonGreedyConstantStepSize(EpsilonGreedySampleAvg):
    def __init__(self, testbed, k, epsilon, alpha=0.1, horizon=1000, runs=2000):
        super().__init__(testbed, k, epsilon, alpha, horizon, runs)

    def update_and_return_reward(self, arm):
        r, is_optimal = self.testbed.sample_reward(arm)
        e = self.estimates[arm]
        e[1] += 1
        e[0] += self.alpha*(r-e[0])
        return r, is_optimal

def save(avg_rwd, avg_optimal, filepath):
    with open(filepath+"_rwd", 'w') as fp:
        for rwd in avg_rwd:
            fp.write(str(rwd) + "\n")
    with open(filepath+"_opti", 'w') as fp:
        for op in avg_optimal:
            fp.write(str(op) + "\n")

def main():
#    stationary_testbed = StationaryNArmedTestBed(k=10)
#    ep_greedy_sample_avg1 = EpsilonGreedySampleAvg(stationary_testbed, 10, 0.1)
    # average reward at each time step
    # algorithm: sample average for action value, greedy + exploration
#    save(*ep_greedy_sample_avg1.run(), "epsilon-0.1")

#    ep_greedy_sample_avg2 = EpsilonGreedySampleAvg(stationary_testbed, 10, 0.01)
    # average reward at each time step
    # algorithm: sample average for action value, greedy + exploration
#    save(*ep_greedy_sample_avg2.run(), "epsilon-0.01")

    # average reward at each time step
    # algorithm: sample average for action value, greedy
#    save(*ep_greedy_sample_avg2.run(explore=False), "greedy")

    non_stationary_testbed = NonStationaryNArmedTestBed(k=10)
    ep_greedy_sample_avg = EpsilonGreedySampleAvg(non_stationary_testbed, 10, 0.1, alpha=0.1, horizon=3000, runs=1000)
    save(*ep_greedy_sample_avg.run(), "sample-avg-non-stationary")

#    ep_greedy_constant_step = EpsilonGreedyConstantStepSize(non_stationary_testbed, 10, 0.1, alpha=0.1, horizon=3000, runs=1000)
#    save(*ep_greedy_constant_step.run(), "constant-step-non-stationary")



if __name__ == '__main__':
    main()



