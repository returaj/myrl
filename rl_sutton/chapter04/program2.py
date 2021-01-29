#! /usr/bin/env python3

# Gambler problem: exapmle 4.3 and exercise 4.9

import numpy as np


ENV = None
ID2STATE = {}


class State:
    def __init__(self, stake):
        self.stake = stake
        self.rwd_trans = None
        self.state_trans = None

    def get_transition(self):
        global ENV
        if (self.rwd_trans is None) or (self.state_trans is None):
            self.rwd_trans, self.state_trans = ENV.get_transition(self)
        return self.rwd_trans, self.state_trans


class Environment:
    def __init__(self, ph, win_stake=100, reward=1):
        self.ph = ph
        self.win_stake = win_stake
        self.reward = reward

    def get_transition(self, state):
        curr_stake = state.stake
        min_action, max_action = 0, min(curr_stake, self.win_stake - curr_stake)
        min_state, max_state = 0, self.win_stake
        min_rwd, max_rwd = 0, self.reward
        rwd_trans = np.zeros((max_action-min_action+1, max_rwd-min_rwd+1))
        state_trans = np.zeros((max_action-min_action+1, max_state-min_state+1))
        for a in range(min_action, max_action+1):
            # wins
            wstake = curr_stake + a
            wrwd = self.reward if wstake == self.win_stake else 0
            rwd_trans[a][wrwd] += self.ph
            state_trans[a][wstake] += self.ph

            # loose
            lstake = curr_stake - a
            lrwd = 0
            rwd_trans[a][lrwd] += (1-self.ph)
            state_trans[a][lstake] += (1-self.ph)
        return rwd_trans, state_trans


class ValueIteration:
    def __init__(self, theta=1e-5):
        self.theta = theta
        self.v = np.zeros(ENV.win_stake + 1)
        self.policy = np.zeros(ENV.win_stake+1)

    def optimal_value_evaluaion(self):
        #policy_stable = False
        delta = 1
        while (delta > self.theta):
        #while not policy_stable:
            delta = 0
            #policy_stable = True
            for s in range(1, ENV.win_stake):
                tmp = self.v[s]
                #sa = self.policy[s]
                state = ID2STATE[s]
                rwd_trans, state_trans = state.get_transition()
                max_v = 0; max_a = 0
                for a in range(0, len(rwd_trans)):
                    exp_r = 0
                    for r in range(0, ENV.reward+1):
                        exp_r += r * rwd_trans[a][r]
                    exp_v = 0
                    for ns in range(0, ENV.win_stake+1):
                        exp_v += self.v[ns] * state_trans[a][ns]
                    if max_v < exp_r + exp_v:
                        max_v = exp_r + exp_v
                        max_a = a
                self.v[s] = max_v
                self.policy[s] = max_a
                delta = max(delta, abs(tmp - max_v))
                #if max_a != sa:
                #    policy_stable = False

    def optimal_policy(self):
        self.optimal_value_evaluaion()
        for s in range(1, ENV.win_stake):
            state = ID2STATE[s]
            rwd_trans, state_trans = state.get_transition()
            max_v = -1; max_a = 0
            for a in range(0, len(rwd_trans)):
                exp_r = 0
                for r in range(0, ENV.reward+1):
                    exp_r += r*rwd_trans[a][r]
                exp_v = 0
                for ns in range(0, ENV.reward+1):
                    exp_v += self.v[ns] * state_trans[a][ns]
                if max_v < exp_r + exp_v:
                    max_v = exp_r + exp_v
                    max_a = a
            self.policy[s] = max_a


def map_id_to_state(ph):
    global ENV
    ENV = Environment(ph)
    for i in range(1, ENV.win_stake):
        state = State(i)
        state.get_transition()
        ID2STATE[i] = state


def main():
    # probability of winning (head) = 0.4
    map_id_to_state(0.4)
    vi = ValueIteration()
    vi.optimal_policy()
    print(vi.v)    


if __name__ == '__main__':
    main()






