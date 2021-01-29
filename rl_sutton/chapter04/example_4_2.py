#! /usr/bin/env python3

# Jacks car rental problem. Example 4.2

from multiprocessing import Pool
import numpy as np
import pickle
import sys


class Environment:
    def __init__(self, req1=3, ret1=3, req2=4, ret2=2, rent=10, cost=2, max_transfer=5, max_car=20):
        self.req1 = req1
        self.req1_dist = None        
        self.ret1 = ret1
        self.ret1_dist = None
        self.req2 = req2
        self.req2_dist = None
        self.ret2 = ret2
        self.ret2_dist = None
        self.rent = rent
        self.cost = cost
        self.max_transfer = max_transfer
        self.max_car = max_car

    def poisson_dist(self, l, n):
        dist = []
        factor = np.exp(-l)
        x = 1; lmbda = 1
        for i in range(0, n+1):
            if i > 0:
                x *= i; lmbda *= l
            dist.append(factor * lmbda / x)
        return dist

    def get_station1_dist(self):
        if (self.req1_dist is None) or (self.ret1_dist is None):
            self.req1_dist = self.poisson_dist(self.req1, self.max_car)
            self.ret1_dist = self.poisson_dist(self.ret1, self.max_car)
        return self.req1_dist, self.ret1_dist

    def get_station2_dist(self):
        if (self.req2_dist is None) or (self.ret2_dist is None):
            self.req2_dist = self.poisson_dist(self.req2, self.max_car)
            self.ret2_dist = self.poisson_dist(self.ret2, self.max_car)
        return self.req2_dist, self.ret2_dist

    def get_transition(self, state):
        req1, ret1 = self.get_station1_dist()
        req2, ret2 = self.get_station2_dist()
        min_action, max_action = -min(self.max_transfer, state.s2), min(self.max_transfer, state.s1)
        min_rwd, max_rwd = -self.cost*max(max_action, -min_action), 2*self.max_car*self.rent
        min_state, max_state = 0, (self.max_car + 1)*(self.max_car + 1)
        rwd_trans = np.zeros((max_action-min_action+1, max_rwd-min_rwd+1))
        state_trans = np.zeros((max_action-min_action+1, max_state-min_state))
        for a in range(min_action, max_action+1):
            c1, c2 = min(20, state.s1-a), min(20, state.s2+a)
            cost = self.cost * abs(a)
            for rq1 in range(0, self.max_car+1):
                for rt1 in range(0, self.max_car+1):
                    for rq2 in range(0, self.max_car+1):
                        for rt2 in range(0, self.max_car+1):
                            rwd = -cost; s1 = 0; s2 = 0
                            if c1 < rq1:
                                rwd += c1*self.rent
                                s1 = rt1
                            else:
                                rwd += self.rent * rq1
                                s1 = min(self.max_car, c1-rq1 + rt1)
                            if c2 < rq2:
                                rwd += c2 * self.rent
                                s2 = rt2
                            else:
                                rwd += self.rent * rq2
                                s2 = min(self.max_car, c2-rq2 + rt2)
                            action_id = abs(min_action) + a
                            state_id = State(s1, s2).hash()
                            rwd_id = abs(min_rwd) + rwd
                            prob = req1[rq1] * ret1[rt1] * req2[rq2] * ret2[rt2]
                            if action_id < 0 or state_id < 0 or rwd_id < 0 :
                                raise Exception(f"Some id is negative: {action_id}, {state_id}, {rwd_id}")
                            rwd_trans[action_id][rwd_id] += prob
                            state_trans[action_id][state_id] += prob
        return abs(min_action), abs(min_rwd), rwd_trans, state_trans


ENV = Environment()
ID2STATE = {}


class State:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.id = None
        self.state_trans = None
        self.rwd_trans = None
        self.shift_action = None
        self.shift_rwd = None

    def get_transition(self):
        if (self.state_trans is None) or (self.rwd_trans is None):
            self.shift_action, self.shift_rwd, self.rwd_trans, self.state_trans = ENV.get_transition(self)
        return self.rwd_trans, self.state_trans

    def hash(self):
        if self.id:
            return self.id
        self.id = 21*self.s1 + self.s2
        return self.id

    def from_id(state_id):
        s1 = state_id // 21
        s2 = state_id % 21
        return State(s1, s2)


class PolicyIteration:
    def __init__(self, theta=0.00001, gama=0.9):
        self.theta = theta
        self.max_state = (ENV.max_car + 1)* (ENV.max_car + 1)
        self.v = np.zeros(self.max_state)
        self.p = np.zeros(self.max_state)
        self.gama = gama

    def policy_evaluation(self):
        delta = 1
        while (delta < self.theta):
            delta = 0
            for sid in range(0, self.max_state):
                tmp = self.v[sid]
                state = ID2STATE[sid]
                rwd_trans, state_trans = state.get_transition()
                sa, sr = state.shift_action, state.shift_rwd
                exp_v = 0
                for nsid in range(0, self.max_state):
                    exp_v += self.gama * self.v[nsid] * state_trans[self.p[sid]+sa][nsid]
                exp_r = 0
                for r in range(0, len(rwd_trans[0])):
                    exp_r += (r - sr) * rwd_trans[self.p[sid]+sa][r]
                self.v[sid] = exp_r + exp_v
                delta = max(delta, abs(tmp - self.v[sid]))

    def policy_improvement(self):
        policy_stable = True
        self.policy_evaluation()
        for sid in range(0, self.max_state):
            old_action = self.p[sid]
            state = ID2STATE[sid]
            rwd_trans, state_trans = state.get_transition()
            sa, sr = state.shift_action, state.shift_rwd
            max_a = 0; max_v = 0;            
            for a in range(0, len(rwd_trans)):
                exp_v = 0
                for nsid in range(0, len(state_trans[0])):
                    exp_v += self.gama * self.v[nsid] * state_trans[a][nsid]
                exp_r = 0
                for r in range(0, len(rwd_trans[0])):
                    exp_r += (r - sr) * rwd_trans[a][r]
                if max_v < exp_v + exp_r:
                    max_v = exp_v + exp_r
                    max_a = a - sa
            self.p[sid] = max_a
            if old_action != max_a:
                policy_stable = False
        self.print_policy()
        if not policy_stable:
            self.policy_improvement()

    def print_policy(self):
        for s1 in range(20, -1, -1):
            for s2 in range(0, 21):
                sid = State(s1, s2).hash()
                print(self.p[sid], end=" ")
            print()


def in_process(sid):
    state = State.from_id(sid)
    state.get_transition()
    return state


def map_id_to_state():
    args = list(range(0, 21*21))
    with Pool(4) as pool:
        for i, state in enumerate(pool.imap_unordered(in_process, args)):
            ID2STATE[state.hash()] = state
            sys.stderr.write(f"\rCompleted: {100*(i+1)/441}")


def main():
    map_id_to_state()
    with open('tranition.pkl', 'wb') as fp:
        pickle.dump(ID2STATE, fp)
    if not ID2STATE:
        with open('transition.pkl', 'rb') as fp:
            tmp = pickle.load(fp)
        ID2STATE.update(tmp)
    pe = PolicyIteration()
    pe.policy_improvement()
    

if __name__ == "__main__":
    main()


