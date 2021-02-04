#! /usr/env/bin python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# length and width of the grid
LENGTH, WIDTH = 10, 7

# MAX_DISTANCE
MAX_DISTANCE = LENGTH*WIDTH

EXAMPLE_6_5 = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

EXERCISE_6_9_1 = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (1, 1), 5: (-1, -1), 6: (1, -1), 7: (-1, 1)}

EXERCISE_6_9_2 = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (1, 1), 5: (-1, -1), 6: (1, -1), 7: (-1, 1), 8: (0, 0)}

# type of action North, South, East, West movement
MAP_ACTION = EXERCISE_6_9_1


class Environment:
    def __init__(self):
        self.start, self.goal = (3, 0), (3, 7)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.stochastic_wind = [0, -1, 1]

    def get_wind_value(self, x, y):
        return self.wind[y]

    def stochastic_wind_direction(self, x, y):
        if self.wind[y] == 0:
            return 0
        return self.stochastic_wind[np.random.randint(3)]

    def get_start(self):
        return self.start

    def is_goal(self, x, y):
        return x == self.goal[0] and y == self.goal[1]

    def get_new_pos(self, x, y, a, stochastic=False):
        xmove, ymove = MAP_ACTION[a]
        xnew, ynew = x+xmove, y+ymove
        rwd = -1
        if xnew < 0 or ynew < 0 or xnew >= WIDTH or ynew >= LENGTH:
            return x, y, rwd
        wind_value = self.get_wind_value(x, y)
        if stochastic:
            wind_value += self.stochastic_wind_direction(x, y)
        xnew = min(WIDTH-1, max(0, xnew-wind_value))
        return xnew, ynew, rwd


class QState:
    def __init__(self, x, y, a, alpha=0.5, gama=1):
        self.x, self.y, self.a = x, y, a
        self.alpha, self.gama = alpha, gama
        self.q = 0
        self.hashid = None

    def update(self, rwd, next_q_value):
        self.q += self.alpha * (rwd + self.gama*next_q_value - self.q)

    def hash(self):
        if self.hashid is None:
            self.hashid = LENGTH*self.x + self.y + LENGTH*WIDTH*self.a
        return self.hashid


class TD0:
    def __init__(self, env, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon
        self.map_id_to_qstate = self.initialize_map()
        self.policy = self.initialize_policy()

    def initialize_map(self):
        map_id_to_qstate = {}
        for a in range(0, len(MAP_ACTION)):
            for x in range(0, WIDTH):
                for y in range(0, LENGTH):
                    qstate = QState(x, y, a)
                    map_id_to_qstate[qstate.hash()] = qstate
        return map_id_to_qstate

    def initialize_policy(self):
        # take eastward movement in all states
        return np.ones((WIDTH, LENGTH))

    def select_action(self, x, y):
        if self.epsilon > np.random.rand():
            return np.random.randint(0, len(MAP_ACTION))
        return self.policy[x][y]

    def update_policy(self, qstate):
        x, y = qstate.x, qstate.y
        hid = qstate.hash() % 70    
        q_greedy = qstate
        for a in range(0, len(MAP_ACTION)):
            qa = self.map_id_to_qstate[hid + 70*a]
            if qa.q > q_greedy.q:
                q_greedy = qa
        self.policy[x][y] = q_greedy.a

    def estimate(self, episodes=3000, stochastic=False):
        for ep in range(1, episodes+1):
            start_x, start_y = self.env.get_start()
            curr_action = self.select_action(start_x, start_y)
            q = self.map_id_to_qstate[QState(start_x, start_y, curr_action).hash()]
            while not self.env.is_goal(q.x, q.y):
                next_x, next_y, rwd = self.env.get_new_pos(q.x, q.y, q.a, stochastic)
                next_a = self.select_action(next_x, next_y)
                # this is equal to max(a) q(n_x, n_y, a)
                next_q = self.map_id_to_qstate[QState(next_x, next_y, next_a).hash()]
                q.update(rwd, next_q.q)
                self.update_policy(q)
                q = next_q
        return self.policy

    def estimate_q_learning(self, episodes=3000):
        for ep in range(1, episodes+1):
            x, y = self.env.get_start()
            while not self.env.is_goal(x, y):
                a = self.select_action(x, y)
                n_x, n_y, rwd = self.env.get_new_pos(x, y, a)
                q = self.map_id_to_qstate[QState(x, y, a).hash()]
                n_q = self.map_id_to_qstate[QState(n_x, n_y, self.policy[n_x][n_y]).hash()]
                q.update(rwd, n_q.q)
                self.update_policy(q)
                x, y = n_x, n_y
        return self.policy


def save_figure(policy, env):
    X, Y = [], []
    x, y = env.get_start()
    X.append(y+0.5); Y.append(6-x+0.5)
    dist = 1
    while not env.is_goal(x, y):
        x, y, _ = env.get_new_pos(x, y, policy[x][y])
        X.append(y+0.5); Y.append(6-x+0.5)
        dist += 1
        if dist > MAX_DISTANCE:
            print("Not able to find path from start to goal. More episodes are needed to converge")
            break
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y)
    xtick = np.arange(0, 11, 1)
    ytick = np.arange(0, 8, 1)
    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    plt.grid(which='major')
    plt.savefig("figure6_9_1.png")


def main():
    env = Environment()
    td0 = TD0(env)
    ## TD0 algorithm takes longer episodes to converge, approx 3000 episodes to converge
    policy = td0.estimate(8000, stochastic=False)
    ## Q learning is faster it takes much less number of episodes to converge to optimal. approx 200 episodes
    # policy = td0.estimate_q_learning(200)
    save_figure(policy, env)


if __name__ == '__main__':
    main()


