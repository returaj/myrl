#!/usr/bin/env python3

# example : 5.3

import matplotlib
matplotlib.use('Agg')  # no UI backend
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


# black jack with infinite deck
class Environment:
    def __init__(self):
        # King, Queen, Jack each of value 10
        # Ace can be either 1 or 11
        # 2, 3, 4, ..., 9 cards has value equal to its face value
        self.prob = np.cumsum([1/13] * 13)

    def hit(self):
        rand = np.random.rand()
        for c in range(0, 13):
            if rand < self.prob[c]:
                return c+1;
        raise Exception("Card dealt is invaid!!")

    def card_value(self, card, curr_sum):
        if card != 1:
            return min(card, 10)
        if curr_sum + 11 <= 21:
            return 11
        return 1


class State:
    def __init__(self, dealer_showing, player_sum, has_usable_ace):
        self.dealer_showing = dealer_showing
        self.player_sum = player_sum
        self.has_usable_ace = has_usable_ace
        self.v = 0
        self.cnt = 0
        self.hashid = None

    def hash(self):
        if self.hashid is None:
            self.hashid = (self.dealer_showing-1) + 10*(self.player_sum-12) + 100*self.has_usable_ace
        return self.hashid

    def update(self, rwd):
        self.cnt += 1
        self.v += (rwd - self.v) / self.cnt


class QState(State):
    def __init__(self, dealer_showing, player_sum, has_usable_ace, action):
        super().__init__(dealer_showing, player_sum, has_usable_ace)
        self.action = action   # True: hit and False: stick

    def hash(self):
        if self.hashid is None:
            self.hashid = (self.dealer_showing-1) + 10*(self.player_sum-12) \
                          + 100*self.has_usable_ace + 200*self.action
        return self.hashid


class Dealer:
    def __init__(self, env, min_stick_value=17):
        # plays with a fixed strategy if sum is greater than or equal to 17
        # dealer choose to stick else hit
        self.env = env
        self.min_stick_value = min_stick_value
        self.policy = None

    def get_policy(self):
        if self.policy is None:
            self.policy = {}
            # for not usable ace
            self.policy[False] = [True]*(self.min_stick_value-1) + [False]*(21-self.min_stick_value+1)
            # for usable ace
            self.policy[True] = [True]*(self.min_stick_value-1) + [False]*(21-self.min_stick_value+1)
        return self.policy

    def play(self, state):
        policy = self.get_policy()
        curr_sum = self.env.card_value(state.dealer_showing, 0)
        has_usable_ace = curr_sum == 11   # dealer has a usable ace
        while curr_sum < 21 and policy[has_usable_ace][curr_sum-1]:
            card = self.env.hit()
            value = self.env.card_value(card, curr_sum)
            curr_sum += value
            if value == 11:
                has_usable_ace = True
            if curr_sum > 21 and has_usable_ace:
                curr_sum -= 10
                has_usable_ace = False
        return curr_sum


class Player:
    def __init__(self, env):
        # it is optimal for player to always hits if current_sum is less than 12
        # so policy contains action for current_sum equals 12 to 21
        self.env = env
        self.policy = None

    def get_policy(self):
        if self.policy is None:
            self.policy = {}
            # for not usable ace
            self.policy[False] = np.zeros((10, 10), dtype=bool)
            # for usable ace
            self.policy[True] = np.zeros((10, 10), dtype=bool)
        return self.policy

    def play(self, state):
        policy = self.get_policy()
        game_play = []
        dealer_showing = state.dealer_showing
        curr_sum = state.player_sum
        has_usable_ace = state.has_usable_ace
        action = state.action
        while curr_sum <= 21:
            game_play.append(QState(dealer_showing, curr_sum, has_usable_ace, action).hash())
            if not action:
                break
            card = self.env.hit()
            value = self.env.card_value(card, curr_sum)
            curr_sum += value
            if curr_sum > 21 and has_usable_ace:
                curr_sum -= 10
                has_usable_ace = False
            if curr_sum <= 21:
                action = policy[has_usable_ace][dealer_showing-1][curr_sum-12]
        return (game_play, curr_sum)

    def update_policy(self, state, map_id_to_state):
        s1 = state.hash(); state1 = map_id_to_state[s1]
        s2 = s1%200 if s1>=200 else (200+s1); state2 = map_id_to_state[s2]
        assert state1.has_usable_ace == state2.has_usable_ace and \
               state1.dealer_showing == state2.dealer_showing and \
               state1.player_sum == state2.player_sum and \
               state1.action != state2.action
        policy = self.get_policy()
        policy[state.has_usable_ace][state.dealer_showing-1][state.player_sum-12] = \
                state1.action if state1.v > state2.v else state2.action


class Episode:
    def __init__(self):
        self.start_states = self.initialize_start_states()
        self.start = self.select_start() 

    def initialize_start_states(self):
        start_states = []
        for action in range(0, 2):
            for has_usable_ace in range(0, 2):
                for psum in range(12, 22):
                    for dshow in range(1, 11):
                        start_states.append(QState(dshow, psum, has_usable_ace==1, action==1))
        np.random.shuffle(start_states)
        return start_states

    def select_start(self):
        i = -1; total_nums = len(self.start_states)
        while True:
            i = (i+1) % total_nums
            yield self.start_states[i]

    def generate(self, player, dealer):
        start = next(self.start)
        game_play, player_sum = player.play(start)
        if player_sum == 21:   # wins by reaching 21
            return (game_play, 1)
        elif player_sum > 21:  # player goes bust
            return (game_play, -1)
        dealer_sum = dealer.play(start)
        if dealer_sum > 21:    # dealer goes bust
            return (game_play, -1)
        elif dealer_sum == player_sum:    # player_sum and dealer_sum is same
            return (game_play, 0)
        else:
            rwd = -1 if dealer_sum > player_sum else 1   # both have valid sum [<21], one with higher sum wins
            return (game_play, rwd)


class MonteCarloPrediction:
    def __init__(self, num_of_episodes=10000):
        self.num_of_episodes = num_of_episodes

    def evaluate(self, num_episodes):
        map_id_to_states = {}
        for action in range(0, 2):
            for has_usable_state in range(0, 2):
                for psum in range(12, 22):
                    for dshow in range(1, 11):
                        state = QState(dshow, psum, has_usable_state==1, action==1)
                        map_id_to_states[state.hash()] = state
        env = Environment()
        ep = Episode()
        player = Player(env)
        dealer = Dealer(env)
        for i in range(num_episodes):
            game_play, R = ep.generate(player, dealer)
            G = 0
            for i in range(len(game_play)-1, -1, -1):
                G += R
                state = map_id_to_states[game_play[i]]
                state.update(G)
                player.update_policy(state, map_id_to_states)
                R = 0
        return player.get_policy()

    def save_figures(self):
        num_ep = 500_000
        optimal_policy = self.evaluate(num_ep)
        x = list(range(12, 22))
        y = list(range(1, 11))
        X, Y = np.meshgrid(x, y)
        Z_usable_ace = optimal_policy[True]
        Z_not_usable_ace = optimal_policy[False]
        with open('usable_ace.pkl', 'wb') as fp:
            pkl.dump(Z_usable_ace, fp)
        with open('not_usable_ace.pkl', 'wb') as fp:
            pkl.dump(Z_not_usable_ace, fp)
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z_not_usable_ace)
        ax1.set_title(f'Not usable ace: {num_ep}')
        ax2.plot_surface(X, Y, Z_usable_ace)
        ax2.set_title(f'Usable ace: {num_ep}')
        plt.savefig("figure5_3.png")


def main():
    mc_prediction = MonteCarloPrediction()
    mc_prediction.save_figures()


if __name__ == '__main__':
    main()



