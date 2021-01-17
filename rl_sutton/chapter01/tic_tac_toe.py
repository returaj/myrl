# Original code is from : https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py
# This code has been rewritten and modified for testing


import numpy as np
import pickle


BOARD_ROW = 3
BOARD_COL = 3
BOARD_SPACE = BOARD_ROW*BOARD_COL


class State:
    def __init__(self):
        self.board = np.zeros((BOARD_ROW, BOARD_COL))
        self.winner = 0
        self.end = None

    def hash(self):
        h = 0
        for x in np.nditer(self.board):
            h = 3*h + x + 1
        return h

    def next(self, i, j, symbol):
        assert self.board[i, j] == 0
        new_state = State()
        new_state.board = np.copy(self.board)
        new_state.board[i, j] = symbol
        return new_state

    def is_end(self):
        if self.end is not None:
            return self.end

        check = []
        # check row
        for i in range(BOARD_ROW):
            check.append(sum(self.board[i, :]))

        # check col
        for i in range(BOARD_COL):
            check.append(sum(self.board[:, i]))

        # check diagonal
        diagonal = 0; reverse_diagonal = 0
        for i in range(BOARD_ROW):
            diagonal += self.board[i, i]
            reverse_diagonal += self.board[BOARD_ROW-i-1, i]
        check.append(diagonal)
        check.append(reverse_diagonal)

        for x in check:
            if x == 3:
                self.end = True
                self.winner = 1
                return self.end
            elif x == -3:
                self.end = True
                self.winner = -1
                return self.end

        for x in np.nditer(self.board):
            if x == 0:
                self.end = False
                return self.end

        self.end = True
        return self.end

    def print_state(self):
        for i in range(BOARD_ROW):
            print('-------------')
            out = '| '
            for j in range(BOARD_COL):
                if self.board[i, j] == 1:
                    token = '*'
                elif self.board[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')


class Player:
    def __init__(self, epsilon=0, step_size=0.1):
        self.epsilon = epsilon
        self.step_size = step_size
        self.values = dict()
        self.states = []
        self.greedy = []
        self.symbol = 0

    def set_symbol(self, symbol):
        self.symbol = symbol
        for h, state in ALL_STATES.items():
            if state.is_end():
                if state.winner == symbol:
                    self.values[h] = 1.0
                elif state.winner == 0:
                    self.values[h] = 0.5
                else:
                    self.values[h] = 0.0
            else:
                self.values[h] = 0.5

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def reset(self):
        self.states = []
        self.greedy = []

    def back_up(self):
        states = [s.hash() for s in self.states]
        for i in reversed(range(len(states)-1)):
            td = self.greedy[i]*(self.values[states[i+1]] - self.values[states[i]])
            #td = self.values[states[i+1]] - self.values[states[i]]
            self.values[states[i]] += self.step_size*td

    def play(self):
        state = self.states[-1]
        possible_moves = []
        for i in range(BOARD_ROW):
            for j in range(BOARD_COL):
                if state.board[i, j] == 0:
                    new_state_hash = state.next(i, j, self.symbol).hash()
                    possible_moves.append((self.values[new_state_hash], i, j))
        if self.epsilon > np.random.rand():
            m = possible_moves[np.random.randint(len(possible_moves))]
            self.greedy[-1] = False
            return m[1], m[2], self.symbol
        np.random.shuffle(possible_moves)
        possible_moves.sort(reverse=True, key=lambda x: x[0])
        m = possible_moves[0]
        return m[1], m[2], self.symbol

    def save_values(self):
        filepath = "first.pickle" if self.symbol==1 else "second.pickle"
        with open(filepath, 'wb') as fp:
            pickle.dump(self.values, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_values(self):
        filepath = "first.pickle" if self.symbol==1 else "second.pickle"
        with open(filepath, 'rb') as fp:
            self.values = pickle.load(fp)


class HumanPlayer:
    def __init__(self):
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None
        self.symbol = 0

    def reset(self):
        self.state = None

    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_state(self, state):
        self.state = state

    def play(self):
        self.state.print_state()
        key = input("Input your position: ")
        p = self.keys.index(key)
        i = p//BOARD_COL
        j = p%BOARD_COL
        return i, j, self.symbol


class Judge:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.player1.set_symbol(1)
        self.player2.set_symbol(-1)
        self.end_state = None

    def reset(self):
        self.player1.reset()
        self.player2.reset()
        self.end_state = None

    def alternate(self):
        while True:
            yield self.player1
            yield self.player2

    def game_play(self, print_state=False):
        self.reset()
        state = State()

        if print_state:
            state.print_state()

        self.player1.set_state(state)
        self.player2.set_state(state)

        alternate = self.alternate()

        while not state.is_end():
            player = next(alternate)
            px, py, symbol = player.play()
            state = state.next(px, py, symbol)
            self.player1.set_state(state)
            self.player2.set_state(state)
            if print_state:
                state.print_state()

        self.end_state = state
        return state.winner


def explore_all_states(state, symbol, all_states):
    for i in range(BOARD_ROW):
        for j in range(BOARD_COL):
            if state.board[i, j] == 0:
                new_state = state.next(i, j, symbol)
                new_state_hash = new_state.hash()
                if new_state_hash not in all_states:
                    all_states[new_state_hash] = new_state
                    if not new_state.is_end():
                        explore_all_states(new_state, -symbol, all_states)

def get_all_states():
    all_states = dict()
    curr_state = State()
    curr_state_hash = curr_state.hash()
    all_states[curr_state_hash] = curr_state
    explore_all_states(curr_state, 1, all_states)
    return all_states

ALL_STATES = get_all_states()

def train(epochs=1_00_000):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judge = Judge(player1, player2)

    win_player1 = 0
    win_player2 = 0
    draw = 0

    for i in range(1, epochs+1):
        winner = judge.game_play()
        player1.back_up()
        player2.back_up()
        if winner == 1:
            win_player1 += 1
        elif winner == -1:
            win_player2 += 1
        else:
            draw += 1
        if i % 10000 == 0:
            p = win_player1/(win_player1 + win_player2)
            print(f"Epoch: {i}, player1: {win_player1}, player2: {win_player2}, percetage_win_by_player1: {p}")
    player1.save_values()
    player2.save_values()

def bot_compete(epochs=1000):
    p1 = Player(epsilon=0)
    p2 = Player(epsilon=0)
    judge = Judge(p1, p2)

    p1.load_values()
    p2.load_values()

    w1 = 0; w2 = 0; d = 0
    for e in range(1, epochs+1):
        winner = judge.game_play()
        if winner == 1:
            w1 += 1
        elif winner == -1:
            w2 += 1
        else:
            d += 1
    print(f"Player1: {w1}, Player2: {w2}, Draw: {d}")

def compete(match=6):
    human = HumanPlayer()
    ai = Player(epsilon=0)

    judge = Judge(human, ai)
    ai.load_values()

    for i in range(match):
        winner = judge.game_play()
        judge.end_state.print_state()
        if winner == 1:
            print("Human wins")
        elif winner == -1:
            print("AI wins")
        else:
            print("DRAW")
    ai.save_values()


if __name__ == '__main__':
    train()
    bot_compete()
    compete(6)
