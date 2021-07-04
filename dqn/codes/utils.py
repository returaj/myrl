import gym
import torch as th
import numpy as np
import cv2
import random
import time
from queue import Queue


class Environment:
    def __init__(self, config):
        self.config = config
        self.preprocess = Preprocessing(config)
        self.game = gym.make(config.map)
        self.state_memory = Queue()
        self.curr_state = None
        self.ftype = None
        self.in_size = None
        self.prev_lives = 0 

    def update_frame_info(self):
        frame = self.preprocess.process(self.game.reset())
        self.ftype = self.preprocess.frame_type(frame)
        self.in_size = frame.shape[0]

    def get_frame_type(self):
        if self.ftype is None:
            self.update_frame_info()
        return self.ftype

    def get_in_size(self):
        if self.in_size is None:
            self.update_frame_info()
        return self.in_size

    def get_num_actions(self):
        return self.game.action_space.n

    def reset_state_memory(self):
        while self.state_memory.qsize() > 0:
            self.state_memory.get()

    def add_to_state_memory(self, game_frame):
        frame = self.preprocess.process(game_frame)
        self.state_memory.put(frame)
        ftype = self.get_frame_type()
        ftype_config = getattr(self.config, ftype)
        if self.state_memory.qsize() > (ftype_config.history * (ftype_config.skip+1)):
            self.state_memory.get()

    def reset(self):
        self.reset_state_memory()
        self.add_to_state_memory(self.game.reset())
        self.curr_state = None

    def step(self, action):
        game_frame, reward, is_done, info = self.game.step(action)
        lives = info.get('ale.lives', 1)
#        lives = 0 if lives < self.prev_lives else lives
        is_done = self.get_is_done(is_done, lives)
        # setting cart-pole is_done reward to -5
        scaled_reward = self.get_scaled_reward(reward, lives)
        self.add_to_state_memory(game_frame)
        next_state = self.preprocess.state2tensor(self.state_memory)
        ret = (self.curr_state, scaled_reward, next_state, is_done)
        self.curr_state = next_state
        self.prev_lives = lives
        return ret

    def get_is_done(self, is_done, lives):
#        return True if lives == 0 else is_done
        return is_done

    def get_scaled_reward(self, reward, lives):
#        if lives == 0:
#            return -5
        if lives < self.prev_lives:
            return -2
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        return reward

    def render(self, render_type):
        self.game.render(render_type)
        time.sleep(self.config.play.render_time)

    def close(self):
        self.game.close()


class ReplayMemory:
    def __init__(self, config):
        self.config = config
        self.D = None
        self.size = 0
        self.index = 0

    def get_size(self):
        return self.size

    def get_mem(self):
        if self.D is None:
            self.D = [None] * self.config.train.replay_memory_size
        return self.D

    def sample(self):
        batch_size = min(self.config.train.batch_size, self.size)
        assert batch_size <= self.size
        idx = random.sample(range(self.size), batch_size)
        return [self.D[i] for i in idx]

    def add(self, elem):
        D = self.get_mem()
        max_len = len(self.D)
        D[self.index % max_len] = elem
        self.index = (self.index + 1) % max_len
        self.size = min(self.size+1, max_len)

    def close(self):
        if self.D:
            self.D.clear()
            self.index, self.size = 0, 0


class Preprocessing:
    def __init__(self, config):
        self.config = config

    def frame_type(self, frame):
        if frame.ndim > 3:
            raise Exception("Invalid frame dimension: {}".format(frame.ndim))
        return "ram" if frame.ndim == 1 else "image"

    def process(self, frame, prev_frame=None):
        """
        frame: numpy array
        prev_frame: numpy array
        """
        ftype = self.frame_type(frame)
        if ftype == "ram":
            return self.process_ram(frame, prev_frame)
        if ftype == "image":
            return self.process_image(frame, prev_frame)
        raise Exception("Not able to preprocess frame of dim: {}".format(frame.ndim))

    def process_ram(self, frame, prev_frame=None):
        return frame.astype('float32')

    def process_image(self, frame, prev_frame=None):
        if prev_frame is not None:
            frame = np.maximum(frame, prev_frame)
        frame = frame[30:-12, 5:-4]
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_frame, _, _ = cv2.split(yuv)
        in_size = self.config.image.in_size
        resize_frame = cv2.resize(y_frame, (in_size, in_size), interpolation=cv2.INTER_NEAREST)
        return resize_frame.astype('float32')

    def state2tensor(self, state):
        """
        state: Queue
        history: int
        """
        def totensor(t):
            dim = t[0].ndim
            if dim == 1:
                return th.tensor(np.diff(t, axis=0)[0])
            while len(t) < self.config.image.history:  # image in_channel
                t.append(t[-1])
            assert len(t) == self.config.image.history
            return th.tensor(t, dtype=th.float)

        t = []; time = 0
        for s in reversed(state.queue):
            skip = getattr(self.config, self.frame_type(s)).skip
            if time % (skip + 1) == 0:
                t.append(s)
            else:
                t[-1] = np.maximum(t[-1], s)
            time += 1
        return totensor(t)



