import numpy as np
import torch as th
from models import Model
from queue import Queue
from config import Configuration
from utils import ReplayMemory, Environment


class Execute:
    def __init__(self, path):
        self.config = Configuration.construct(path)
        self.env = Environment(self.config)
        self.memory = ReplayMemory(self.config)
        self.model = Model(self.config)
        self.ep = None

    def get_epsilon(self, is_play):
        if is_play:
            return self.config.play.ep
        ep_start = self.config.train.ep.start
        ep_final = self.config.train.ep.final
        ep_num_frames = self.config.train.ep.num_frames
        decay = (ep_start - ep_final) / ep_num_frames
        if self.ep is None:
            self.ep = ep_start
        self.ep = max(self.ep - decay, ep_final)
        return self.ep

    def log(self, **kawrgs):
        log = ""
        for name, value in kawrgs.items():
            log += f"{name}: {value}, "
        print(log)

    def run_episode(self, episode=1, steps=0, is_play=True, debug=False):
        config = self.config

        self.env.reset()
        action = 1
        _, _, curr_state, is_done = self.env.step(action)
        total_reward = 0
        update_net = 0; C = config.train.network_update_freq
        t = 0; T = config.max_episode_length

        while not is_done and t < T:
            if t % config.action_repeat == 0:
                ep = self.get_epsilon(is_play)
                action = self.model.choose_action(curr_state, ep)
            prev_state, reward, curr_state, is_done = self.env.step(action)
            total_reward += reward
            t += 1

            if is_play:
                self.env.render("human")
                if debug and t % config.play.debug.time == 0:
                    self.log(ftype=self.env.get_frame_type(), action=action, reward=total_reward)
                continue

            self.memory.add((prev_state, action, reward, curr_state, is_done))
            if self.memory.get_size() > config.train.replay_start_size:
                for i in range(config.train.batch_run):
                    batch = self.memory.sample()
                    self.model.optimize(batch)
                    steps = (steps + 1) % C
                if steps % C == 0:
                    self.model.update_qhat()
                    update_net += 1

        if not is_play and debug and episode % config.train.debug.time == 0:
            self.log(ftype=self.env.get_frame_type(), total_reward=total_reward, network_update_steps=update_net, episode_time=t, ep=ep)

        return total_reward, steps

    def load_model(self):
        ftype = self.env.get_frame_type()
        in_size = self.env.get_in_size()
        num_actions = self.env.get_num_actions()
        self.model.load_model(ftype, in_size, num_actions)

    def play(self, debug=False):
        self.load_model()
        for ep in range(1):
            self.run_episode(is_play=True, debug=debug)

    def train(self, debug=False):
        self.load_model()
        optimize_steps = 0
        episodes = self.config.train.episodes
        for episode in range(1, episodes+1):
            reward, steps = self.run_episode(episode=episode, steps=optimize_steps, is_play=False, debug=debug)
            optimize_steps += steps
            if episode % self.config.train.save_model_episode == 0:
                self.model.save_model()
        self.model.update_qhat()
        self.model.save_model()

    def close(self):
        self.env.close()
        self.memory.close()


if __name__ == '__main__':
    runner = Execute("./config.yaml")
    runner.train(debug=True)
#    runner.play(True)
    runner.close()


