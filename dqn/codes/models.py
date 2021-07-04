import re
import os
import numpy as np
import torch as th
from torch.nn import Module, Linear, Conv2d, SmoothL1Loss
from torch.optim import RMSprop
import torch.nn.functional as F

class DqnRam(Module):
	def __init__(self, in_size, num_actions):
		super(DqnRam, self).__init__()
		self.fc1 = Linear(in_size, 50)
		self.fc2 = Linear(50, 30)
		self.fc3 = Linear(30, num_actions)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)


class DqnImage(Module):
    def __init__(self, in_size, num_actions, in_channel):
        super(DqnImage, self).__init__()
        self.cnn1 = Conv2d(in_channel, 16, (5, 5), stride=2)
        self.cnn2 = Conv2d(16, 8, (5, 5), stride=2)
        shape = self.conv2d_size_out(self.conv2d_size_out(in_size))
        self.fc = Linear(shape*shape*8, num_actions)
#        self.fc2 = Linear(128, num_actions)

    def conv2d_size_out(self, size, kernel_size=5, stride=2):
        return (size - kernel_size) // stride + 1

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = x.reshape(x.size(0), -1)
#        x = F.relu(self.fc(x))
        return self.fc(x)


class Model:
    def __init__(self, config):
        self.config = config
        self.optimizer = None
        self.criterion = None

    def load_model(self, ftype, in_size, num_actions):
        self.num_actions = num_actions
        self.ftype = ftype
        if ftype == 'ram':
            self.q = DqnRam(in_size, num_actions)
            self.qhat = DqnRam(in_size, num_actions)
        elif ftype == 'image':
            in_channel = self.config.image.history
            self.q = DqnImage(in_size, num_actions, in_channel)
            self.qhat = DqnImage(in_size, num_actions, in_channel)
        self.initialize()

    def initialize(self):
        path = self.config.model_path
        if os.path.exists(path):
            self.q.load_state_dict(th.load(path))
            self.qhat.load_state_dict(th.load(path))

    def choose_action(self, state, ep=0.01):
        if self.ftype == 'image' and len(state.shape) != 4:
            if len(state.shape) != 3:
                raise Exception("Incorrect state shape {}".format(state.shape))
            state = state.unsqueeze(0)
        if np.random.rand() < ep:
            return np.random.randint(self.num_actions)
        with th.no_grad():
            return int(np.argmax(self.q(state)))

    def get_optimizer(self):
        if self.optimizer is None:
            self.optimizer = RMSprop(self.q.parameters())
        return self.optimizer

    def get_criterion(self):
        if self.criterion is None:
            self.criterion = SmoothL1Loss()
        return self.criterion

    def to_supervised_label(self, batch):
        size = len(batch)
        s, a, r, n, d = 0, 1, 2, 3, 4
        states, next_states = [], []
        for elem in batch:
            states.append(elem[s])
            next_states.append(elem[n])
        th_states = th.stack(states)
        th_next_states = th.stack(next_states)

        q_esti = self.q(th_states)
        x = th.stack([q_esti[i][batch[i][a]] for i in range(size)])

        with th.no_grad():
            q_next_esti = self.qhat(th_next_states)
        y = []
        for i in range(size):
            expected = batch[i][r]
            if not batch[i][d]:
                expected += self.config.gama * th.max(q_next_esti[i])
            y.append(expected)
        return (x, th.tensor(y, dtype=th.float))

    def optimize(self, batch):
        optimizer = self.get_optimizer()
        criterion = self.get_criterion()

        optimizer.zero_grad()
        x, y = self.to_supervised_label(batch)
        loss = criterion(y, x)
        loss.backward()
        for param in self.q.parameters():
            param.grad.data.clamp_(-5, 5)
        optimizer.step()

    def update_qhat(self):
        self.qhat.load_state_dict(self.q.state_dict())
        # assert self.q.state_dict() == self.qhat.state_dict()

    def save_model(self):
        path = self.config.model_path
        basedir, filename = os.path.dirname(path), os.path.basename(path)
        files = [f for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir, f))]
        regex = "{}[_,0-9]*".format(filename)
        models = [m for m in files if re.search(regex, m)]
        if len(models) > 4:
            models.sort(key = lambda x: int(x.split('_')[-1]))
            os.remove(models[0])
        num = int(models[-1].split('_')[-1]) + 1 if len(models) > 0 else 0
        filename = "{}_{}".format(filename, num)
        th.save(self.q.state_dict(), os.path.join(basedir, filename))







