import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64*16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.view(-1, 1, 4, 4) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.pointer = 0

    def add(self, priority, data):
        idx = self.pointer + self.capacity - 1
        self.data[self.pointer] = data
        self.update(idx, priority)
        self.pointer = (self.pointer + 1) % self.capacity

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, value):
        idx = self._retrieve(0, value)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def _retrieve(self, idx, value):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        elif value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self):
        return self.tree[0]


class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = SumTree(6000)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.abs_error_upper = 1.
        self.gamma = 0.9
        self.epsilon = 0.15
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.model = DQN()
        self.target_model = DQN()
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.criterion = nn.MSELoss()
        self.losses = []

    def remember(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = self.abs_error_upper
        self.memory.add(max_priority, transition)

    def act(self, state, invalid_moves):
        valid_actions = [action for action in [
            0, 1, 2, 3] if action not in invalid_moves]

        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)
        else:
            act_values = self.model(state)
            action_values = act_values.detach().numpy()
            action_values[0][invalid_moves] = -np.inf

            if np.random.rand() <= 0.000:
                print(act_values, np.argmax(action_values), state.sum())
            return np.argmax(action_values)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size):
        this_loss = []
        minibatch = []
        idxs = []
        segment = self.memory.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, priority, data = self.memory.get(value)
            priorities.append(priority)
            idxs.append(idx)
            minibatch.append(data)

        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model(state)
            target_new = target.clone()
            if done:
                target_new[0][action] = reward
            else:
                t = self.target_model(next_state.detach())
                target_new[0][action] = reward + \
                    self.gamma * torch.max(t).item()

            self.optimizer.zero_grad()
            loss = self.criterion(target, target_new)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            self.optimizer.step()

            td_error = abs(
                target_new[0][action].item() - target[0][action].item())
            updated_priority = (td_error + 1e-5) ** self.alpha
            self.memory.update(idxs[i], updated_priority)

            this_loss.append(loss.item())
        self.losses.append(np.mean(this_loss))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
