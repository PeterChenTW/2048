from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DQNConfig
from dueling import OneHotDuelingDQN
from nstep import double_dqn_targets


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
        x = self.fc3(x)

        return x


class SumTree:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.pointer = 0
        self.size = 0

    def add(self, priority, data):
        self.data[self.pointer] = data
        self.update(self.pointer + self.capacity - 1, priority)
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        if not np.isfinite(priority) or priority <= 0:
            raise ValueError('priority must be finite and positive')
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, value):
        total = self.total()
        if self.size == 0 or total <= 0:
            raise ValueError('cannot sample an empty tree')
        if not np.isfinite(value):
            raise ValueError('sample value must be finite')
        value = np.clip(value, 0, np.nextafter(total, 0))
        idx = self._retrieve(0, value)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    def _retrieve(self, idx, value):
        while 2 * idx + 1 < len(self.tree):
            left = 2 * idx + 1
            if value < self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx

    def total(self):
        return self.tree[0]


class DDQNAgent:
    def __init__(self, state_size, action_size, config=None):
        self.config = config if config is not None else DQNConfig()
        if self.config.arch not in ('legacy', 'onehot_dueling'):
            raise ValueError(f'unknown architecture: {self.config.arch}')
        if (isinstance(self.config.n_step, bool) or
                not isinstance(self.config.n_step, int) or self.config.n_step < 1):
            raise ValueError('n_step must be a positive integer')
        self.state_size = state_size
        self.action_size = action_size
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.memory = SumTree(self.config.memory_capacity)
        self.beta = self.config.per_beta_start
        self.replay_calls = 0
        self.epsilon = self.config.epsilon_start
        network = DQN if self.config.arch == 'legacy' else OneHotDuelingDQN
        self.model = network()
        self.target_model = network()
        self.update_target_model()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.lr)
        self.losses = []

    @staticmethod
    def _state_tensor(state):
        return torch.as_tensor(state, dtype=torch.float32).detach().reshape(-1, 16)

    def remember(self, state, action, reward, next_state, done, next_invalid_moves=(), discount=None):
        valid_mask = np.ones(self.action_size, dtype=bool)
        valid_mask[list(next_invalid_moves)] = False
        discount = self.config.gamma if discount is None else discount
        transition = (self._state_tensor(state).clone(), action, reward,
                      self._state_tensor(next_state).clone(), done, discount, valid_mask)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = self.config.abs_error_upper
        self.memory.add(max_priority, transition)

    def q_values(self, state):
        with torch.no_grad():
            return self.model(self._state_tensor(state))

    def act(self, state, invalid_moves):
        valid_actions = [a for a in range(self.action_size) if a not in invalid_moves]
        if not valid_actions:
            raise ValueError('no valid actions')
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))
        values = self.q_values(state).reshape(-1)
        return valid_actions[int(values[valid_actions].argmax().item())]

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def replay(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        if batch_size < 1 or self.memory.size == 0:
            raise ValueError('replay requires a positive batch size and nonempty memory')
        # First update uses beta_start; after anneal_steps updates beta reaches beta_end.
        fraction = min(self.replay_calls / max(1, self.config.per_beta_anneal_steps), 1.0)
        self.beta = (self.config.per_beta_start + fraction *
                     (self.config.per_beta_end - self.config.per_beta_start))
        total = self.memory.total()
        samples = [self.memory.get(self.rng.uniform(i, i + 1) * total / batch_size)
                   for i in range(batch_size)]
        idxs, priorities, transitions = zip(*samples)
        probabilities = np.asarray(priorities) / total
        weights = (self.memory.size * probabilities) ** (-self.beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones, discounts, masks = zip(*transitions)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        discounts = torch.tensor(discounts, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        masks = torch.tensor(np.stack(masks), dtype=torch.bool)
        q = self.model(states).gather(1, actions[:, None]).squeeze(1)
        targets = double_dqn_targets(self.model, self.target_model, rewards,
                                     next_states, dones, masks, discounts)
        td = targets - q
        weight_tensor = torch.tensor(weights, dtype=q.dtype)
        loss = (weight_tensor * td.square()).mean()
        stats = dict(loss=loss.item(), q_mean=q.mean().item(), q_min=q.min().item(),
                     q_max=q.max().item(), target_mean=targets.mean().item(),
                     td_abs_mean=td.abs().mean().item(), is_weight_mean=weights.mean().item(),
                     is_weight_min=weights.min().item(), beta=self.beta, epsilon=self.epsilon)
        stats['nan'] = not all(np.isfinite(v) for v in stats.values())
        if stats['nan']:
            raise FloatingPointError(f'nonfinite replay statistics: {stats}')
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        for idx, error in zip(idxs, td.detach().abs().cpu().numpy()):
            self.memory.update(idx, (float(error) + self.config.per_eps) ** self.config.per_alpha)
        self.losses.append(stats['loss'])
        self.replay_calls += 1
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        stats['epsilon'] = self.epsilon
        return stats

    def load(self, name):
        payload, arch = _read_checkpoint(name)
        if arch != self.config.arch:
            raise ValueError(f'checkpoint architecture {arch!r} cannot load into '
                             f'{self.config.arch!r} agent')
        state = payload['state_dict'] if 'format' in payload else payload
        self.model.load_state_dict(state, strict=True)
        self.update_target_model()

    def save(self, name):
        state = self.model.state_dict()
        if self.config.arch != 'legacy':
            state = dict(format='dqn2048-v2', architecture=self.model.architecture,
                         state_dict=state, config=asdict(self.config))
        torch.save(state, name)


def _read_checkpoint(path):
    """Read CPU weights and validate the version/architecture before loading."""
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # PyTorch versions before weights_only
        payload = torch.load(path, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError('checkpoint must contain a state dictionary or dqn2048-v2 payload')
    if 'format' not in payload:
        return payload, 'legacy'
    if payload['format'] != 'dqn2048-v2':
        raise ValueError(f"unsupported checkpoint format: {payload['format']!r}")
    architectures = {'onehot_dueling_v1': 'onehot_dueling', 'legacy': 'legacy'}
    architecture = payload.get('architecture')
    if architecture not in architectures:
        raise ValueError(f'unsupported checkpoint architecture: {architecture!r}')
    arch = architectures[architecture]
    if payload.get('config', {}).get('arch', arch) != arch:
        raise ValueError('checkpoint architecture conflicts with its config')
    return payload, arch


def load_checkpoint(path):
    """Build an agent for either a raw legacy checkpoint or a versioned one."""
    payload, arch = _read_checkpoint(path)
    config = DQNConfig(**payload.get('config', {})) if 'format' in payload else DQNConfig()
    config.arch = arch
    agent = DDQNAgent(16, 4, config)
    state = payload['state_dict'] if 'format' in payload else payload
    agent.model.load_state_dict(state, strict=True)
    agent.update_target_model()
    return agent
