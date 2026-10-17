"""One-step Double TD learning of scalar afterstate values with prioritized replay."""
import numpy as np
import torch

from afterstate_policy import afterstate_scores, select_action
from config import DQNConfig
from RL import DDQNAgent


class AfterstateAgent(DDQNAgent):
    def __init__(self, state_size, action_size, config=None):
        config = config if config is not None else DQNConfig(arch='onehot_afterstate')
        if config.arch != 'onehot_afterstate':
            raise ValueError('AfterstateAgent requires arch=onehot_afterstate')
        if config.n_step != 1:
            raise ValueError('onehot_afterstate requires n_step=1')
        super().__init__(state_size, action_size, config)

    def _reward_options(self):
        return dict(gamma=self.config.gamma,
                    reward_mode=getattr(self.config, 'reward_mode', 'log'),
                    reward_scale=getattr(self.config, 'reward_scale', 256.0))

    def act(self, state, invalid_moves):
        return select_action(self.model, self._state_tensor(state).reshape(16),
                             invalid_moves, self.epsilon, self.rng, **self._reward_options())

    def greedy_action(self, grid, invalid_moves):
        # A greedy evaluation must not consume the training exploration RNG.
        result = afterstate_scores(self.model, grid, **self._reward_options())
        legal = result.legal[0].copy()
        legal[list(invalid_moves)] = False
        if not legal.any():
            raise ValueError('no valid actions')
        scores = result.scores[0].clone()
        scores[~torch.as_tensor(legal)] = -torch.inf
        return int(scores.argmax().item())

    def remember(self, after_state, next_state, done):
        transition = (self._state_tensor(after_state).clone(),
                      self._state_tensor(next_state).clone(), bool(done))
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = self.config.abs_error_upper
        self.memory.add(max_priority, transition)

    @torch.no_grad()
    def _targets(self, next_states, dones):
        """Online selects r + gamma*V; target evaluates that same legal action.

        Neither the reward nor the exploratory action of the stored move belongs
        in V(x_t)'s target. Terminal post-spawn boards have value zero.
        """
        targets = torch.zeros(len(next_states), dtype=torch.float32)
        active = torch.nonzero(~dones, as_tuple=True)[0]
        if not len(active):
            return targets
        result = afterstate_scores(self.model, next_states[active], **self._reward_options())
        has_action = result.legal.any(axis=1)
        rows = np.flatnonzero(has_action)
        if len(rows):
            choices = result.scores[rows].argmax(dim=1).numpy()
            chosen_after = torch.as_tensor(result.after[rows, choices], dtype=torch.float32)
            values = self.target_model(chosen_after).reshape(-1)
            rewards = torch.as_tensor(result.rewards[rows, choices], dtype=torch.float32)
            targets[active[rows]] = rewards + self.config.gamma * values
        return targets

    def replay(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        if batch_size < 1 or self.memory.size == 0:
            raise ValueError('replay requires a positive batch size and nonempty memory')
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
        states, next_states, dones = zip(*transitions)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)
        q = self.model(states).reshape(-1)
        targets = self._targets(next_states, dones)
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
        if self.config.epsilon_schedule == 'update_exp':
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        stats['epsilon'] = self.epsilon
        return stats
