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
        # Separate stream so flag-off runs consume self.rng exactly as before.
        self.sym_rng = (np.random.default_rng(None if self.config.seed is None else
                                              [self.config.seed, 0x5133])
                        if self.config.sym_aug == 'random' else None)

    def _reward_options(self):
        options = dict(gamma=self.config.gamma,
                       reward_mode=self.config.reward_mode,
                       reward_scale=self.config.reward_scale)
        if self.config.value_rescale != 'none':
            options.update(value_rescale=self.config.value_rescale,
                           value_rescale_eps=self.config.value_rescale_eps)
        return options

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
        self._remember_transition(transition)

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
            if self.config.value_rescale == 'h':
                from value_rescale import h, h_inv
                eps = self.config.value_rescale_eps
                # Bootstrap in the original scale, regress h(target). Terminal
                # rows keep target 0 = h(0): no bootstrap past game over (gamma=1 safe).
                targets[active[rows]] = h(rewards + self.config.gamma * h_inv(values, eps), eps)
            else:
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
        if self.sym_rng is not None:
            # One uniformly random D4 element per sampled transition, applied to both
            # the afterstate and its successor (V is D4-invariant; no action index is
            # stored in afterstate transitions, and the target recomputes legality).
            from symmetry import apply_batch
            sym_ids = self.sym_rng.integers(0, 8, size=len(states))
            states = apply_batch(states, sym_ids)
            next_states = apply_batch(next_states, sym_ids)
        q = self.model(states).reshape(-1)
        targets = self._targets(next_states, dones)
        td = targets - q
        weight_tensor = torch.tensor(weights, dtype=q.dtype)
        loss = (weight_tensor * td.square()).mean()
        stats = dict(loss=loss.item(), q_mean=q.mean().item(), q_min=q.min().item(),
                     q_max=q.max().item(), target_mean=targets.mean().item(),
                     td_abs_mean=td.abs().mean().item(), is_weight_mean=weights.mean().item(),
                     is_weight_min=weights.min().item(), beta=self.beta, epsilon=self.epsilon)
        stats.update(self._replay_age_stats(idxs))
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
