"""Training hyperparameters shared by the DQN agent and training driver."""
from dataclasses import dataclass


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    memory_capacity: int = 6000
    batch_size: int = 128
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    # Unit: replay calls that perform an optimizer update, not episodes/transitions.
    per_beta_anneal_steps: int = 100_000
    per_eps: float = 1e-5
    abs_error_upper: float = 1.0
    per_max_priority: str = 'scan'
    track_replay_age: bool = False
    epsilon_start: float = 0.15
    epsilon_min: float = 0.001
    epsilon_decay: float = 0.999
    epsilon_schedule: str = 'update_exp'
    epsilon_decay_steps: int = 200_000  # Environment steps for linear_steps.
    epsilon_decay_start_episode: int = 0
    target_update_episodes: int = 50
    grad_clip: float = 1.0
    seed: int | None = None
    arch: str = 'legacy'  # legacy, onehot_dueling, onehot_conv2x2, onehot_afterstate
    n_step: int = 1
    reward_mode: str = 'log'
    reward_scale: float = 256.0
    restart_frac: float = 0.0
    restart_pool_size: int = 5000
