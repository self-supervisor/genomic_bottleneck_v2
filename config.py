from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class TrainConfig:
    clipping_val: float
    seed: int
    wandb_prefix: str
    bayesian_agent_to_sample: Optional[Any]
    is_weight_sharing: bool
    number_of_cell_types: int
    complexity_cost: float
    env_name: str
    num_envs: int
    episode_length: int
    device: str
    num_timesteps: int
    eval_frequency: int
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_update_epochs: int
    reward_scaling: float
    entropy_cost: float
    discounting: float
    learning_rate: float
