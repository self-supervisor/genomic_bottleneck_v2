# from utils import *
from Custom_layers import BayesianLinear
from typing import List


def calculate_parameters(mlp):
    # The number of parameters between two layers is the product of the
    # number of neurons in these two layers, plus the number of neurons
    # in the current layer (for bias terms).
    # There are no parameters for the first layer, so we start from the second layer.
    return sum([mlp[i - 1] * mlp[i] + mlp[i] for i in range(1, len(mlp))])


def calculate_compression_ratio(
    env,
    vanilla_policy_layers: List[int],
    vanilla_value_layers: List[int],
    number_of_cell_types: int,
) -> float:
    number_of_vanilla_policy_parameters = calculate_parameters(vanilla_policy_layers)
    number_of_vanilla_value_parameters = calculate_parameters(vanilla_value_layers)
    total_vanilla_parameters = (
        number_of_vanilla_policy_parameters + number_of_vanilla_value_parameters
    )
    compressed_policy_layers = [
        env.observation_space.shape[-1],
        number_of_cell_types,
        number_of_cell_types,
        1,
    ]
    compressed_value_layers = [
        env.observation_space.shape[-1],
        number_of_cell_types,
        number_of_cell_types,
        1,
    ]
    number_of_compressed_policy_parameters = calculate_parameters(
        compressed_policy_layers
    )
    number_of_compressed_value_parameters = calculate_parameters(
        compressed_value_layers
    )
    total_compressed_parameters = (
        number_of_compressed_policy_parameters + number_of_compressed_value_parameters
    )
    return total_vanilla_parameters / total_compressed_parameters


def main(args):
    # Load the data
    # from IPython.display import clear_output

    import collections
    from datetime import datetime
    import functools
    import math
    import os
    import time
    from typing import Any, Callable, Dict, Optional, Sequence
    import jax

    print("jax devices", jax.devices())

    import brax

    from brax import envs
    from brax.envs.wrappers import gym as gym_wrapper
    from brax.envs.wrappers import torch as torch_wrapper
    from brax.io import metrics
    from brax.training.agents.ppo import train as ppo
    import gym
    import numpy as np
    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    import wandb
    import random
    from blitz.losses import kl_divergence_from_nn
    from blitz.utils import variational_estimator

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    config = vars(args)
    # Here is a PPO Agent written in PyTorch:

    # In[2]:

    class Agent(nn.Module):
        """Standard PPO Agent with GAE and observation normalization."""

        def __init__(
            self,
            policy_layers: Sequence[int],
            value_layers: Sequence[int],
            entropy_cost: float,
            discounting: float,
            reward_scaling: float,
            device: str,
        ):
            super(Agent, self).__init__()

            policy = []
            for w1, w2 in zip(policy_layers, policy_layers[1:]):
                policy.append(nn.Linear(w1, w2),)
                policy.append(nn.SiLU())
            policy.pop()  # drop the final activation
            self.policy = nn.Sequential(*policy)

            value = []
            for w1, w2 in zip(value_layers, value_layers[1:]):
                value.append(nn.Linear(w1, w2),)
                value.append(nn.SiLU())
            value.pop()  # drop the final activation
            self.value = nn.Sequential(*value)

            self.num_steps = torch.zeros((), device=device)
            self.running_mean = torch.zeros(policy_layers[0], device=device)
            self.running_variance = torch.zeros(policy_layers[0], device=device)

            self.entropy_cost = entropy_cost
            self.discounting = discounting
            self.reward_scaling = reward_scaling
            self.lambda_ = 0.95
            self.epsilon = 0.3
            self.device = device

        @torch.jit.export
        def dist_create(self, logits):
            """Normal followed by tanh.

            torch.distribution doesn't work with torch.jit, so we roll our own."""
            loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
            scale = F.softplus(scale) + 0.001
            return loc, scale

        @torch.jit.export
        def dist_sample_no_postprocess(self, loc, scale):
            return torch.normal(loc, scale)

        @classmethod
        def dist_postprocess(cls, x):
            return torch.tanh(x)

        @torch.jit.export
        def dist_entropy(self, loc, scale):
            log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
            entropy = 0.5 + log_normalized
            entropy = entropy * torch.ones_like(loc)
            dist = torch.normal(loc, scale)
            log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
            entropy = entropy + log_det_jacobian
            return entropy.sum(dim=-1)

        @torch.jit.export
        def dist_log_prob(self, loc, scale, dist):
            log_unnormalized = -0.5 * ((dist - loc) / scale).square()
            log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
            log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
            log_prob = log_unnormalized - log_normalized - log_det_jacobian
            return log_prob.sum(dim=-1)

        @torch.jit.export
        def update_normalization(self, observation):
            self.num_steps += observation.shape[0] * observation.shape[1]
            input_to_old_mean = observation - self.running_mean
            mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
            self.running_mean = self.running_mean + mean_diff
            input_to_new_mean = observation - self.running_mean
            var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
            self.running_variance = self.running_variance + var_diff

        @torch.jit.export
        def normalize(self, observation):
            variance = self.running_variance / (self.num_steps + 1.0)
            variance = torch.clip(variance, 1e-6, 1e6)
            return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

        @torch.jit.export
        def get_logits_action(self, observation):
            observation = self.normalize(observation)
            logits = self.policy(observation)
            loc, scale = self.dist_create(logits)
            action = self.dist_sample_no_postprocess(loc, scale)
            return logits, action

        @torch.jit.export
        def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
            truncation_mask = 1 - truncation
            # Append bootstrapped value to get [v1, ..., v_t+1]
            values_t_plus_1 = torch.cat(
                [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
            )
            deltas = (
                reward + self.discounting * (1 - termination) * values_t_plus_1 - values
            )
            deltas *= truncation_mask

            acc = torch.zeros_like(bootstrap_value)
            vs_minus_v_xs = torch.zeros_like(truncation_mask)

            for ti in range(truncation_mask.shape[0]):
                ti = truncation_mask.shape[0] - ti - 1
                acc = (
                    deltas[ti]
                    + self.discounting
                    * (1 - termination[ti])
                    * truncation_mask[ti]
                    * self.lambda_
                    * acc
                )
                vs_minus_v_xs[ti] = acc

            # Add V(x_s) to get v_s.
            vs = vs_minus_v_xs + values
            vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
            advantages = (
                reward + self.discounting * (1 - termination) * vs_t_plus_1 - values
            ) * truncation_mask
            return vs, advantages

        @torch.jit.export
        def loss(self, td: Dict[str, torch.Tensor]):
            observation = self.normalize(td["observation"])
            policy_logits = self.policy(observation[:-1])
            baseline = self.value(observation)
            baseline = torch.squeeze(baseline, dim=-1)

            # Use last baseline value (from the value function) to bootstrap.
            bootstrap_value = baseline[-1]
            baseline = baseline[:-1]
            reward = td["reward"] * self.reward_scaling
            termination = td["done"] * (1 - td["truncation"])

            loc, scale = self.dist_create(td["logits"])
            behaviour_action_log_probs = self.dist_log_prob(loc, scale, td["action"])
            loc, scale = self.dist_create(policy_logits)
            target_action_log_probs = self.dist_log_prob(loc, scale, td["action"])

            with torch.no_grad():
                vs, advantages = self.compute_gae(
                    truncation=td["truncation"],
                    termination=termination,
                    reward=reward,
                    values=baseline,
                    bootstrap_value=bootstrap_value,
                )

            rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
            surrogate_loss1 = rho_s * advantages
            surrogate_loss2 = (
                rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
            )
            policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

            # Value function loss
            v_error = vs - baseline
            v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

            # Entropy reward
            entropy = torch.mean(self.dist_entropy(loc, scale))
            entropy_loss = self.entropy_cost * -entropy

            return policy_loss + v_loss + entropy_loss

    class BayesianAgent(nn.Module):
        """Standard PPO Agent with GAE and observation normalization."""

        def __init__(
            self,
            number_of_cell_types: int,
            policy_layers: Sequence[int],
            value_layers: Sequence[int],
            entropy_cost: float,
            discounting: float,
            reward_scaling: float,
            device: str,
        ):
            super(Agent, self).__init__()

            policy = []
            for w1, w2 in zip(policy_layers, policy_layers[1:]):
                policy.append(
                    BayesianLinear(
                        in_features=w1,
                        out_features=w2,
                        neuron_types_in=number_of_cell_types,
                        neuron_types_out=number_of_cell_types,
                    )
                )
                policy.append(nn.SiLU())
            policy.pop()  # drop the final activation
            self.policy = nn.Sequential(*policy)

            value = []
            for w1, w2 in zip(value_layers, value_layers[1:]):
                value.append(
                    BayesianLinear(
                        in_features=w1,
                        out_features=w2,
                        neuron_types_in=number_of_cell_types,
                        neuron_types_out=number_of_cell_types,
                    )
                )
                value.append(nn.SiLU())
            value.pop()  # drop the final activation
            self.value = nn.Sequential(*value)

            self.num_steps = torch.zeros((), device=device)
            self.running_mean = torch.zeros(policy_layers[0], device=device)
            self.running_variance = torch.zeros(policy_layers[0], device=device)

            self.entropy_cost = entropy_cost
            self.discounting = discounting
            self.reward_scaling = reward_scaling
            self.lambda_ = 0.95
            self.epsilon = 0.3
            self.device = device

        @torch.jit.export
        def dist_create(self, logits):
            """Normal followed by tanh.

            torch.distribution doesn't work with torch.jit, so we roll our own."""
            loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
            scale = F.softplus(scale) + 0.001
            return loc, scale

        @torch.jit.export
        def dist_sample_no_postprocess(self, loc, scale):
            return torch.normal(loc, scale)

        @classmethod
        def dist_postprocess(cls, x):
            return torch.tanh(x)

        @torch.jit.export
        def dist_entropy(self, loc, scale):
            log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
            entropy = 0.5 + log_normalized
            entropy = entropy * torch.ones_like(loc)
            dist = torch.normal(loc, scale)
            log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
            entropy = entropy + log_det_jacobian
            return entropy.sum(dim=-1)

        @torch.jit.export
        def dist_log_prob(self, loc, scale, dist):
            log_unnormalized = -0.5 * ((dist - loc) / scale).square()
            log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
            log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
            log_prob = log_unnormalized - log_normalized - log_det_jacobian
            return log_prob.sum(dim=-1)

        @torch.jit.export
        def update_normalization(self, observation):
            self.num_steps += observation.shape[0] * observation.shape[1]
            input_to_old_mean = observation - self.running_mean
            mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
            self.running_mean = self.running_mean + mean_diff
            input_to_new_mean = observation - self.running_mean
            var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
            self.running_variance = self.running_variance + var_diff

        @torch.jit.export
        def normalize(self, observation):
            variance = self.running_variance / (self.num_steps + 1.0)
            variance = torch.clip(variance, 1e-6, 1e6)
            return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

        @torch.jit.export
        def get_logits_action(self, observation):
            observation = self.normalize(observation)
            logits = self.policy(observation)
            loc, scale = self.dist_create(logits)
            action = self.dist_sample_no_postprocess(loc, scale)
            return logits, action

        @torch.jit.export
        def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
            truncation_mask = 1 - truncation
            # Append bootstrapped value to get [v1, ..., v_t+1]
            values_t_plus_1 = torch.cat(
                [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
            )
            deltas = (
                reward + self.discounting * (1 - termination) * values_t_plus_1 - values
            )
            deltas *= truncation_mask

            acc = torch.zeros_like(bootstrap_value)
            vs_minus_v_xs = torch.zeros_like(truncation_mask)

            for ti in range(truncation_mask.shape[0]):
                ti = truncation_mask.shape[0] - ti - 1
                acc = (
                    deltas[ti]
                    + self.discounting
                    * (1 - termination[ti])
                    * truncation_mask[ti]
                    * self.lambda_
                    * acc
                )
                vs_minus_v_xs[ti] = acc

            # Add V(x_s) to get v_s.
            vs = vs_minus_v_xs + values
            vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
            advantages = (
                reward + self.discounting * (1 - termination) * vs_t_plus_1 - values
            ) * truncation_mask
            return vs, advantages

        @torch.jit.export
        def loss(self, td: Dict[str, torch.Tensor]):
            observation = self.normalize(td["observation"])
            policy_logits = self.policy(observation[:-1])
            baseline = self.value(observation)
            baseline = torch.squeeze(baseline, dim=-1)

            # Use last baseline value (from the value function) to bootstrap.
            bootstrap_value = baseline[-1]
            baseline = baseline[:-1]
            reward = td["reward"] * self.reward_scaling
            termination = td["done"] * (1 - td["truncation"])

            loc, scale = self.dist_create(td["logits"])
            behaviour_action_log_probs = self.dist_log_prob(loc, scale, td["action"])
            loc, scale = self.dist_create(policy_logits)
            target_action_log_probs = self.dist_log_prob(loc, scale, td["action"])

            with torch.no_grad():
                vs, advantages = self.compute_gae(
                    truncation=td["truncation"],
                    termination=termination,
                    reward=reward,
                    values=baseline,
                    bootstrap_value=bootstrap_value,
                )

            rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
            surrogate_loss1 = rho_s * advantages
            surrogate_loss2 = (
                rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
            )
            policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

            # Value function loss
            v_error = vs - baseline
            v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

            # Entropy reward
            entropy = torch.mean(self.dist_entropy(loc, scale))
            entropy_loss = self.entropy_cost * -entropy

            return policy_loss + v_loss + entropy_loss

    StepData = collections.namedtuple(
        "StepData", ("observation", "logits", "action", "reward", "done", "truncation")
    )

    def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
        """Map a function over each field in StepData."""
        items = {}
        keys = sds[0]._asdict().keys()
        for k in keys:
            items[k] = f(*[sd._asdict()[k] for sd in sds])
        return StepData(**items)

    def eval_unroll(agent, env, length):
        """Return number of episodes and average reward for a single unroll."""
        observation = env.reset()
        episodes = torch.zeros((), device=agent.device)
        episode_reward = torch.zeros((), device=agent.device)
        for _ in range(length):
            _, action = agent.get_logits_action(observation)
            observation, reward, done, _ = env.step(Agent.dist_postprocess(action))
            episodes += torch.sum(done)
            episode_reward += torch.sum(reward)
        return episodes, episode_reward / episodes

    def train_unroll(agent, env, observation, num_unrolls, unroll_length):
        """Return step data over multple unrolls."""
        sd = StepData([], [], [], [], [], [])
        for _ in range(num_unrolls):
            one_unroll = StepData([observation], [], [], [], [], [])
            for _ in range(unroll_length):
                logits, action = agent.get_logits_action(observation)
                observation, reward, done, info = env.step(
                    Agent.dist_postprocess(action)
                )
                one_unroll.observation.append(observation)
                one_unroll.logits.append(logits)
                one_unroll.action.append(action)
                one_unroll.reward.append(reward)
                one_unroll.done.append(done)
                one_unroll.truncation.append(info["truncation"])
            one_unroll = sd_map(torch.stack, one_unroll)
            sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
        td = sd_map(torch.stack, sd)
        return observation, td

    def train(
        seed,
        is_weight_sharing: bool,
        number_of_cell_types: int,
        env_name: str = "ant",
        num_envs: int = 2048,
        episode_length: int = 1000,
        device: str = "cuda",
        num_timesteps: int = 30_000_000,
        eval_frequency: int = 10,
        unroll_length: int = 5,
        batch_size: int = 1024,
        num_minibatches: int = 32,
        num_update_epochs: int = 4,
        reward_scaling: float = 0.1,
        entropy_cost: float = 1e-2,
        discounting: float = 0.97,
        learning_rate: float = 3e-4,
        progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ):
        """Trains a policy via PPO."""
        number_of_cell_types = int(number_of_cell_types)
        env = envs.create(
            env_name,
            batch_size=num_envs,
            episode_length=episode_length,
            backend="spring",
        )
        env = gym_wrapper.VectorGymWrapper(env, seed=seed)
        # automatically convert between jax ndarrays and torch tensors:
        env = torch_wrapper.TorchWrapper(env, device=device)

        # env warmup
        env.reset()
        action = torch.zeros(env.action_space.shape).to(device)
        env.step(action)

        # create the agent
        vanilla_policy_layers = [
            env.observation_space.shape[-1],
            64,
            64,
            env.action_space.shape[-1] * 2,
        ]
        vanilla_value_layers = [env.observation_space.shape[-1], 64, 64, 1]
        compression_ratio = calculate_compression_ratio(
            env,
            vanilla_policy_layers,
            vanilla_value_layers,
            number_of_cell_types=number_of_cell_types,
        )
        config["compression_ratio"] = compression_ratio

        wandb.init(project="brax-cshl", config=config)
        if is_weight_sharing == True:
            agent = BayesianAgent(
                number_of_cell_types,
                vanilla_policy_layers,
                vanilla_value_layers,
                entropy_cost,
                discounting,
                reward_scaling,
                device,
            )
        elif is_weight_sharing == False:
            agent = Agent(
                vanilla_policy_layers,
                vanilla_value_layers,
                entropy_cost,
                discounting,
                reward_scaling,
                device,
            )

        agent = agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

        sps = 0
        total_steps = 0
        total_loss = 0
        for eval_i in range(eval_frequency + 1):
            if progress_fn:
                t = time.time()
                with torch.no_grad():
                    episode_count, episode_reward = eval_unroll(
                        agent, env, episode_length
                    )
                duration = time.time() - t
                # TODO: only count stats from completed episodes
                episode_avg_length = env.num_envs * episode_length / episode_count
                eval_sps = env.num_envs * episode_length / duration
                progress = {
                    "eval/episode_reward": episode_reward,
                    "eval/completed_episodes": episode_count,
                    "eval/avg_episode_length": episode_avg_length,
                    "speed/sps": sps,
                    "speed/eval_sps": eval_sps,
                    "losses/total_loss": total_loss,
                }
                progress_fn(total_steps, progress)

            if eval_i == eval_frequency:
                break

            observation = env.reset()
            num_steps = batch_size * num_minibatches * unroll_length
            num_epochs = num_timesteps // (num_steps * eval_frequency)
            num_unrolls = batch_size * num_minibatches // env.num_envs
            total_loss = 0
            t = time.time()
            for _ in range(num_epochs):
                observation, td = train_unroll(
                    agent, env, observation, num_unrolls, unroll_length
                )

                # make unroll first
                def unroll_first(data):
                    data = data.swapaxes(0, 1)
                    return data.reshape([data.shape[0], -1] + list(data.shape[3:]))

                td = sd_map(unroll_first, td)

                # update normalization statistics
                agent.update_normalization(td.observation)

                for _ in range(num_update_epochs):
                    # shuffle and batch the data
                    with torch.no_grad():
                        permutation = torch.randperm(
                            td.observation.shape[1], device=device
                        )

                        def shuffle_batch(data):
                            data = data[:, permutation]
                            data = data.reshape(
                                [data.shape[0], num_minibatches, -1]
                                + list(data.shape[2:])
                            )
                            return data.swapaxes(0, 1)

                        epoch_td = sd_map(shuffle_batch, td)

                    for minibatch_i in range(num_minibatches):
                        td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td)
                        loss = agent.loss(td_minibatch._asdict())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss

            duration = time.time() - t
            total_steps += num_epochs * num_steps
            total_loss = total_loss / (num_epochs * num_update_epochs * num_minibatches)
            sps = num_epochs * num_steps / duration

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    xdata = []
    ydata = []
    eval_sps = []
    train_sps = []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"].cpu())
        eval_sps.append(metrics["speed/eval_sps"])
        train_sps.append(metrics["speed/sps"])

        wandb.log(
            {
                "eval/episode_reward": metrics["eval/episode_reward"].cpu(),
                "speed/eval_sps": metrics["speed/eval_sps"],
                "speed/sps": metrics["speed/sps"],
            },
        )

    train(
        env_name=args.env_name,
        is_weight_sharing=args.is_weight_sharing,
        number_of_cell_types=args.number_of_cell_types,
        progress_fn=progress,
        seed=int(args.seed),
        num_envs=int(args.number_envs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        entropy_cost=float(args.entropy_cost),
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")


if __name__ == "__main__":
    import argparse
    from distutils.util import strtobool

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0)
    parser.add_argument("--learning_rate", default=3e-4)
    parser.add_argument("--entropy_cost", default=1e-2)
    parser.add_argument("--number_envs", default=2048)
    parser.add_argument("--batch_size", default=1024)
    parser.add_argument("--number_of_cell_types", default=5)
    parser.add_argument(
        "--is_weight_sharing",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument("--env_name", default="ant")
    args = parser.parse_args()
    main(args)
