import collections
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import jax

from models import Agent, BayesianAgent
from utils import calculate_compression_ratio

print("jax devices", jax.devices())


import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import metrics
from brax.training.agents.ppo import train as ppo
from torch import nn, optim


def main(args):
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    config = vars(args)
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
        clipping_val: float,
        seed: int,
        wandb_prefix: str,
        bayesian_agent_to_sample: None,
        is_weight_sharing: bool,
        number_of_cell_types: int,
        complexity_cost: float,
        env_name: str = "halfcheetah",
        num_envs: int = 2_048,
        episode_length: int = 1_000,
        device: str = "cuda",
        num_timesteps: int = 100_000_000,
        eval_frequency: int = 30,
        unroll_length: int = 20,
        batch_size: int = 512,
        num_minibatches: int = 32,
        num_update_epochs: int = 8,
        reward_scaling: float = 1,
        entropy_cost: float = 1e-2,
        discounting: float = 0.95,
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

        wandb.init(
            project="brax-cshl",
            config=config,
            dir="/grid/zador/data_norepl/augustine/wandb_logging",
        )

        if bayesian_agent_to_sample is not None:
            agent = bayesian_agent_to_sample.sample_vanilla_agent(
                clipping_val, learning_rate, entropy_cost
            )
        else:
            if is_weight_sharing == True:
                agent = BayesianAgent(
                    clipping_val,
                    number_of_cell_types,
                    vanilla_policy_layers,
                    vanilla_value_layers,
                    entropy_cost,
                    discounting,
                    reward_scaling,
                    device,
                    complexity_cost,
                )
            elif is_weight_sharing == False:
                agent = Agent(
                    clipping_val,
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
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_loss = 0

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
                    "losses/total_policy_loss": total_policy_loss,
                    "losses/total_value_loss": total_value_loss,
                    "losses/total_entropy_loss": total_entropy_loss,
                    "losses/total_loss": total_loss,
                }
                progress_fn(total_steps, progress, wandb_prefix)

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
                        loss, policy_loss, v_loss, entropy_loss, kl_loss = agent.loss(
                            td_minibatch._asdict()
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_policy_loss += policy_loss
                        total_value_loss += v_loss
                        total_entropy_loss += entropy_loss
                        total_loss += loss
                        total_kl_loss += kl_loss

            duration = time.time() - t
            total_steps += num_epochs * num_steps
            total_loss = total_loss / (num_epochs * num_update_epochs * num_minibatches)
            total_entropy_loss = total_entropy_loss / (
                num_epochs * num_update_epochs * num_minibatches
            )
            total_policy_loss = total_policy_loss / (
                num_epochs * num_update_epochs * num_minibatches
            )
            total_value_loss = total_value_loss / (
                num_epochs * num_update_epochs * num_minibatches
            )
            sps = num_epochs * num_steps / duration
        return agent

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    xdata = []
    ydata = []
    eval_sps = []
    train_sps = []
    times = [datetime.now()]

    def progress(num_steps, metrics, wandb_prefix):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"].cpu())
        eval_sps.append(metrics["speed/eval_sps"])
        train_sps.append(metrics["speed/sps"])
        wandb.log(
            {
                f"{wandb_prefix}_losses/total_loss": metrics["losses/total_loss"],
                f"{wandb_prefix}_losses/total_policy_loss": metrics[
                    "losses/total_policy_loss"
                ],
                f"{wandb_prefix}_losses/total_value_loss": metrics[
                    "losses/total_value_loss"
                ],
                f"{wandb_prefix}_losses/total_entropy_loss": metrics[
                    "losses/total_entropy_loss"
                ],
                f"{wandb_prefix}_eval/episode_reward": metrics["eval/episode_reward"],
                f"{wandb_prefix}_speed/eval_sps": metrics["speed/eval_sps"],
                f"{wandb_prefix}_speed/sps": metrics["speed/sps"],
            },
        )

    agent = train(
        clipping_val=(args.clipping_val),
        wandb_prefix="bayesian",
        bayesian_agent_to_sample=None,
        env_name=args.env_name,
        is_weight_sharing=args.is_weight_sharing,
        number_of_cell_types=args.number_of_cell_types,
        progress_fn=progress,
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        entropy_cost=float(args.entropy_cost),
        complexity_cost=float(args.complexity_cost),
        num_timesteps=int(args.num_timesteps),
        eval_frequency=int(args.eval_frequency),
        reward_scaling=float(args.reward_scaling),
        episode_length=int(args.episode_length),
        discounting=float(args.discounting),
        num_update_epochs=int(args.num_updates_per_batch),
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")

    # print("now doing within lifetime learning...")

    if args.is_weight_sharing != False:
        agent = train(
            clipping_val=(args.within_lifetime_clipping_val),
            wandb_prefix="within_lifeteime_learning",
            bayesian_agent_to_sample=agent,
            env_name=args.env_name,
            is_weight_sharing=args.is_weight_sharing,
            number_of_cell_types=args.number_of_cell_types,
            progress_fn=progress,
            seed=int(args.seed),
            num_envs=int(args.num_envs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.within_lifetime_learning_rate),
            entropy_cost=float(args.within_lifetime_entropy_cost),
            complexity_cost=float(args.complexity_cost),
            num_timesteps=int(args.num_timesteps),
            eval_frequency=int(args.eval_frequency),
            reward_scaling=float(args.reward_scaling),
            episode_length=int(args.episode_length),
            discounting=float(args.discounting),
            num_update_epochs=int(args.num_updates_per_batch),
        )


if __name__ == "__main__":
    import argparse
    from distutils.util import strtobool

    parser = argparse.ArgumentParser()

    # Existing arguments
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--entropy_cost", default=1e-2, type=float)
    parser.add_argument("--num_envs", default=2048, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--num_update_epochs", default=2, type=int)
    parser.add_argument("--num_minibatches", default=32, type=int)
    parser.add_argument("--unroll_length", default=5, type=int)
    parser.add_argument("--number_of_cell_types", default=64, type=int)
    parser.add_argument(
        "--is_weight_sharing",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument("--clipping_val", default=0.3, type=float)
    parser.add_argument("--within_lifetime_clipping_val", default=0.3, type=float)
    parser.add_argument("--within_lifetime_learning_rate", default=3e-4, type=float)
    parser.add_argument("--within_lifetime_entropy_cost", default=1e-2, type=float)
    parser.add_argument("--env_name", default="halfcheetah", type=str)
    parser.add_argument("--complexity_cost", type=float, default=1.0)
    parser.add_argument("--num_timesteps", default=100000000, type=int)
    parser.add_argument("--eval_frequency", default=30, type=int)
    parser.add_argument("--reward_scaling", default=10, type=float)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--num_updates_per_batch", default=4, type=int)
    parser.add_argument("--discounting", default=0.97, type=float)

    args = parser.parse_args()
    main(args)
