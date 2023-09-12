import collections
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import jax

from models import Agent, BayesianAgent
from utils import calculate_compression_ratio
from omegaconf import OmegaConf

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
import hydra
from hydra.core.config_store import ConfigStore

from omegaconf import DictConfig
from config import TrainConfig

cs = ConfigStore.instance()
cs.store(name="default", node=TrainConfig)


@hydra.main(config_path=".", config_name="ant")
def main(cfg: DictConfig):
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    dict_config = OmegaConf.to_container(cfg, resolve=True)

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

    def train(cfg, bayesian_agent_to_sample, progress_function, wandb_prefix):
        """Trains a policy via PPO."""
        number_of_cell_types = int(cfg.number_of_cell_types)
        env = envs.create(
            cfg.env_name,
            batch_size=cfg.num_envs,
            episode_length=cfg.episode_length,
            backend="spring",
        )
        env = gym_wrapper.VectorGymWrapper(env, seed=cfg.seed)
        # automatically convert between jax ndarrays and torch tensors:
        env = torch_wrapper.TorchWrapper(env, device=cfg.device)

        # env warmup
        env.reset()
        action = torch.zeros(env.action_space.shape).to(cfg.device)
        env.step(action)

        # create the agent
        vanilla_policy_layers = [
            env.observation_space.shape[-1],
            cfg.hidden_size,
            cfg.hidden_size,
            env.action_space.shape[-1] * 2,
        ]
        vanilla_value_layers = [
            env.observation_space.shape[-1],
            cfg.hidden_size,
            cfg.hidden_size,
            1,
        ]
        compression_ratio = calculate_compression_ratio(
            env,
            vanilla_policy_layers,
            vanilla_value_layers,
            number_of_cell_types=number_of_cell_types,
        )
        dict_config["compression_ratio"] = compression_ratio

        wandb.init(
            project="brax-cshl",
            config=dict_config,
            dir="/grid/zador/data_nlsas_norepl/augustine/wandb_logging",
        )

        if bayesian_agent_to_sample is not None:
            agent = bayesian_agent_to_sample.sample_vanilla_agent(
                cfg.clipping_val, cfg.learning_rate, cfg.entropy_cost
            )
        else:
            if cfg.is_weight_sharing == True:
                agent = BayesianAgent(
                    cfg.clipping_val,
                    number_of_cell_types,
                    vanilla_policy_layers,
                    vanilla_value_layers,
                    cfg.entropy_cost,
                    cfg.discounting,
                    cfg.reward_scaling,
                    cfg.device,
                    cfg.complexity_cost,
                )
            elif cfg.is_weight_sharing == False:
                agent = Agent(
                    cfg.clipping_val,
                    vanilla_policy_layers,
                    vanilla_value_layers,
                    cfg.entropy_cost,
                    cfg.discounting,
                    cfg.reward_scaling,
                    cfg.device,
                )

        agent = agent.to(cfg.device)
        num_of_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

        optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate)

        sps = 0
        total_steps = 0
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_loss = 0

        for eval_i in range(cfg.eval_frequency + 1):
            if progress_function:
                t = time.time()
                with torch.no_grad():
                    episode_count, episode_reward = eval_unroll(
                        agent, env, cfg.episode_length
                    )
                duration = time.time() - t
                # TODO: only count stats from completed episodes
                episode_avg_length = env.num_envs * cfg.episode_length / episode_count
                eval_sps = env.num_envs * cfg.episode_length / duration
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
                progress_function(total_steps, progress, wandb_prefix)

            if eval_i == cfg.eval_frequency:
                break

            observation = env.reset()
            num_steps = cfg.batch_size * cfg.num_minibatches * cfg.unroll_length
            num_epochs = cfg.num_timesteps // (num_steps * cfg.eval_frequency)
            num_unrolls = cfg.batch_size * cfg.num_minibatches // env.num_envs
            total_loss = 0
            t = time.time()
            for _ in range(num_epochs):
                observation, td = train_unroll(
                    agent, env, observation, num_unrolls, cfg.unroll_length
                )

                # make unroll first
                def unroll_first(data):
                    data = data.swapaxes(0, 1)
                    return data.reshape([data.shape[0], -1] + list(data.shape[3:]))

                td = sd_map(unroll_first, td)

                # update normalization statistics
                agent.update_normalization(td.observation)

                for _ in range(cfg.num_update_epochs):
                    # shuffle and batch the data
                    with torch.no_grad():
                        permutation = torch.randperm(
                            td.observation.shape[1], device=cfg.device
                        )

                        def shuffle_batch(data):
                            data = data[:, permutation]
                            data = data.reshape(
                                [data.shape[0], cfg.num_minibatches, -1]
                                + list(data.shape[2:])
                            )
                            return data.swapaxes(0, 1)

                        epoch_td = sd_map(shuffle_batch, td)

                    for minibatch_i in range(cfg.num_minibatches):
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
            total_loss = total_loss / (
                num_epochs * cfg.num_update_epochs * cfg.num_minibatches
            )
            total_entropy_loss = total_entropy_loss / (
                num_epochs * cfg.num_update_epochs * cfg.num_minibatches
            )
            total_policy_loss = total_policy_loss / (
                num_epochs * cfg.num_update_epochs * cfg.num_minibatches
            )
            total_value_loss = total_value_loss / (
                num_epochs * cfg.num_update_epochs * cfg.num_minibatches
            )
            sps = num_epochs * num_steps / duration
            proportion_of_max_score = (
                (episode_reward - cfg.min_performance)
                / (cfg.SOTA_performance - cfg.min_performance),
            )
        wandb.log(
            {f"{wandb_prefix}_percentage_of_SOTA_reward": proportion_of_max_score}
        )
        return agent, num_of_params

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

    agent, num_params_evolutionary = train(
        cfg,
        bayesian_agent_to_sample=None,
        progress_function=progress,
        wandb_prefix="evolutionary_learning",
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")

    print("now doing within lifetime learning...")

    if cfg.is_weight_sharing != False:
        cfg.num_timesteps = cfg.num_timesteps * 2
        cfg.eval_frequency = cfg.eval_frequency * 2
        agent, num_params_within_lifetime = train(
            cfg,
            bayesian_agent_to_sample=agent,
            progress_function=progress,
            wandb_prefix="within_lifetime_learning",
        )

    wandb.log(
        {
            "pytorch_reported_compression": num_params_within_lifetime
            / num_params_evolutionary
        }
    )


if __name__ == "__main__":
    main()
