import collections
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, List

import jax

from models import Agent, BayesianAgent
from utils import calculate_compression_ratio

print("jax devices", jax.devices())


import random

import scipy
import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import metrics
from brax.io import torch as brax_torch
from brax.training.agents.ppo import train as ppo
from torch import nn, optim
from tqdm import tqdm
from utils import make_legs_longer
from brax.spring.base import State, Transform, Motion


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

    def make_one_state(state: State):
        from jaxlib.xla_extension import ArrayImpl

        new_state_dict = {}
        for key in state.__dict__.keys():
            if type(state.__dict__[key]) == Transform:
                new_transform = Transform(
                    pos=state.__dict__[key].pos[0], rot=state.__dict__[key].rot[0],
                )
                new_state_dict[key] = new_transform
            elif type(state.__dict__[key]) == ArrayImpl:
                new_jnp_array = state.__dict__[key][0]
                new_state_dict[key] = new_jnp_array
            elif type(state.__dict__[key]) == Motion:
                new_motion = Motion(
                    ang=state.__dict__[key].ang[0], vel=state.__dict__[key].vel[0],
                )
                new_state_dict[key] = new_motion
            elif state.__dict__[key] == None:
                new_state_dict[key] = None
            else:
                raise ValueError("Uknown type in state class", key)
        return State(**new_state_dict)

    def extract_one_trajectory(rollout: List[State]) -> List[State]:
        new_rollout = []
        for i in range(len(rollout)):
            new_rollout.append(make_one_state(rollout[i]))
        return new_rollout

    def save_rollout_to_html(env, rollout: List[State], html_name: str = None) -> None:

        if html_name != None:
            from brax.io import html

            rollout = extract_one_trajectory(rollout=rollout)
            html_path = f"trajectory_{html_name}.html"
            html_data = html.render(sys=env.env._env.sys, states=rollout)
            with open(html_path, "w") as f:
                f.write(html_data)

    def eval_unroll(agent, env, length, html_name: str = None):
        """Return number of episodes and average reward for a single unroll."""
        observation = env.reset()
        episodes = torch.zeros((), device=agent.device)
        episode_reward = torch.zeros((), device=agent.device)
        rollout = []
        for _ in range(length):
            rollout.append(env.env._state.pipeline_state)
            _, action = agent.get_logits_action(observation)
            observation, reward, done, _ = env.step(Agent.dist_postprocess(action))
            episodes += torch.sum(done)
            episode_reward += torch.sum(reward)

        save_rollout_to_html(env, rollout=rollout, html_name=html_name)
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

    def eval_agent(agent, env, episode_length, html_name: str = None):
        t = time.time()
        with torch.no_grad():
            episode_count, episode_reward = eval_unroll(
                agent, env, episode_length, html_name
            )
        duration = time.time() - t
        episode_avg_length = env.num_envs * episode_length / episode_count
        eval_sps = env.num_envs * episode_length / duration
        return episode_reward, episode_count, episode_avg_length, eval_sps

    def make_population_histogram(episode_rewards_list, mean_reward, wandb_prefix):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=episode_rewards_list, name="histogram of sample network performance",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[mean_reward, mean_reward],
                y=[0, 100],
                mode="lines",
                name="mean network performance",
                line=dict(color="red", width=2),
            )
        )
        fig.update_layout(
            xaxis_title="episode return",
            yaxis_title="count",
            title="population histogram of network returns",
        )

        wandb.log({f"sampled_population_of_performances_{wandb_prefix}": fig})

    def eval_population(
        agent, seed, env_name, num_envs, clipping_val, learning_rate, entropy_cost,
    ):
        episode_rewards_list_normal_legs = []
        episode_rewards_list_normal_legs_sanity_check = []
        episode_rewards_list_long_legs = []
        episode_std_list_normal_legs = []
        episode_std_list_normal_legs_sanity_check = []
        episode_std_list_long_legs = []
        episode_length = 1000

        make_legs_longer(length_adjustment=1.0)
        env = envs.create(
            env_name,
            batch_size=num_envs,
            episode_length=episode_length,
            backend="spring",
        )
        env = gym_wrapper.VectorGymWrapper(env, seed=seed)
        env = torch_wrapper.TorchWrapper(env, device=agent.device)

        make_legs_longer(length_adjustment=1.2)

        env_long_legs = envs.create(
            env_name,
            batch_size=num_envs,
            episode_length=episode_length,
            backend="spring",
        )
        env_long_legs = gym_wrapper.VectorGymWrapper(env_long_legs, seed=seed)
        env_long_legs = torch_wrapper.TorchWrapper(env_long_legs, device=agent.device)

        mean_agent = agent.sample_mean_agent(clipping_val, learning_rate, entropy_cost)
        mean_agent_reward_normal_legs, _, _, _ = eval_agent(
            mean_agent, env, episode_length, html_name="short"
        )
        mean_agent_reward_long_legs, _, _, _ = eval_agent(
            mean_agent, env_long_legs, episode_length, html_name="long"
        )

        make_legs_longer(length_adjustment=1.0)

        for i in tqdm(range(100)):
            sampled_agent = agent.sample_vanilla_agent(
                clipping_val, learning_rate, entropy_cost
            )
            current_agent_returns_normal_legs = []
            current_agent_returns_normal_legs_sanity_check = []
            current_agent_returns_long_legs = []
            for _ in tqdm(range(10)):
                episode_reward, _, _, _ = eval_agent(sampled_agent, env, episode_length)
                current_agent_returns_normal_legs.append(episode_reward.cpu().numpy())

                episode_reward, _, _, _ = eval_agent(
                    sampled_agent, env_long_legs, episode_length
                )
                current_agent_returns_long_legs.append(episode_reward.cpu().numpy())

                episode_reward, _, _, _ = eval_agent(sampled_agent, env, episode_length)
                current_agent_returns_normal_legs_sanity_check.append(
                    episode_reward.cpu().numpy()
                )

            episode_rewards_list_normal_legs.append(
                np.mean(np.array(current_agent_returns_normal_legs))
            )
            episode_std_list_normal_legs.append(
                scipy.stats.sem(np.array(current_agent_returns_normal_legs))
            )
            episode_rewards_list_normal_legs_sanity_check.append(
                np.mean(np.array(current_agent_returns_normal_legs_sanity_check))
            )
            episode_std_list_normal_legs_sanity_check.append(
                np.mean(np.array(current_agent_returns_normal_legs_sanity_check))
            )
            episode_rewards_list_long_legs.append(
                np.mean(np.array(current_agent_returns_long_legs))
            )
            episode_std_list_long_legs.append(
                scipy.stats.sem(np.array(current_agent_returns_long_legs))
            )

        make_population_histogram(
            episode_rewards_list_normal_legs,
            mean_agent_reward_normal_legs.cpu().numpy(),
            wandb_prefix="normal_legs",
        )
        make_population_histogram(
            episode_rewards_list_long_legs,
            mean_agent_reward_long_legs.cpu().numpy(),
            wandb_prefix="long_legs",
        )
        make_scatter_plot(
            episode_rewards_list_normal_legs,
            episode_rewards_list_long_legs,
            episode_std_list_normal_legs,
            episode_std_list_long_legs,
            wandb_prefix="",
        )
        make_scatter_plot(
            episode_rewards_list_normal_legs,
            episode_rewards_list_normal_legs_sanity_check,
            episode_std_list_normal_legs,
            episode_std_list_normal_legs_sanity_check,
            wandb_prefix=" sanity check, ",
        )

    def make_scatter_plot(x, y, error_x_vals, error_y_vals, wandb_prefix):
        import plotly.graph_objects as go

        # Fitting a straight line
        slope, intercept = np.polyfit(x, y, 1)
        line_y = [slope * xi + intercept for xi in x]

        fig = go.Figure()

        # Scatter plot with error bars
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                error_x=dict(
                    type="data",  # Data means we provide actual values for error
                    array=error_x_vals,
                    visible=True,
                ),
                error_y=dict(type="data", array=error_y_vals, visible=True),
                mode="markers",
                name="return of normal legs vs long legs",
                marker=dict(color="red", size=3),
            )
        )

        # Straight line plot
        fig.add_trace(
            go.Scatter(x=x, y=line_y, mode="lines", name="Fit", line=dict(color="blue"))
        )

        fig.update_layout(
            xaxis_title="normal legs return", yaxis_title="long legs return",
        )

        wandb.log(
            {
                f"visualising how population ranking changes with different leg length{wandb_prefix}slope={slope:.3f}": fig
            }
        )

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
        reward_scaling: float = 10,
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

        wandb.init(
            project="brax-cshl",
            config=config,
            dir="/grid/zador/data_nlsas_norepl/mavorpar/wandb_logging/wandb",
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
                (
                    episode_reward,
                    episode_count,
                    episode_avg_length,
                    eval_sps,
                ) = eval_agent(agent, env, episode_length)
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
                progress_fn(
                    total_steps, progress, wandb_prefix, logging_population=False
                )
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

    def progress(num_steps, metrics, wandb_prefix, logging_population=True):
        if logging_population == False:
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
                    f"{wandb_prefix}_eval/episode_reward": metrics[
                        "eval/episode_reward"
                    ],
                    f"{wandb_prefix}_speed/eval_sps": metrics["speed/eval_sps"],
                    f"{wandb_prefix}_speed/sps": metrics["speed/sps"],
                },
            )
            times.append(datetime.now())
            xdata.append(num_steps)
            ydata.append(metrics["eval/episode_reward"].cpu())
            eval_sps.append(metrics["speed/eval_sps"])
            train_sps.append(metrics["speed/sps"])
        elif logging_population == True:
            wandb.log(
                {"population_eval/episode_reward": metrics["eval/episode_reward"]},
            )
        else:
            raise ValueError("logging_population must be a Boolean")

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
        eval_frequency=int(args.num_timesteps / 2e6),
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")

    print("now evaluating population statistics...")

    eval_population(
        agent=agent,
        seed=args.seed,
        env_name=args.env_name,
        num_envs=args.batch_size,
        clipping_val=args.clipping_val,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
    )

    print("now doing within lifetime learning...")

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
            eval_frequency=int(args.num_timesteps / 2e6),
        )


if __name__ == "__main__":
    import argparse
    from distutils.util import strtobool

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0)
    parser.add_argument("--learning_rate", default=3e-4)
    parser.add_argument("--entropy_cost", default=1e-2)
    parser.add_argument("--num_envs", default=4096)
    parser.add_argument("--batch_size", default=2048)
    parser.add_argument("--num_update_epochs", default=4)
    parser.add_argument("--num_minibatches", default=32)
    parser.add_argument("--unroll_length", default=5)
    parser.add_argument("--number_of_cell_types", default=64)
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
    parser.add_argument("--env_name", default="ant")
    parser.add_argument("--complexity_cost", type=float, default=0.0001)
    parser.add_argument("--num_timesteps", type=int, default=50_000_000)
    args = parser.parse_args()
    main(args)
