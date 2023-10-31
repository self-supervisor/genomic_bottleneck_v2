import collections
import os
import time
from datetime import datetime
from typing import Callable, List, Tuple, Union

import jax

print("jax devices", jax.devices())
import random

import hydra
import numpy as np
import scipy
import torch
import wandb
from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from config import TrainConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch import optim
from tqdm import tqdm

from models import Agent, BayesianAgent
from utils import calculate_compression_ratio

cs = ConfigStore.instance()
cs.store(name="default", node=TrainConfig)
from brax.spring.base import Motion, State, Transform

from utils import make_legs_longer


def write_to_csv(*, cfg_to_log: dict, path: str = "csv_logs/") -> None:
    """Write the config and final reward to a csv file."""
    if not os.path.exists(path):
        os.makedirs(path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(path, f"{timestamp}.csv")

    assert not os.path.exists(file_path)

    with open(file_path, "a") as f:
        f.write(",".join([str(k) for k in cfg_to_log.keys()]) + "\n")
        f.write(",".join([str(v) for v in cfg_to_log.values()]) + "\n")


@hydra.main(config_path="configs", config_name="ant")
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    dict_config = OmegaConf.to_container(cfg, resolve=True)
    cfg_to_log = dict(cfg)

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

    def make_one_state(state: State) -> State:
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

    def eval_unroll(
        agent: Union[BayesianAgent, Agent], env, length: int, html_name: str = None
    ) -> Tuple[int, float]:
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

    def train_unroll(
        agent: Union[BayesianAgent, Agent],
        env,
        observation: torch.Tensor,
        num_unrolls: int,
        unroll_length: int,
    ) -> Tuple[torch.Tensor]:
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

    def eval_agent(
        agent: Union[BayesianAgent, Agent],
        env,
        episode_length: int,
        html_name: str = None,
    ) -> Tuple[float]:
        t = time.time()
        with torch.no_grad():
            episode_count, episode_reward = eval_unroll(
                agent, env, episode_length, html_name
            )
        duration = time.time() - t
        episode_avg_length = env.num_envs * episode_length / episode_count
        eval_sps = env.num_envs * episode_length / duration
        return episode_reward, episode_count, episode_avg_length, eval_sps

    def make_population_histogram(
        episode_rewards_list: List[float], mean_reward: float, wandb_prefix: str
    ) -> None:
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
        agent: Union[BayesianAgent, Agent],
        seed: int,
        env_name: str,
        num_envs: int,
        clipping_val: float,
        learning_rate: float,
        entropy_cost: float,
        networks_to_sample: int,
        samples_to_take: int,
    ) -> None:
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

        for i in tqdm(range(networks_to_sample)):
            sampled_agent = agent.sample_vanilla_agent(
                clipping_val, learning_rate, entropy_cost
            )
            current_agent_returns_normal_legs = []
            current_agent_returns_normal_legs_sanity_check = []
            current_agent_returns_long_legs = []
            for _ in tqdm(range(samples_to_take)):
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

    def make_scatter_plot(
        x: List[float],
        y: List[float],
        error_x_vals: List[float],
        error_y_vals: List[float],
        wandb_prefix: str,
    ):
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
        *,
        cfg: DictConfig,
        bayesian_agent_to_sample: BayesianAgent,
        progress_fn: Callable,
        wandb_prefix: str,
    ):
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
            dir="/grid/zador/mavorpar/wandb_logging",
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
            else:
                raise ValueError(
                    f"is_weight_sharing must be True or False but is {cfg.is_weight_sharing}"
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
            if progress_fn:
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
                progress_fn(total_steps, progress, wandb_prefix)

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
        if "within_lifetime" not in wandb_prefix:
            percentage_of_SOTA_reward = (
                (
                    (episode_reward - cfg.min_performance)
                    / (cfg.SOTA_performance - cfg.min_performance)
                )
                .detach()
                .cpu()
                .numpy()
                .item()
            )
            wandb.log(
                {f"{wandb_prefix}_percentage_of_SOTA_reward": percentage_of_SOTA_reward}
            )
            cfg_to_log["compression_ratio"] = compression_ratio
        else:
            percentage_of_SOTA_reward = None

        return agent, num_of_params, percentage_of_SOTA_reward

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    xdata = []
    ydata = []
    eval_sps = []
    train_sps = []
    times = [datetime.now()]

    def progress(num_steps: str, metrics: dict, wandb_prefix: str):
        times.append(datetime.now())
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
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"].cpu())
        eval_sps.append(metrics["speed/eval_sps"])
        train_sps.append(metrics["speed/sps"])

    agent, num_params_evolutionary, percentage_of_SOTA_reward = train(
        cfg=cfg,
        bayesian_agent_to_sample=None,
        progress_fn=progress,
        wandb_prefix="evolutionary_learning",
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")

    if cfg.eval_population:
        print("now evaluating population statistics...")
        if cfg.is_weight_sharing == True:
            eval_population(
                agent=agent,
                seed=cfg.seed,
                env_name=cfg.env_name,
                num_envs=cfg.batch_size,
                clipping_val=cfg.clipping_val,
                learning_rate=cfg.learning_rate,
                entropy_cost=cfg.entropy_cost,
                networks_to_sample=cfg.networks_to_sample,
                samples_to_take=cfg.samples_to_take,
            )

    if cfg.is_weight_sharing != False:
        cfg.num_timesteps = cfg.num_timesteps * 2
        cfg.eval_frequency = cfg.eval_frequency * 2
        agent, num_params_within_lifetime, _ = train(
            cfg=cfg,
            bayesian_agent_to_sample=agent,
            progress_fn=progress,
            wandb_prefix="within_lifetime_learning",
        )

    if cfg.is_weight_sharing:
        wandb.log(
            {
                "pytorch_reported_compression": num_params_within_lifetime
                / num_params_evolutionary
            }
        )

    cfg_to_log["proportion_of_max_score"] = percentage_of_SOTA_reward
    cfg_to_log["num_params_evolutionary"] = num_params_evolutionary
    write_to_csv(cfg_to_log=cfg_to_log)


if __name__ == "__main__":
    main()
