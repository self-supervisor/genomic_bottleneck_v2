#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--LFF_scale", type=float, default=0.0001)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env_name", type=str, default="ant")
parser.add_argument("--reward_noise", type=float, default=0.0)
args = parser.parse_args()

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


def default_mlp_init():
    return nn.initializers.uniform(scale=0.05)


def lff_weight_init(scale: float, num_inputs: int):
    return nn.initializers.normal(stddev=scale / num_inputs)


def lff_bias_init():
    return nn.initializers.uniform(scale=2)


class LFF(nn.Module):
    num_output_features: int
    num_input_features: int
    scale: float

    def setup(self):
        self.dense = nn.Dense(
            features=self.num_output_features,
            kernel_init=lff_weight_init(
                scale=self.scale, num_inputs=self.num_input_features
            ),
            bias_init=lff_bias_init(),
        )

    def __call__(self, x):
        return jnp.pi * jnp.sin(self.dense(x) - 1)


from functools import partial
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class RewardPertubationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(
        self, env: environment.Environment, perturbation_multiplier: float = 0.0
    ):
        super().__init__(env)
        self.pertubation_multiplier = perturbation_multiplier

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        key, pertubation_key = jax.random.split(key)
        reward = reward + self.pertubation_multiplier * jax.random.normal(
            pertubation_key,
            shape=reward.shape,
        )
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = envs.wrapper.EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = envs.wrapper.AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class ActorCritic(nn.Module):
    scale: float
    action_dim: Sequence[int]
    activation: str = "relu"
    input_size: int = 1

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = LFF(
            num_output_features=40 * self.input_size,
            num_input_features=x.shape[-1],
            scale=self.scale,
        )(x)
        # actor_mean = nn.Dense(
        #     256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(x)
        # actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            1024, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            1024, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # critic = nn.Dense(
        #     256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(x)
        # critic = activation(critic)
        critic = LFF(
            num_output_features=40 * self.input_size,
            num_input_features=x.shape[-1],
            scale=self.scale,
        )(x)
        critic = nn.Dense(
            1024, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1024, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = (
        RewardPertubationWrapper(
            BraxGymnaxWrapper(config["ENV_NAME"]),
            perturbation_multiplier=config["REWARD_NOISE"],
        ),
        None,
    )
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            scale=config["LFF_SCALE"],
            action_dim=env.action_space(env_params).shape[0],
            activation=config["ACTIVATION"],
            input_size=env.observation_space(env_params).shape[0],
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


config = {
    "LR": 3e-4,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ENV_NAME": args.env_name,
    "ANNEAL_LR": False,
    "LFF_SCALE": float(args.LFF_scale),
    "REWARD_NOISE": float(args.reward_noise),
}

rng = jax.random.PRNGKey(0)  # args.seed)
# rng = jax.random.split(rng, 1)
train_jit = jax.jit(make_train(config))
out = train_jit(rng)


import io
from typing import List, Optional, Tuple

import brax
import jax
import numpy as onp
import trimesh
from brax import base, math
from jax import numpy as jp
from PIL import Image
from pytinyrenderer import TinyRenderCamera as Camera
from pytinyrenderer import TinyRenderLight as Light
from pytinyrenderer import TinySceneRenderer as Renderer


class TextureRGB888:
    def __init__(self, pixels):
        self.pixels = pixels
        self.width = int(onp.sqrt(len(pixels) / 3))
        self.height = int(onp.sqrt(len(pixels) / 3))


class Grid(TextureRGB888):
    def __init__(self, grid_size, color):
        grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.int32)
        grid[:, :] = onp.array(color)
        grid[0] = onp.zeros((grid_size, 3), dtype=onp.int32)
        grid[:, 0] = onp.zeros((grid_size, 3), dtype=onp.int32)
        super().__init__(list(grid.ravel()))


_BASIC = TextureRGB888([133, 118, 102])
_TARGET = TextureRGB888([255, 34, 34])
_GROUND = Grid(100, [200, 200, 200])


def _scene(sys: brax.System, state: brax.State) -> Tuple[Renderer, List[int]]:
    """Converts a brax System and state to a pytinyrenderer scene and instances."""
    scene = Renderer()
    instances = []

    def take_i(obj, i):
        return jax.tree_map(lambda x: jp.take(x, i, axis=0), obj)

    link_names = [n or f"link {i}" for i, n in enumerate(sys.link_names)]
    link_names += ["world"]
    link_geoms = {}
    for batch in sys.geoms:
        num_geoms = len(batch.friction)
        for i in range(num_geoms):
            link_idx = -1 if batch.link_idx is None else batch.link_idx[i]
            link_geoms.setdefault(link_names[link_idx], []).append(take_i(batch, i))

    for _, geom in link_geoms.items():
        for col in geom:
            tex = TextureRGB888((col.rgba[:3] * 255).astype("uint32"))
            if isinstance(col, base.Capsule):
                half_height = col.length / 2
                model = scene.create_capsule(
                    col.radius, half_height, 2, tex.pixels, tex.width, tex.height
                )
            elif isinstance(col, base.Box):
                model = scene.create_cube(
                    col.halfsize, tex.pixels, tex.width, tex.height, 16.0
                )
            elif isinstance(col, base.Sphere):
                model = scene.create_capsule(
                    col.radius, 0, 2, tex.pixels, tex.width, tex.height
                )
            elif isinstance(col, base.Plane):
                tex = _GROUND
                model = scene.create_cube(
                    [1000.0, 1000.0, 0.0001], tex.pixels, tex.width, tex.height, 8192
                )
            elif isinstance(col, base.Convex):
                # convex objects are not visual
                continue
            elif isinstance(col, base.Mesh):
                tm = trimesh.Trimesh(vertices=col.vert, faces=col.face)
                vert_norm = tm.vertex_normals
                model = scene.create_mesh(
                    col.vert.reshape((-1)).tolist(),
                    vert_norm.reshape((-1)).tolist(),
                    [0] * col.vert.shape[0] * 2,
                    col.face.reshape((-1)).tolist(),
                    tex.pixels,
                    tex.width,
                    tex.height,
                    1.0,
                )
            else:
                raise RuntimeError(f"unrecognized collider: {type(col)}")

            i = col.link_idx if col.link_idx is not None else -1
            x = state.x.concatenate(base.Transform.zero((1,)))
            instance = scene.create_object_instance(model)
            off = col.transform.pos
            pos = onp.array(x.pos[i]) + math.rotate(off, x.rot[i])
            rot = col.transform.rot
            rot = math.quat_mul(x.rot[i], rot)
            scene.set_object_position(instance, list(pos))
            scene.set_object_orientation(instance, [rot[1], rot[2], rot[3], rot[0]])
            instances.append(instance)

    return scene, instances


def _eye(sys: brax.System, state: brax.State) -> List[float]:
    """Determines the camera location for a Brax system."""
    xj = state.x.vmap().do(sys.link.joint)
    dist = onp.concatenate(xj.pos[None, ...] - xj.pos[:, None, ...])
    dist = onp.linalg.norm(dist, axis=1).max()
    off = [2 * dist, -2 * dist, dist]
    return list(state.x.pos[0, :] + onp.array(off))


def _up(unused_sys: brax.System) -> List[float]:
    """Determines the up orientation of the camera."""
    return [0, 0, 1]


def get_camera(
    sys: brax.System, state: brax.State, width: int, height: int, ssaa: int = 2
) -> Camera:
    """Gets camera object."""
    eye, up = _eye(sys, state), _up(sys)
    hfov = 58.0
    vfov = hfov * height / width
    target = [state.x.pos[0, 0], state.x.pos[0, 1], 0]
    camera = Camera(
        viewWidth=width * ssaa,
        viewHeight=height * ssaa,
        position=eye,
        target=target,
        up=up,
        hfov=hfov,
        vfov=vfov,
    )
    return camera


def render_array(
    sys: brax.System,
    state: brax.State,
    width: int,
    height: int,
    light: Optional[Light] = None,
    camera: Optional[Camera] = None,
    ssaa: int = 2,
) -> onp.ndarray:
    """Renders an RGB array of a brax system and QP."""
    if (len(state.x.pos.shape), len(state.x.rot.shape)) != (2, 2):
        raise RuntimeError("unexpected shape in state")
    scene, instances = _scene(sys, state)
    target = state.x.pos[0, :]
    if light is None:
        direction = [0.57735, -0.57735, 0.57735]
        light = Light(
            direction=direction,
            ambient=0.8,
            diffuse=0.8,
            specular=0.6,
            shadowmap_center=target,
        )
    if camera is None:
        eye, up = _eye(sys, state), _up(sys)
        hfov = 58.0
        vfov = hfov * height / width
        camera = Camera(
            viewWidth=width * ssaa,
            viewHeight=height * ssaa,
            position=eye,
            target=target,
            up=up,
            hfov=hfov,
            vfov=vfov,
        )
    img = scene.get_camera_image(instances, light, camera).rgb
    arr = onp.reshape(
        onp.array(img, dtype=onp.uint8), (camera.view_height, camera.view_width, -1)
    )
    if ssaa > 1:
        arr = onp.asarray(Image.fromarray(arr).resize((width, height)))
    return arr


def render(
    sys: brax.System,
    states: List[brax.State],
    width: int,
    height: int,
    light: Optional[Light] = None,
    cameras: Optional[List[Camera]] = None,
    ssaa: int = 2,
    fmt="png",
) -> bytes:
    """Returns an image of a brax system and QP."""
    if not states:
        raise RuntimeError("must have at least one qp")
    if cameras is None:
        cameras = [None] * len(states)

    frames = [
        Image.fromarray(render_array(sys, state, width, height, light, camera, ssaa))
        for state, camera in zip(states, cameras)
    ]
    f = io.BytesIO()
    if len(frames) == 1:
        frames[0].save(f, format=fmt)
    else:
        frames[0].save(
            f,
            format=fmt,
            append_images=frames[1:],
            save_all=True,
            duration=sys.dt * 1000,
            loop=0,
        )
    return f.getvalue()


import numpy as np


def slow_video_frame_generation(frames: int = 10) -> np.ndarray:
    env = envs.create(env_name="ant", backend="spring")
    # @title Visualizing a trajectory of the learned inference function
    model = ActorCritic(
        scale=1, action_dim=env.action_size, input_size=env.observation_size
    )

    # create an env with auto-reset

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    images = []
    rng = jax.random.PRNGKey(seed=0)
    state = env.reset(rng=rng)
    for _ in range(frames):
        images.append(
            render_array(env.sys, state.pipeline_state, width=256, height=256)
        )
        act_rng, rng = jax.random.split(rng)
        policy, _ = model.apply(out["runner_state"][0].params, state.obs)
        action = policy.sample(seed=act_rng)
        state = env.step(state, action)
    return np.array(images)


def frames_to_gif(
    frames: np.ndarray,
    output_path: str = "output.gif",
    duration: int = 200,
    loop: int = 0,
) -> None:
    from PIL import Image

    image_frames = []

    # Iterate through each frame and convert it to a PIL Image
    for frame in frames:
        # Create a PIL Image from the frame
        image = Image.fromarray(frame.astype("uint8"), "RGB")
        image_frames.append(image)

    # Save the frames as a GIF
    image_frames[0].save(
        output_path,
        format="GIF",
        append_images=image_frames[1:],
        save_all=True,
        duration=duration,
        loop=loop,
    )


frames = slow_video_frame_generation(frames=1000)
frames_to_gif(frames)
