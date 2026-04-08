"""
train.py
--------
Entrena un agente PPO sobre FourLockedRoomEnv usando observaciones RGB.

Uso
---
    python train.py
    python train.py --config configs/ppo_fourlocked.yaml

Requisitos
----------
    pip install stable-baselines3[extra] shimmy gymnasium minigrid pygame pyyaml openpyxl

Tensorboard
-----------
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Callable, Any

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper

from envs.four_locked_room_env import FourLockedRoomEnv
from config import ExperimentConfig
from utils.run_logger import append_run, update_run


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UTILIDADES GENERALES
# ═══════════════════════════════════════════════════════════════════════════════

def set_global_seeds(seed: int) -> None:
    """Fija semillas globales para mejorar reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WRAPPER: observación RGB pura (H,W,C) uint8
# ═══════════════════════════════════════════════════════════════════════════════

class RGBFlatWrapper(gym.ObservationWrapper):
    """
    Convierte la observación de MiniGrid a una imagen RGB pura.

    Aplica:
      - RGBImgObsWrapper: obs -> {'image': (H,W,3)}
      - ImgObsWrapper:    obs -> (H,W,3)
    """
    def __init__(self, env: gym.Env):
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        super().__init__(env)
        self.observation_space = self.env.observation_space

    def observation(self, obs):
        return obs


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WRAPPER: reward shaping + métricas explícitas
# ═══════════════════════════════════════════════════════════════════════════════

class ShapedRewardWrapper(gym.Wrapper):
    """
    Añade señales intermedias de reward para guiar el aprendizaje.

    Recompensas base de MiniGrid:
      - recompensa positiva por llegar al goal
      - penalización implícita por tardar más pasos

    Bonus adicionales:
      - Recoger la llave roja  -> +KEY_BONUS
      - Abrir la puerta roja   -> +DOOR_BONUS

    Además deja trazas explícitas en `info` para logging:
      - base_reward
      - shaped_reward
      - bonus_key
      - bonus_door
      - is_success
      - episode_got_key
      - episode_opened_door
    """

    KEY_BONUS = 0.30
    DOOR_BONUS = 0.50
    GOAL_BONUS = 1.0
    STEP_PENALTY = 0.001

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._key_picked = False
        self._door_opened = False

    def reset(self, **kwargs):
        self._key_picked = False
        self._door_opened = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0

        got_key_now = False
        if not self._key_picked:
            carrying = self.env.unwrapped.carrying
            if carrying is not None and carrying.type == "key" and carrying.color == "red":
                shaped_reward += self.KEY_BONUS
                self._key_picked = True
                got_key_now = True

        opened_door_now = False
        if not self._door_opened and self._key_picked:
            env_u = self.env.unwrapped
            for room in env_u.rooms:
                if room.locked:
                    door = env_u.grid.get(*room.door_pos)
                    if door is None or (hasattr(door, "is_open") and door.is_open):
                        shaped_reward += self.DOOR_BONUS
                        self._door_opened = True
                        opened_door_now = True
                        break
        
        is_success = bool(terminated and self._door_opened)
        
        if is_success:
            shaped_reward += self.GOAL_BONUS
        elif not terminated and not truncated:
            shaped_reward -= self.STEP_PENALTY

        info["shaped_reward"] = float(shaped_reward)
        info["bonus_key"] = got_key_now
        info["bonus_door"] = opened_door_now
        info["is_success"] = is_success
        info["episode_got_key"] = self._key_picked
        info["episode_opened_door"] = self._door_opened

        return obs, shaped_reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CNN PERSONALIZADA
# ═══════════════════════════════════════════════════════════════════════════════

class MinigridCNN(BaseFeaturesExtractor):
    """
    Extractor CNN para observaciones RGB de MiniGrid.
    Espera tensores en formato (C,H,W), que llegan tras VecTransposeImage.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            cnn_out_dim = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs.float() / 255.0))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CALLBACK: logging de métricas en TensorBoard
# ═══════════════════════════════════════════════════════════════════════════════

class InfoLoggerCallback(BaseCallback):
    """
    Métricas de episodio:
      - key_rate
      - door_rate
      - success_rate
      - key_to_door_gap
      - door_to_goal_gap

    Además guarda el último success_rate para poder escribirlo en Excel al final.
    """
    def __init__(self, window_size: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.key_ep: list[int] = []
        self.door_ep: list[int] = []
        self.success_ep: list[int] = []
        self.last_success_rate: float = 0.0
        self.last_key_rate: float = 0.0
        self.last_door_rate: float = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                self.key_ep.append(int(info.get("episode_got_key", False)))
                self.door_ep.append(int(info.get("episode_opened_door", False)))
                self.success_ep.append(int(info.get("is_success", False)))

        if self.n_calls % 1000 == 0 and len(self.success_ep) > 0:
            w = self.window_size

            self.last_key_rate = float(np.mean(self.key_ep[-w:])) if self.key_ep else 0.0
            self.last_door_rate = float(np.mean(self.door_ep[-w:])) if self.door_ep else 0.0
            self.last_success_rate = float(np.mean(self.success_ep[-w:]))

            self.logger.record("custom/key_rate", self.last_key_rate)
            self.logger.record("custom/door_rate", self.last_door_rate)
            self.logger.record("custom/success_rate", self.last_success_rate)
            self.logger.record("custom/key_to_door_gap", self.last_key_rate - self.last_door_rate)
            self.logger.record("custom/door_to_goal_gap", self.last_door_rate - self.last_success_rate)

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FACTORY DE ENTORNO
# ═══════════════════════════════════════════════════════════════════════════════

def make_env_fn(
    size: int = 19,
    seed: int = 0,
    full_obs: bool = True,
    key_bonus: float = 0.30,
    door_bonus: float = 0.50,
    goal_bonus: float = 1.0,
    step_penalty: float = 0.001,
) -> Callable[[], gym.Env]:
    """
    Devuelve una función que crea y envuelve el entorno.
    Necesario para SubprocVecEnv.
    """
    def _init() -> gym.Env:
        env = FourLockedRoomEnv(size=size, render_mode="rgb_array")

        if full_obs:
            env = FullyObsWrapper(env)

        env = ShapedRewardWrapper(env)
        env.KEY_BONUS = key_bonus
        env.DOOR_BONUS = door_bonus
        env.GOAL_BONUS = goal_bonus
        env.STEP_PENALTY = step_penalty

        env = RGBFlatWrapper(env)
        env = Monitor(env)

        # Semillado explícito del entorno y espacios
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass

        return env

    return _init


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ARGPARSE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena PPO en FourLockedRoomEnv")
    p.add_argument(
        "--config",
        type=str,
        default="ppo_fourlocked.yaml",
        help="Ruta al archivo YAML de configuración",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ENTRENAMIENTO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    # ── Cargar config ────────────────────────────────────────────────────────
    cfg = ExperimentConfig.from_yaml(args.config)

    experiment_name = cfg.get("experiment", "name", default="ppo_fourlocked")
    seed = int(cfg.get("experiment", "seed", default=42))
    total_timesteps = int(cfg.get("experiment", "total_timesteps", default=3_000_000))
    checkpoint_every = int(cfg.get("experiment", "checkpoint_every", default=50_000))
    use_eval = bool(cfg.get("experiment", "eval", default=True))
    n_eval_episodes = int(cfg.get("experiment", "n_eval_episodes", default=20))
    eval_freq = int(cfg.get("experiment", "eval_freq", default=100_000))

    size = int(cfg.get("env", "size", default=19))
    full_obs = bool(cfg.get("env", "full_obs", default=True))
    n_envs = int(cfg.get("env", "n_envs", default=8))

    key_bonus = float(cfg.get("reward", "key_bonus", default=0.30))
    door_bonus = float(cfg.get("reward", "door_bonus", default=0.50))
    goal_bonus = float(cfg.get("reward", "goal_bonus", default=1.0))
    step_penalty = float(cfg.get("reward", "step_penalty", default=0.001))

    # ── Seeds globales ───────────────────────────────────────────────────────
    set_global_seeds(seed)

    # ── Directorios de run ───────────────────────────────────────────────────
    timestamp_slug = datetime.now().strftime("%b%d_%H_%M_%S")
    timestamp_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_id = f"{experiment_name}_{timestamp_slug}"
    run_dir = Path("runs") / timestamp_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = run_dir / cfg.get("logging", "tensorboard_subdir", default="tensorboard")
    tb_dir.mkdir(parents=True, exist_ok=True)

    cfg_used_path = run_dir / "config_used.yaml"
    cfg.save_yaml(cfg_used_path)

    excel_path = Path("runs") / cfg.get("logging", "excel_name", default="experiments.xlsx")
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"[train] Run ID: {run_id}")
    print(f"[train] Run dir: {run_dir}")
    print(f"[train] Config usada: {cfg_used_path}")
    print(f"[train] Excel global: {excel_path}")
    print(f"[train] Device: {device}")
    if device == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # ── Entorno de entrenamiento ────────────────────────────────────────────
    vec_env = SubprocVecEnv(
        [
            make_env_fn(
                size=size,
                seed=seed + i,
                full_obs=full_obs,
                key_bonus=key_bonus,
                door_bonus=door_bonus,
                goal_bonus=goal_bonus,
                step_penalty=step_penalty,
            )
            for i in range(n_envs)
        ]
    )
    vec_env = VecTransposeImage(vec_env)

    # ── Entorno de evaluación ────────────────────────────────────────────────
    eval_env = None
    if use_eval:
        eval_env = SubprocVecEnv(
            [
                make_env_fn(
                    size=size,
                    seed=seed + 10_000,
                    full_obs=full_obs,
                    key_bonus=key_bonus,
                    door_bonus=door_bonus,
                    goal_bonus=goal_bonus,
                    step_penalty=step_penalty,
                )
            ]
        )
        eval_env = VecTransposeImage(eval_env)

    # ── Policy kwargs ────────────────────────────────────────────────────────
    features_dim = int(cfg.get("model", "features_dim", default=512))
    pi_layers = cfg.get("model", "pi_layers", default=[256, 256])
    vf_layers = cfg.get("model", "vf_layers", default=[256, 256])
    policy_name = cfg.get("model", "policy", default="CnnPolicy")

    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(pi=pi_layers, vf=vf_layers),
        normalize_images=False,
    )

    # ── Modelo PPO ───────────────────────────────────────────────────────────
    model = PPO(
        policy=policy_name,
        env=vec_env,
        device=device,
        learning_rate=float(cfg.get("ppo", "learning_rate", default=2.5e-4)),
        n_steps=int(cfg.get("ppo", "n_steps", default=512)),
        batch_size=int(cfg.get("ppo", "batch_size", default=256)),
        n_epochs=int(cfg.get("ppo", "n_epochs", default=4)),
        gamma=float(cfg.get("ppo", "gamma", default=0.99)),
        gae_lambda=float(cfg.get("ppo", "gae_lambda", default=0.95)),
        clip_range=float(cfg.get("ppo", "clip_range", default=0.2)),
        ent_coef=float(cfg.get("ppo", "ent_coef", default=0.01)),
        vf_coef=float(cfg.get("ppo", "vf_coef", default=0.5)),
        max_grad_norm=float(cfg.get("ppo", "max_grad_norm", default=0.5)),
        tensorboard_log=str(tb_dir),
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )

    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"[train] Parámetros del modelo: {param_count:,}")

    # ── Registrar la run en Excel al inicio ─────────────────────────────────
    row_idx = append_run(
        excel_path,
        {
            "run_id": run_id,
            "timestamp": timestamp_human,
            "experiment_name": experiment_name,
            "seed": seed,
            "size": size,
            "full_obs": full_obs,
            "n_envs": n_envs,
            "total_timesteps": total_timesteps,
            "learning_rate": cfg.get("ppo", "learning_rate"),
            "n_steps": cfg.get("ppo", "n_steps"),
            "batch_size": cfg.get("ppo", "batch_size"),
            "n_epochs": cfg.get("ppo", "n_epochs"),
            "gamma": cfg.get("ppo", "gamma"),
            "gae_lambda": cfg.get("ppo", "gae_lambda"),
            "clip_range": cfg.get("ppo", "clip_range"),
            "ent_coef": cfg.get("ppo", "ent_coef"),
            "vf_coef": cfg.get("ppo", "vf_coef"),
            "max_grad_norm": cfg.get("ppo", "max_grad_norm"),
            "key_bonus": key_bonus,
            "door_bonus": door_bonus,
            "features_dim": features_dim,
            "pi_layers": str(pi_layers),
            "vf_layers": str(vf_layers),
            "param_count": param_count,
            "status": "running",
            "best_mean_reward": "",
            "last_success_rate": "",
            "final_model_path": "",
            "best_model_path": str(run_dir / "best_model.zip") if use_eval else "",
            "notes": "",
        },
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    info_logger = InfoLoggerCallback(window_size=200)
    callbacks: list[BaseCallback] = [info_logger]

    if eval_env is not None:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir),
                eval_freq=max(eval_freq // n_envs, 1),
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=1,
            )
        )

    callback = CallbackList(callbacks)

    # ── Entrenamiento con checkpoints ────────────────────────────────────────
    steps_done = 0
    final_model_path = run_dir / "ppo_fourlocked_final.pt"
    best_model_path = run_dir / "best_model.zip"
    training_status = "finished"
    error_notes = ""

    try:
        while steps_done < total_timesteps:
            chunk = min(checkpoint_every, total_timesteps - steps_done)

            model.learn(
                total_timesteps=chunk,
                callback=callback,
                tb_log_name=experiment_name,
                reset_num_timesteps=(steps_done == 0),
                progress_bar=True,
            )

            steps_done += chunk

            ckpt_path = run_dir / f"ppo_fourlocked_step{steps_done}.pt"
            model.save(str(ckpt_path))
            print(f"[Checkpoint] Guardado: {ckpt_path}")

        model.save(str(final_model_path))
        print(f"[train] Modelo final guardado en {final_model_path}")

    except Exception as e:
        training_status = f"failed: {type(e).__name__}"
        error_notes = str(e)
        raise

    finally:
        # Mejor esfuerzo para sacar métricas de evaluación si existen
        best_mean_reward: Any = ""
        if use_eval:
            eval_npz = run_dir / "evaluations.npz"
            if eval_npz.exists():
                try:
                    data = np.load(eval_npz)
                    results = data.get("results", None)
                    if results is not None and len(results) > 0:
                        # results shape: (n_evals, n_episodes)
                        best_mean_reward = float(np.max(np.mean(results, axis=1)))
                except Exception:
                    best_mean_reward = ""

        update_run(
            excel_path,
            row_idx,
            {
                "status": training_status,
                "best_mean_reward": best_mean_reward,
                "last_success_rate": info_logger.last_success_rate,
                "final_model_path": str(final_model_path) if final_model_path.exists() else "",
                "best_model_path": str(best_model_path) if best_model_path.exists() else "",
                "last_key_rate": info_logger.last_key_rate,
                "last_door_rate": info_logger.last_door_rate,
                "steps_completed": steps_done,
                "notes": error_notes,
            },
        )

        vec_env.close()
        if eval_env is not None:
            eval_env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    train(args)