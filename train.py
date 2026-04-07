"""
train.py
--------
Entrena un agente PPO sobre FourLockedRoomEnv usando observaciones RGB.

Uso
---
    python train.py                        # entrena con hiperparámetros por defecto
    python train.py --timesteps 5_000_000  # más pasos
    python train.py --size 13              # grid más pequeño (más rápido)
    python train.py --eval                 # activa evaluación periódica

Requisitos
----------
    pip install stable-baselines3[extra] shimmy gymnasium minigrid pygame

Tensorboard
-----------
    tensorboard --logdir logs/tensorboard
"""

import argparse
import os
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

import torch.nn as nn
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper

from envs.four_locked_room_env import FourLockedRoomEnv


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  WRAPPER: convierte la obs de MiniGrid a imagen RGB pura (H,W,C) uint8
# ═══════════════════════════════════════════════════════════════════════════════

class RGBFlatWrapper(gym.ObservationWrapper):
    """
    Aplica RGBImgObsWrapper + ImgObsWrapper de MiniGrid para obtener
    una observación (H, W, 3) uint8 que SB3 entiende directamente.
    Los 3 canales son los colores RGB renderizados, no índices de objetos.
    """
    def __init__(self, env: gym.Env):
        # Si usáramos solo ImgObsWrapper, la obs sería (H,W) con índices de objetos (los tres canales serían: tipo, color, estado)
        env = RGBImgObsWrapper(env)   # obs → {'image': (H,W,3)}
        env = ImgObsWrapper(env)       # obs → (H,W,3)
        super().__init__(env)
        self.observation_space = self.env.observation_space

    def observation(self, obs):
        return obs


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  REWARD SHAPING WRAPPER
#     Recompensas adicionales para guiar el aprendizaje en la tarea jerárquica:
#       - Recoger la llave roja           → +0.3
#       - Abrir la puerta roja            → +0.5
#       - Llegar al goal (MiniGrid base)  → +1.0  (ya incluido)
#       - Penalización por paso           → -1/max_steps  (ya incluido en base)
# ═══════════════════════════════════════════════════════════════════════════════

class ShapedRewardWrapper(gym.Wrapper):
    """
    Añade señales de reward intermedias para los subobjetivos.
    Cada bonus se da una sola vez por episodio.

    Las recompensas base de MiniGrid ya incluyen:
    - +1.0 por llegar al goal
    - Penalización por paso: -1 / max_steps (para incentivar soluciones más rápidas)

    Bonus adicionales:
    - Recoger la llave roja  → +0.3
    - Abrir la puerta roja   → +0.5
    """

    KEY_BONUS  = 0.30
    DOOR_BONUS = 0.50

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._key_picked  = False
        self._door_opened = False

    def reset(self, **kwargs):
        self._key_picked  = False
        self._door_opened = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward

        # Bonus: agente recoge la llave roja
        if not self._key_picked:
            carrying = self.env.unwrapped.carrying
            if carrying is not None and carrying.type == "key" and carrying.color == "red":
                shaped += self.KEY_BONUS
                self._key_picked = True
                info["bonus_key"] = True

        # Bonus: puerta roja pasa a estar abierta
        if not self._door_opened and self._key_picked:
            # Busca en el grid si alguna puerta roja está abierta
            env_u = self.env.unwrapped
            for room in env_u.rooms:
                if room.locked:
                    door = env_u.grid.get(*room.door_pos)
                    # La puerta pasa a None cuando se abre completamente
                    if door is None or (hasattr(door, "is_open") and door.is_open):
                        shaped += self.DOOR_BONUS
                        self._door_opened = True
                        info["bonus_door"] = True
                        break

        return obs, shaped, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CNN PERSONALIZADA
#     MinigridCNN: tres capas conv + flatten → features_dim
#     Diseñada para imágenes pequeñas (tile_size=8, grid ~19x19 → ~168x168 px)
# ═══════════════════════════════════════════════════════════════════════════════

class MinigridCNN(BaseFeaturesExtractor):
    """
    Extractor de features CNN para observaciones RGB de MiniGrid.
    Arquitectura inspirada en la Nature DQN pero ajustada a resoluciones
    pequeñas (< 200 px) con menos filtros para no sobreajustar.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # tras VecTransposeImage: C,H,W

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

        # Calcula la dimensión de salida de la CNN dinámicamente
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
# 4.  CALLBACK: métricas adicionales en TensorBoard
# ═══════════════════════════════════════════════════════════════════════════════

class InfoLoggerCallback(BaseCallback):
    """
    Loguea métricas de episodio en TensorBoard:
      - bonus_key y bonus_door (frecuencia de consecución de subobjetivos)
      - success_rate (episodios en que el agente llega al goal)
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._key_successes  = []
        self._door_successes = []
        self._goal_successes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if info.get("bonus_key"):
                self._key_successes.append(1)
            if info.get("bonus_door"):
                self._door_successes.append(1)
            if done:
                # MiniGrid pone "TimeLimit.truncated" si se acaba el tiempo
                success = not info.get("TimeLimit.truncated", False) and \
                          info.get("episode", {}).get("r", 0) > 0.9
                self._goal_successes.append(int(success))

        # Loguea cada 1000 pasos aprox.
        if self.n_calls % 1000 == 0 and self._goal_successes:
            window = 200
            self.logger.record(
                "custom/key_rate",
                np.mean(self._key_successes[-window:]) if self._key_successes else 0,
            )
            self.logger.record(
                "custom/door_rate",
                np.mean(self._door_successes[-window:]) if self._door_successes else 0,
            )
            self.logger.record(
                "custom/success_rate",
                np.mean(self._goal_successes[-window:]),
            )
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FACTORY DE ENTORNO
# ═══════════════════════════════════════════════════════════════════════════════

def make_env_fn(size: int = 19, seed: int = 0) -> Callable[[], gym.Env]:
    """Devuelve una función que crea y envuelve el entorno (necesario para SubprocVecEnv)."""
    def _init() -> gym.Env:
        env = FourLockedRoomEnv(size=size, render_mode="rgb_array")
        # Queremos que el agente vea todo el mapa
        env = FullyObsWrapper(env)
        env = ShapedRewardWrapper(env)
        env = RGBFlatWrapper(env)
        env = Monitor(env)
        return env
    return _init


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRENAMIENTO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_EVERY = 50_000   # cada cuántos timesteps guardar checkpoint

def train(args: argparse.Namespace) -> None:
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("logs/checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Usando device: {device}")
    if device == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

    # ── Entornos de entrenamiento (paralelos) ──────────────────────────────
    n_envs = args.n_envs
    vec_env = SubprocVecEnv(
        [make_env_fn(size=args.size, seed=i) for i in range(n_envs)]
    )
    vec_env = VecTransposeImage(vec_env)   # (H,W,C) → (C,H,W) para PyTorch

    # ── Entorno de evaluación (1 único, determinista) ──────────────────────
    eval_env = None
    if args.eval:
        eval_env = SubprocVecEnv([make_env_fn(size=args.size, seed=9999)])
        eval_env = VecTransposeImage(eval_env)

    # ── Política PPO con CNN personalizada ────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        normalize_images=False,   # ya normalizamos en forward() de la CNN
    )

    model = PPO(
        policy          = "CnnPolicy",
        env             = vec_env,
        device          = device,
        # ── Hiperparámetros ────────────────────────────────────────────
        learning_rate   = 2.5e-4,
        n_steps         = 512,       # pasos por env antes de actualizar
        batch_size      = 256,       # minibatch para el optimizador
        n_epochs        = 4,         # épocas de optimización por update
        gamma           = 0.99,      # descuento
        gae_lambda      = 0.95,      # GAE lambda
        clip_range      = 0.2,       # PPO clip
        ent_coef        = 0.01,      # entropía → exploración
        vf_coef         = 0.5,       # coeficiente del value loss
        max_grad_norm   = 0.5,
        # ── Logging ───────────────────────────────────────────────────
        tensorboard_log = "logs/tensorboard",
        verbose         = 1,
        policy_kwargs   = policy_kwargs,
    )

    print(f"[train] Parámetros del modelo: "
          f"{sum(p.numel() for p in model.policy.parameters()):,}")

    # ── Callbacks (solo InfoLogger y, opcionalmente, EvalCallback) ────────
    callbacks = [InfoLoggerCallback()]

    if eval_env is not None:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path = "models/",
                log_path             = "logs/eval",
                eval_freq            = max(100_000 // n_envs, 1),
                n_eval_episodes      = 20,
                deterministic        = True,
                verbose              = 1,
            )
        )

    # ── Bucle de entrenamiento con checkpoints manuales en .pt ────────────
    steps_done = 0
    try:
        while steps_done < args.timesteps:
            chunk = min(CHECKPOINT_EVERY, args.timesteps - steps_done)
            model.learn(
                total_timesteps     = chunk,
                callback            = CallbackList(callbacks),
                tb_log_name         = "ppo_fourlocked",
                reset_num_timesteps = (steps_done == 0),  # solo True en el primer chunk
                progress_bar        = True,
            )
            steps_done += chunk

            # Guardar checkpoint en formato .pt
            ckpt_path = os.path.join(
                "logs/checkpoints", f"ppo_fourlocked_step{steps_done}.pt"
            )
            model.save(ckpt_path)
            print(f"[Checkpoint] Guardado: {ckpt_path}")

        # ── Guardar modelo final ───────────────────────────────────────
        final_path = "models/ppo_fourlocked_final.pt"
        model.save(final_path)
        print(f"[train] Modelo final guardado en {final_path}")

    finally:
        vec_env.close()
        if eval_env is not None:
            eval_env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ARGPARSE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena PPO en FourLockedRoomEnv")
    p.add_argument("--timesteps", type=int,   default=3_000_000,
                   help="Total de timesteps de entrenamiento (default: 3M)")
    p.add_argument("--size",      type=int,   default=19,
                   help="Tamaño del grid (default: 19)")
    p.add_argument("--n_envs",    type=int,   default=8,
                   help="Nº de entornos paralelos (default: 8)")
    p.add_argument("--eval",      action="store_true",
                   help="Activa EvalCallback con entorno de evaluación separado")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
