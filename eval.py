"""
eval.py
-------
Evalúa un modelo PPO entrenado en FourLockedRoomEnv y opcionalmente graba vídeo.

Uso
---
    python eval.py --run_dir runs/Apr07_23_34_33
    python eval.py --run_dir runs/Apr07_23_34_33 --episodes 10
    python eval.py --run_dir runs/Apr07_23_34_33 --checkpoint 500000
    python eval.py --run_dir runs/Apr07_23_34_33 --no_video
    python eval.py --run_dir runs/Apr07_23_34_33 --stochastic

Requisitos
----------
    pip install stable-baselines3[extra] shimmy gymnasium minigrid pygame pyyaml openpyxl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from minigrid.wrappers import FullyObsWrapper

from envs.four_locked_room_env import FourLockedRoomEnv
from config import ExperimentConfig
from train import RGBFlatWrapper, ShapedRewardWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evalúa PPO en FourLockedRoomEnv")
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Carpeta del run, por ejemplo: runs/Apr07_23_34_33",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Número de episodios de evaluación",
    )
    p.add_argument(
        "--checkpoint",
        type=int,
        default=6500000,
        help="Si se indica, carga runs/<run_dir>/ppo_fourlocked_step<checkpoint>.pt",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Fuerza predicción determinista",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Fuerza predicción no determinista",
    )
    p.add_argument(
        "--tile_size",
        type=int,
        default=32,
        help="Tile size para render y vídeo",
    )
    p.add_argument(
        "--video_prefix",
        type=str,
        default=f"ppo_eval",
        help="Prefijo del nombre de los vídeos",
    )
    p.add_argument(
        "--no_video",
        action="store_true",
        help="Desactiva la grabación de vídeo",
    )
    return p.parse_args()


def resolve_model_path(run_dir: Path, checkpoint: int | None) -> Path:
    if checkpoint is None:
        model_path = run_dir / "ppo_fourlocked_final.pt"
    else:
        model_path = run_dir / f"ppo_fourlocked_step{checkpoint}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    return model_path


def make_single_eval_env(
    *,
    size: int,
    seed: int,
    full_obs: bool,
    key_bonus: float,
    door_bonus: float,
    goal_bonus: float,
    step_penalty: float,
    tile_size: int,
    record_video_folder: str | None = None,
    video_prefix: str = "ppo_eval",
    max_videos: int | None = None,
) -> gym.Env:
    env = FourLockedRoomEnv(
        size=size,
        render_mode="rgb_array",
        tile_size=tile_size,
    )

    if full_obs:
        env = FullyObsWrapper(env)

    env = ShapedRewardWrapper(env)
    env.KEY_BONUS = key_bonus
    env.DOOR_BONUS = door_bonus
    env.GOAL_BONUS = goal_bonus
    env.STEP_PENALTY = step_penalty

    env = RGBFlatWrapper(env)
    env = Monitor(env)

    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    try:
        env.observation_space.seed(seed)
    except Exception:
        pass

    if record_video_folder is not None:
        if max_videos is None:
            episode_trigger = lambda ep: True
        else:
            episode_trigger = lambda ep: ep < max_videos

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=record_video_folder,
            episode_trigger=episode_trigger,
            name_prefix=video_prefix,
            fps=8,
        )
        print(f"[Video] Recording enabled -> {record_video_folder}")

    return env


def make_vec_eval_env(
    *,
    size: int,
    seed: int,
    full_obs: bool,
    key_bonus: float,
    door_bonus: float,
    goal_bonus: float,
    step_penalty: float,
    tile_size: int,
    record_video_folder: str | None = None,
    video_prefix: str = "ppo_eval",
    max_videos: int | None = None,
):
    def _init():
        return make_single_eval_env(
            size=size,
            seed=seed,
            full_obs=full_obs,
            key_bonus=key_bonus,
            door_bonus=door_bonus,
            goal_bonus=goal_bonus,
            step_penalty=step_penalty,
            tile_size=tile_size,
            record_video_folder=record_video_folder,
            video_prefix=video_prefix,
            max_videos=max_videos,
        )

    env = DummyVecEnv([_init])
    env = VecTransposeImage(env)
    return env


def extract_terminal_info(info: dict[str, Any]) -> dict[str, Any]:
    episode_info = info.get("episode", {})
    return {
        "return": float(episode_info.get("r", 0.0)),
        "length": int(episode_info.get("l", 0)),
        "is_success": bool(info.get("is_success", False)),
        "got_key": bool(info.get("episode_got_key", False)),
        "opened_door": bool(info.get("episode_opened_door", False)),
    }


def main() -> None:
    args = parse_args()

    if args.deterministic and args.stochastic:
        raise ValueError("No puedes usar --deterministic y --stochastic a la vez")

    deterministic = True
    if args.stochastic:
        deterministic = False
    elif args.deterministic:
        deterministic = True

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta del run: {run_dir}")

    cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe config_used.yaml en: {cfg_path}")

    cfg = ExperimentConfig.from_yaml(cfg_path)
    model_path = resolve_model_path(run_dir, args.checkpoint)

    size = int(cfg.get("env", "size", default=19))
    full_obs = bool(cfg.get("env", "full_obs", default=True))
    seed = int(cfg.get("experiment", "seed", default=42))

    key_bonus = float(cfg.get("reward", "key_bonus", default=0.30))
    door_bonus = float(cfg.get("reward", "door_bonus", default=0.50))
    goal_bonus = float(cfg.get("reward", "goal_bonus", default=1.0))
    step_penalty = float(cfg.get("reward", "step_penalty", default=0.001))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_dir = None
    if not args.no_video:
        video_dir = str(run_dir / "videos_eval")
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[eval] Run dir: {run_dir}")
    print(f"[eval] Config: {cfg_path}")
    print(f"[eval] Model: {model_path}")
    print(f"[eval] Device: {device}")
    print(f"[eval] full_obs: {full_obs}")
    print(f"[eval] episodes: {args.episodes}")
    print(f"[eval] deterministic: {deterministic}")
    print("=" * 80)

    vec_env = make_vec_eval_env(
        size=size,
        seed=seed + 20_000,
        full_obs=full_obs,
        key_bonus=key_bonus,
        door_bonus=door_bonus,
        goal_bonus=goal_bonus,
        step_penalty=step_penalty,
        tile_size=args.tile_size,
        record_video_folder=video_dir,
        video_prefix=args.video_prefix,
        max_videos=args.episodes
    )

    model = PPO.load(
        str(model_path),
        env=vec_env,
        device=device,
    )

    returns = []
    lengths = []
    successes = []
    key_hits = []
    door_hits = []

    try:
        obs = vec_env.reset()

        for ep in range(args.episodes):
            done = False
            print(f"\n[Episode {ep + 1}] Starting...")

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = vec_env.step(action)
                done = bool(dones[0])

                if done:
                    final = extract_terminal_info(infos[0])

                    returns.append(final["return"])
                    lengths.append(final["length"])
                    successes.append(int(final["is_success"]))
                    key_hits.append(int(final["got_key"]))
                    door_hits.append(int(final["opened_door"]))

                    print(
                        f"Episode {ep + 1}: "
                        f"return={final['return']:.3f}, "
                        f"len={final['length']}, "
                        f"success={final['is_success']}, "
                        f"key={final['got_key']}, "
                        f"door={final['opened_door']}"
                    )

        print("\n" + "-" * 60)
        print(f"Mean return:  {np.mean(returns):.3f} ± {np.std(returns):.3f}")
        print(f"Mean length:  {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        print(f"Success rate: {np.mean(successes):.3f}")
        print(f"Key rate:     {np.mean(key_hits):.3f}")
        print(f"Door rate:    {np.mean(door_hits):.3f}")

    finally:
        vec_env.close()

    if video_dir is not None:
        print(f"[eval] Videos saved in: {video_dir}")


if __name__ == "__main__":
    main()