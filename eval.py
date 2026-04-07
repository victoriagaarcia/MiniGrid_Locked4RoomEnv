import os
import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from envs.four_locked_room_env import FourLockedRoomEnv
from train_old import RGBFlatWrapper, ShapedRewardWrapper


# -----------------------------
# Configuración
# -----------------------------
ALGO = "ppo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL_PATH = "models/ppo_fourlocked_final.zip"
MODEL_DATE = "Mar29_09_36_01"  # ajusta esto
LOAD_STEP =  500_000            # ajusta esto
MODEL_PATH = f"runs/{MODEL_DATE}/ppo_fourlocked_step{LOAD_STEP}.pt"
# VIDEO_DIR = "eval_videos"
VIDEO_DIR = f"runs/{MODEL_DATE}/videos_eval"
VIDEO_PREFIX = f"ppo_eval_step{LOAD_STEP}"
VIDEO_PREFIX = "ppo_eval"

N_EPISODES = 5

SEED = 42
SIZE = 19
TILE_SIZE = 32
DETERMINISTIC = True

os.makedirs(VIDEO_DIR, exist_ok=True)


def make_single_fourlocked_env(
    size=19,
    tile_size=32,
    record_video_folder=None,
    video_prefix="eval",
):
    env = FourLockedRoomEnv(
        size=size,
        render_mode="rgb_array",
        tile_size=tile_size,
    )

    if record_video_folder is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(record_video_folder),
            episode_trigger=lambda ep: True,
            name_prefix=video_prefix,
        )
        print(f"[Video] Recording enabled. Videos will be saved to: {record_video_folder}")

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ShapedRewardWrapper(env)
    env = RGBFlatWrapper(env)

    return env


def make_vec_fourlocked_env(
    size=19,
    tile_size=32,
    record_video_folder=None,
    video_prefix="eval",
):
    def _init():
        return make_single_fourlocked_env(
            size=size,
            tile_size=tile_size,
            record_video_folder=record_video_folder,
            video_prefix=video_prefix,
        )

    env = DummyVecEnv([_init])
    env = VecTransposeImage(env)
    return env


def get_raw_env(vec_env):
    return vec_env.venv.envs[0].unwrapped


def main():
    vec_env = make_vec_fourlocked_env(
        size=SIZE,
        tile_size=TILE_SIZE,
        record_video_folder=VIDEO_DIR,
        video_prefix=VIDEO_PREFIX,
    )

    model = PPO.load(
        MODEL_PATH,
        env=vec_env,
        device=DEVICE,
    )

    raw_env = get_raw_env(vec_env)

    returns = []
    lengths = []

    try:
        for ep in range(N_EPISODES):
            obs = vec_env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0

            print(f"\n[Episode {ep + 1}] Starting...")

            while not done:
                action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                obs, rewards, dones, infos = vec_env.step(action)

                ep_return += float(rewards[0])
                ep_len += 1
                done = bool(dones[0])

            returns.append(ep_return)
            lengths.append(ep_len)

            print(f"Episode {ep + 1} return: {ep_return:.2f}, len: {ep_len}")

            carrying_obj = getattr(raw_env, "carrying", None)
            carrying_str = (
                f"{carrying_obj.color} {carrying_obj.type}"
                if carrying_obj is not None else "nothing"
            )
            mission = getattr(raw_env, "mission", "")

            print("----EPISODE END----")
            print("step:", ep_len)
            print("mission:", mission)
            print("carrying:", carrying_str)

        print("-" * 60)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"Mean length: {mean_length:.2f} ± {std_length:.2f}")

    finally:
        vec_env.close()

    print(f"Videos saved in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()