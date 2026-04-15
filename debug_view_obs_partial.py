from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from envs.four_locked_room_env import FourLockedRoomEnv


def save_image(img: np.ndarray, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    seed = 42

    env = FourLockedRoomEnv(
        size=19,
        render_mode="rgb_array",
        agent_view_size=7,
    )
    obs, info = env.reset(seed=seed)
    full_render = env.render()

    wrapped_env = FourLockedRoomEnv(
        size=19,
        render_mode="rgb_array",
        agent_view_size=7,
    )
    wrapped_env = RGBImgPartialObsWrapper(wrapped_env)
    wrapped_env = ImgObsWrapper(wrapped_env)

    wrapped_obs, wrapped_info = wrapped_env.reset(seed=seed)

    print("Shape render completo:", full_render.shape)
    print("Shape observación parcial correcta:", wrapped_obs.shape)
    print("dtype observación parcial correcta:", wrapped_obs.dtype)

    save_image(full_render, "debug_outputs/full_env_render_correct.png")
    save_image(wrapped_obs, "debug_outputs/correct_partial_agent_obs.png")

    print("Guardado:")
    print(" - debug_outputs/full_env_render_correct.png")
    print(" - debug_outputs/correct_partial_agent_obs.png")

    env.close()
    wrapped_env.close()


if __name__ == "__main__":
    main()