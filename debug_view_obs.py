from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
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

    # Entorno base
    env = FourLockedRoomEnv(
        size=19,
        render_mode="rgb_array",
        agent_view_size=7,
    )

    # Render completo del entorno
    obs, info = env.reset(seed=seed)
    full_render = env.render()

    # Pipeline actual de tu train_partialobs.py
    wrapped_env = FourLockedRoomEnv(
        size=19,
        render_mode="rgb_array",
        agent_view_size=7,
    )
    wrapped_env = RGBImgObsWrapper(wrapped_env)
    wrapped_env = ImgObsWrapper(wrapped_env)

    wrapped_obs, wrapped_info = wrapped_env.reset(seed=seed)

    print("Shape render completo:", full_render.shape)
    print("Shape observación actual:", wrapped_obs.shape)
    print("dtype observación actual:", wrapped_obs.dtype)

    save_image(full_render, "debug_outputs/full_env_render.png")
    save_image(wrapped_obs, "debug_outputs/current_agent_obs.png")

    print("Guardado:")
    print(" - debug_outputs/full_env_render.png")
    print(" - debug_outputs/current_agent_obs.png")

    env.close()
    wrapped_env.close()


if __name__ == "__main__":
    main()