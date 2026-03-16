from __future__ import annotations

from envs.four_locked_room_env import FourLockedRoomEnv
from minigrid.manual_control import ManualControl


def main() -> None:
    env = FourLockedRoomEnv(
        render_mode="human",
        size=19,
        max_steps=500,
    )

    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()