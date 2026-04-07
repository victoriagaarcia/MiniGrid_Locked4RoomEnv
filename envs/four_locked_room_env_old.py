from __future__ import annotations

from dataclasses import dataclass

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Wall # Door
from minigrid.minigrid_env import MiniGridEnv

from envs.color_door_old import ColorDoor as Door


@dataclass
class FourRoom:
    top: tuple[int, int]
    size: tuple[int, int]
    door_pos: tuple[int, int]
    color: str | None = None
    locked: bool = False

    def rand_pos(self, env: MiniGridEnv) -> tuple[int, int]:
        top_x, top_y = self.top
        size_x, size_y = self.size
        return env._rand_pos(
            top_x + 1,
            top_x + size_x - 1,
            top_y + 1,
            top_y + size_y - 1,
        )


class FourLockedRoomEnv(MiniGridEnv):
    """
    Custom MiniGrid environment with 4 rooms instead of 6.

    Layout:
    - central vertical hallway
    - 2 rooms on the left
    - 2 rooms on the right

    Task:
    - pick up the key
    - unlock the locked room
    - reach the goal
    """

    def __init__(self, size: int = 19, max_steps: int | None = None, **kwargs):
        self.size = size

        if max_steps is None:
            max_steps = 10 * size

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES] * 3,
        )

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(lockedroom_color: str, keyroom_color: str, door_color: str) -> str:
        return (
            f"get the {lockedroom_color} key from the {keyroom_color} room, "
            f"unlock the {door_color} door and go to the goal"
        )

    def _gen_grid(self, width: int, height: int) -> None:
        # Create empty grid
        self.grid = Grid(width, height)

        # =========================
        # 1. Outer walls
        # =========================
        for i in range(width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 1, Wall())

        for j in range(height):
            self.grid.set(0, j, Wall())
            self.grid.set(width - 1, j, Wall())

        # =========================
        # 2. Central vertical hallway
        # =========================
        left_hall_x = width // 2 - 2
        right_hall_x = width // 2 + 2

        for j in range(height):
            self.grid.set(left_hall_x, j, Wall())
            self.grid.set(right_hall_x, j, Wall())

        # =========================
        # 3. Split vertically into 2 levels
        #    => 4 rooms total
        # =========================
        self.rooms: list[FourRoom] = []

        split_rows = 2
        room_span = height // split_rows

        for n in range(split_rows):
            y = n * room_span

            # Horizontal walls separating upper/lower rooms
            for i in range(0, left_hall_x):
                self.grid.set(i, y, Wall())

            for i in range(right_hall_x, width):
                self.grid.set(i, y, Wall())

            room_w = left_hall_x + 1
            room_h = room_span + 1

            # Put door roughly centered vertically in each room
            door_y = y + room_span // 2
            door_y = max(1, min(height - 2, door_y))

            # Left room
            self.rooms.append(
                FourRoom(
                    top=(0, y),
                    size=(room_w, room_h),
                    door_pos=(left_hall_x, door_y),
                )
            )

            # Right room
            self.rooms.append(
                FourRoom(
                    top=(right_hall_x, y),
                    size=(room_w, room_h),
                    door_pos=(right_hall_x, door_y),
                )
            )

        # =========================
        # 4. Choose locked room
        # =========================
        locked_room = self._rand_elem(self.rooms)
        locked_room.locked = True

        goal_pos = locked_room.rand_pos(self)
        self.grid.set(*goal_pos, Goal("yellow"))

        # =========================
        # 5. Assign colors + place doors
        # =========================
        colors = set(COLOR_NAMES)

        for room in self.rooms:
            color = self._rand_elem(sorted(colors))
            colors.remove(color)
            room.color = color

            if room.locked:
                self.grid.set(*room.door_pos, Door(color, is_locked=True))
            else:
                self.grid.set(*room.door_pos, Door(color))

        # =========================
        # 6. Place key in another room
        # =========================
        while True:
            key_room = self._rand_elem(self.rooms)
            if key_room != locked_room:
                break

        key_pos = key_room.rand_pos(self)
        # self.grid.set(*key_pos, Key(locked_room.color))
        self.grid.set(*key_pos, Key("red"))  # Key color doesn't affect logic, only visualization

        # =========================
        # 7. Place agent in hallway
        # =========================
        self.agent_pos = self.place_agent(
            top=(left_hall_x, 0),
            size=(right_hall_x - left_hall_x, height),
        )

        # =========================
        # 8. Mission
        # =========================
        self.mission = (
            f"get the {locked_room.color} key from the {key_room.color} room, "
            f"unlock the {locked_room.color} door and go to the goal"
        )