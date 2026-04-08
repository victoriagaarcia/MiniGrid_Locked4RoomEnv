from __future__ import annotations

from dataclasses import dataclass

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

from envs.color_door import ColorDoor as Door


@dataclass
class SixRoom:
    top: tuple[int, int]
    size: tuple[int, int]
    door_pos: tuple[int, int]
    color: str | None = None
    locked: bool = False

    def rand_pos(self, env: MiniGridEnv) -> tuple[int, int]:
        """
        Get a random position within the room (not including walls).
        Useful for placing the key and goal.
        """
        top_x, top_y = self.top
        size_x, size_y = self.size
        return env._rand_pos(
            top_x + 1,
            top_x + size_x - 1,
            top_y + 1,
            top_y + size_y - 1,
        )


class SixLockedRoomEnv(MiniGridEnv):
    """
    Custom MiniGrid environment with 6 rooms.

    Layout
    ------
    Central vertical hallway flanked by 3 rooms on each side
    (top, middle, bottom — left and right).

        +--+--+--+
        |L |  |R |   row 0
        +--+H +--+
        |L |  |R |   row 1
        +--+H +--+
        |L |  |R |   row 2
        +--+--+--+

    Task
    ----
    1. Find the red key somewhere in one of the open (green) rooms.
    2. Unlock the locked room (red door).
    3. Reach the goal inside the locked room.

    Visual conventions (via ColorDoor)
    -----------------------------------
    - Open/unlocked doors → GREEN
    - Locked door         → RED
    """

    def __init__(self, size: int = 19, max_steps: int | None = None, **kwargs):
        self.size = size

        if max_steps is None:
            max_steps = 10 * size

        mission_space = MissionSpace(
            mission_func=self._gen_mission
        )

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "Find the red key in one of the green rooms, unlock the red door and reach the goal"

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
        # 2. Central vertical hallway (same width as FourLockedRoomEnv)
        # =========================
        left_hall_x  = width // 2 - 2   # left wall of hallway
        right_hall_x = width // 2 + 2   # right wall of hallway

        for j in range(height):
            self.grid.set(left_hall_x, j, Wall())
            self.grid.set(right_hall_x, j, Wall())

        # =========================
        # 3. Split vertically into 3 rows => 6 rooms total
        # =========================
        self.rooms: list[SixRoom] = []

        split_rows = 3
        room_span  = height // split_rows   # approximate height per row

        for n in range(split_rows):
            y = n * room_span

            # Horizontal walls separating rows (skip outer walls at y=0 and y=height-1)
            if y > 0:
                for i in range(0, left_hall_x):
                    self.grid.set(i, y, Wall())
                for i in range(right_hall_x, width):
                    self.grid.set(i, y, Wall())

            room_w = left_hall_x + 1      # room width  (left side)
            room_h = room_span + 1        # room height (including shared wall)

            # Door placed roughly in the middle of each row strip
            door_y = y + room_span // 2
            door_y = max(1, min(height - 2, door_y))

            # Left room for this row
            self.rooms.append(
                SixRoom(
                    top=(0, y),
                    size=(room_w, room_h),
                    door_pos=(left_hall_x, door_y),
                )
            )

            # Right room for this row
            self.rooms.append(
                SixRoom(
                    top=(right_hall_x, y),
                    size=(room_w, room_h),
                    door_pos=(right_hall_x, door_y),
                )
            )

        # ── 5. Choose locked room & place goal ────────────────────────────────
        locked_room = self._rand_elem(self.rooms)
        locked_room.locked = True

        goal_pos = locked_room.rand_pos(self)
        self.grid.set(*goal_pos, Goal())

        # ── 6. Assign colors + place doors ────────────────────────────────────
        #   Locked room → red door; all unlocked rooms → green door.
        #   ColorDoor handles the actual rendering color based on is_locked.
        for room in self.rooms:
            room.color = "red" if room.locked else "green"
            self.grid.set(*room.door_pos, Door(is_locked=room.locked))

        # ── 7. Place key in a random *open* room ──────────────────────────────
        open_rooms = [r for r in self.rooms if not r.locked]
        key_room = self._rand_elem(open_rooms)
        key_pos = key_room.rand_pos(self)
        self.grid.set(*key_pos, Key(locked_room.color))  # La llave siempre será roja

        # ── 8. Place agent in the hallway ─────────────────────────────────────
        self.place_agent(
            top=(left_hall_x, 1),
            size=(right_hall_x - left_hall_x + 1, height - 2),
        )

        # ── 9. Mission string ─────────────────────────────────────────────────
        self.mission = self._gen_mission()
