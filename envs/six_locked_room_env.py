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
    Custom MiniGrid environment with 6 rooms, mirroring the layout of
    the original MiniGrid LockedRoom environment.

    Layout:
    - Central vertical hallway (2 cells wide)
    - 3 rooms on the left  (3 rows × 1 column)
    - 3 rooms on the right (3 rows × 1 column)
    => 6 rooms total

    Visual layout (H = hallway, L = left room, R = right room):
        +---+--+---+
        | L |H | R |   row 0
        +---+--+---+
        | L |H | R |   row 1
        +---+--+---+
        | L |H | R |   row 2
        +---+--+---+

    ColorDoor logic (same as FourLockedRoomEnv):
        - Locked door  -> rendered RED
        - Unlocked door -> rendered GREEN
        - Key is always red (visualization only; logic is color-agnostic)

    Task:
        - Pick up the red key from one of the unlocked rooms
        - Unlock the locked room's door (red door)
        - Reach the goal inside the locked room
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

        # =========================
        # 4. Choose locked room randomly
        # =========================
        locked_room = self._rand_elem(self.rooms)
        locked_room.locked = True

        goal_pos = locked_room.rand_pos(self)
        self.grid.set(*goal_pos, Goal("yellow"))

        # =========================
        # 5. Assign unique colors + place ColorDoors
        #    Locked door  -> ColorDoor rendered RED  (is_locked=True)
        #    Unlocked door -> ColorDoor rendered GREEN (is_locked=False)
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
        # 6. Place red key in a random unlocked room
        # =========================
        while True:
            key_room = self._rand_elem(self.rooms)
            if key_room != locked_room:
                break

        key_pos = key_room.rand_pos(self)
        self.grid.set(*key_pos, Key("red"))  # Red key — color is visual only

        # =========================
        # 7. Place agent in the central hallway
        # =========================
        self.agent_pos = self.place_agent(
            top=(left_hall_x, 0),
            size=(right_hall_x - left_hall_x, height),
        )

        # =========================
        # 8. Mission string
        # =========================
        self.mission = (
            f"get the {locked_room.color} key from the {key_room.color} room, "
            f"unlock the {locked_room.color} door and go to the goal"
        )
