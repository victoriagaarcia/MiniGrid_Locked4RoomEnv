from __future__ import annotations

from dataclasses import dataclass

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

from envs.color_door import ColorDoor as Door


@dataclass
class FourRoom:
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


class FourLockedRoomEnv(MiniGridEnv):
    """
    Custom MiniGrid environment with 4 rooms.

    Layout
    ------
    Central vertical hallway flanked by 2 rooms on each side
    (top-left, bottom-left, top-right, bottom-right).

    Task
    ----
    1. Find the key (same color as the locked door) somewhere in the open rooms.
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

        # Mission space para entorno con colores de habitación variados
        # mission_space = MissionSpace(
        #     mission_func=self._gen_mission,
        #     ordered_placeholders=[COLOR_NAMES, COLOR_NAMES],
        # ) 

        # Mission space sin colores (misión fija)
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

    # @staticmethod
    # def _gen_mission(lockedroom_color: str, keyroom_color: str) -> str:
        # Función para entorno con colores de habitación variados
        # return (
        #     f"get the {lockedroom_color} key from the {keyroom_color} room, "
        #     f"unlock the red door and go to the goal"
        # )

    @staticmethod
    def _gen_mission() -> str:
        return "Find the red key in one of the green rooms, unlock the red door and reach the goal"

    def _gen_grid(self, width: int, height: int) -> None:
        # Grid vacío
        self.grid = Grid(width, height)

        # ── 1. Outer walls ────────────────────────────────────────────────────
        for i in range(width): # Paredes horizontales (borde superior e inferior)
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 1, Wall())
        for j in range(height): # Paredes verticales (borde izquierdo y derecho)
            self.grid.set(0, j, Wall())
            self.grid.set(width - 1, j, Wall())

        # ── 2. Central vertical hallway (two columns) ─────────────────────────
        #   left_hall_x  : right wall of left rooms  / left wall of hallway
        #   right_hall_x : left wall of right rooms  / right wall of hallway
        left_hall_x  = width  // 2 - 1
        right_hall_x = width  // 2 + 1

        for j in range(height): # Paredes verticales del pasillo central
            self.grid.set(left_hall_x,  j, Wall())
            self.grid.set(right_hall_x, j, Wall())

        # ── 3. Horizontal divider (splits each side into top / bottom room) ───
        mid_y = height // 2

        # Paredes horizontales del divisor (con hueco para puertas)
        for i in range(1, left_hall_x): # left side
            self.grid.set(i, mid_y, Wall())
        for i in range(right_hall_x + 1, width - 1): # right side
            self.grid.set(i, mid_y, Wall())

        # ── 4. Define the 4 rooms ─────────────────────────────────────────────
        #   Each room gets a door on the hallway wall, vertically centred.
        self.rooms: list[FourRoom] = []

        room_configs = [
            # (top-left corner,            approx size,                     door column,   door row)
            ((0,            0),            (left_hall_x  + 1, mid_y  + 1),  left_hall_x,   mid_y  // 2),
            ((0,            mid_y),        (left_hall_x  + 1, height - mid_y), left_hall_x, mid_y + (height - mid_y) // 2),
            ((right_hall_x, 0),            (width - right_hall_x, mid_y + 1),  right_hall_x, mid_y  // 2),
            ((right_hall_x, mid_y),        (width - right_hall_x, height - mid_y), right_hall_x, mid_y + (height - mid_y) // 2),
        ]

        for (top, size, door_x, door_y) in room_configs:
            # Asegurar que la posición de la puerta esté dentro de los límites del entorno
            door_y = max(1, min(height - 2, door_y))
            self.rooms.append(
                FourRoom(top=top, size=size, door_pos=(door_x, door_y))
            )

        # ── 5. Choose locked room & place goal ────────────────────────────────
        locked_room = self._rand_elem(self.rooms)
        locked_room.locked = True

        goal_pos = locked_room.rand_pos(self)
        self.grid.set(*goal_pos, Goal())
        
        self.locked_room = locked_room
        self.goal_pos = goal_pos

        # ── 6. Assign colors + place doors ────────────────────────────────────
        #   Locked room is red; all unlocked rooms are green.
        for room in self.rooms:
            room.color = "red" if room.locked else "green"
            self.grid.set(*room.door_pos, Door(is_locked=room.locked))

        # ── 7. Place key in a random *open* room ──────────────────────────────
        open_rooms = [r for r in self.rooms if not r.locked]
        key_room = self._rand_elem(open_rooms)
        key_pos = key_room.rand_pos(self)
        # Key color matches the locked room so the agent knows which door to open
        self.grid.set(*key_pos, Key(locked_room.color)) # La llave siempre será roja

        self.key_room = key_room
        self.key_pos = key_pos

        # ── 8. Place agent in the hallway ─────────────────────────────────────
        self.place_agent(
            top=(left_hall_x, 1),
            size=(right_hall_x - left_hall_x + 1, height - 2),
        )

        # ── 9. Mission string ─────────────────────────────────────────────────
        self.mission = self._gen_mission()
