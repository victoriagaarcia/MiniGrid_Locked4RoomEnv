from minigrid.core.world_object import Door
from minigrid.utils.rendering import fill_coords, point_in_rect
from minigrid.core.constants import COLORS


class ColorDoor(Door):
    """
    Custom door whose color reflects its state.

    - Locked doors are rendered in RED
    - Unlocked doors are rendered in GREEN

    This only changes visualization, not environment logic.
    """

    def __init__(self, color="grey", is_locked=False):
        super().__init__(color=color, is_locked=is_locked)

    def render(self, img):
        """
        Render the door with color based on state.
        """

        # Choose color depending on state
        if self.is_locked:
            door_color = COLORS["red"]
        else:
            door_color = COLORS["green"]

        if self.is_open:
            # Open door → draw thinner rectangle
            fill_coords(img, point_in_rect(0.3, 0.7, 0.3, 0.7), door_color)
        else:
            # Closed door
            fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.9), door_color)