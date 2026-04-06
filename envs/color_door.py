from minigrid.core.world_object import Door
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle
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
            # Render door frame and thinner rectangle for open door
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), door_color)  # Door frame
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))  # Door handle
        
        else:
            # Render closed door
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), door_color)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))  # Outer frame
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), door_color)  # Inner frame
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))  # Final outline for door

            # Draw door handle (manillar circular)
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), door_color)  # Door handle (circle)
    
    # Definir función toggle
    # def toggle(self, env, pos):
    #     return