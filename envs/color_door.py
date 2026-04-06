from minigrid.core.world_object import Door
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle
from minigrid.core.constants import COLORS


class ColorDoor(Door):
    """
    Custom door whose color reflects its state:
      - Locked doors   → RED
      - Unlocked doors → GREEN
    """

    def __init__(self, is_locked=False):
        # Asignar el color en base a si está bloqueada o no
        initial_color = "red" if is_locked else "green"
        super().__init__(color=initial_color, is_locked=is_locked)
    
    def toggle(self, env, pos):
        """
        Desbloquea la puerta si el agente tiene la llave.
        """
        # Extraemos si ha habido éxito al abrir la puerta
        success = super().toggle(env, pos)
        # Si se ha abierto la puerta, actualizar el color a verde
        if success and not self.is_locked:
            self.color = "green"
        return success

    def render(self, img):
        """
        Render the door shape according to its state, using the appropriate color.
        """
        door_color = COLORS["red"] if self.is_locked else COLORS["green"]

        if self.is_open:
            # Open door: thin frame on the right edge + door handle
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), door_color)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
        else:
            # Closed door: nested rectangles + circular handle
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), door_color)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))   # outer frame gap
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), door_color)  # inner panel
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))  # inner frame gap

            # Circular handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), door_color)
