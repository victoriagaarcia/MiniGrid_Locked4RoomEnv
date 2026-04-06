"""
manual_play.py
--------------
Visualiza FourLockedRoomEnv y permite moverlo con el teclado.

Controles
---------
  ← → : girar el agente a la izquierda / derecha
  ↑   : avanzar
  Espacio / E : interactuar (recoger llave, abrir/cerrar puerta)
  R   : resetear el entorno (nuevo layout aleatorio)
  Q / Esc : salir

Requisitos
----------
  pip install minigrid pygame

Asegúrate de que este fichero esté en la raíz del proyecto,
al mismo nivel que la carpeta `envs/`.
"""

import sys
import pygame
import numpy as np

# ── Importa el entorno ────────────────────────────────────────────────────────
from envs.four_locked_room_env import FourLockedRoomEnv

# ── Constantes de visualización ───────────────────────────────────────────────
TILE_SIZE    = 24         # píxeles por celda
FPS          = 30
WINDOW_TITLE = "FourLockedRoom - Manual Play"

# Colores UI (RGB)
BLACK  = (0,   0,   0)
WHITE  = (255, 255, 255)
GRAY   = (40,  40,  40)
YELLOW = (255, 220,  50)
GREEN  = (80,  200,  80)
RED    = (220,  60,  60)

# Mapeo teclas → acciones MiniGrid
#   0: girar izquierda  1: girar derecha  2: avanzar
#   3: recoger          4: soltar          5: interactuar (toggle puerta)
KEY_ACTION = {
    pygame.K_LEFT:  0,
    pygame.K_RIGHT: 1,
    pygame.K_UP:    2,
    pygame.K_SPACE: 5,
    pygame.K_TAB:   3,
}


def make_env(size: int = 19) -> FourLockedRoomEnv:
    return FourLockedRoomEnv(
        size=size,
        render_mode="rgb_array",   # renderizamos nosotros con pygame
        highlight=True,            # muestra el campo de visión del agente
        tile_size=TILE_SIZE,
    )


def draw_hud(screen: pygame.Surface, env: FourLockedRoomEnv,
             step: int, reward_total: float, font: pygame.font.Font) -> None:
    """Dibuja el HUD en la parte inferior de la ventana."""
    env_w = env.width  * TILE_SIZE
    env_h = env.height * TILE_SIZE
    hud_rect = pygame.Rect(0, env_h, env_w, screen.get_height() - env_h)
    pygame.draw.rect(screen, GRAY, hud_rect)

    carrying = env.carrying
    if carrying is None:
        carry_text = "Carrying: nothing"
        carry_color = WHITE
    else:
        carry_text = f"Carrying: {carrying.color} {carrying.type}"
        carry_color = YELLOW

    lines = [
        (f"Step: {step}   Total reward: {reward_total:.2f}", WHITE),
        (carry_text, carry_color),
        (env.mission, WHITE),
        ("← → ↑  move  |  Space/E  interact  |  R  reset  |  Q  quit", GRAY),
    ]

    y = env_h + 6
    for text, color in lines:
        surf = font.render(text, True, color)
        screen.blit(surf, (8, y))
        y += font.get_linesize() + 2


def main(size: int = 19) -> None:
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    font = pygame.font.SysFont("monospace", 14)

    env = make_env(size)
    obs, _ = env.reset()

    # Tamaño ventana: grid + HUD
    grid_w = env.width  * TILE_SIZE
    grid_h = env.height * TILE_SIZE
    hud_h  = (font.get_linesize() + 2) * 4 + 12
    screen = pygame.display.set_mode((grid_w, grid_h + hud_h))
    clock  = pygame.time.Clock()

    step         = 0
    reward_total = 0.0
    done         = False
    message      = ""       # mensaje temporal (ej. "¡Goal reached!")
    msg_timer    = 0

    def reset_env():
        nonlocal obs, step, reward_total, done, message, msg_timer
        obs, _ = env.reset()
        step = 0
        reward_total = 0.0
        done = False
        message = "Entorno reiniciado"
        msg_timer = FPS * 2

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_r:
                    reset_env()

                elif not done and event.key in KEY_ACTION:
                    action = KEY_ACTION[event.key]
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    reward_total += reward
                    done = terminated or truncated

                    if terminated and reward > 0:
                        message = "🎉 ¡Goal alcanzado!"
                        msg_timer = FPS * 3
                    elif terminated:
                        message = "Episodio terminado"
                        msg_timer = FPS * 2
                    elif truncated:
                        message = f"Tiempo agotado ({step} pasos)"
                        msg_timer = FPS * 2

        # ── Render ──────────────────────────────────────────────────────────
        frame = env.render()  # numpy array (H, W, 3)
        surf  = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        draw_hud(screen, env, step, reward_total, font)

        # Mensaje temporal centrado sobre el grid
        if msg_timer > 0:
            msg_timer -= 1
            msg_surf = font.render(message, True, YELLOW)
            mx = (grid_w - msg_surf.get_width())  // 2
            my = (grid_h - msg_surf.get_height()) // 2
            # fondo semitransparente
            bg = pygame.Surface((msg_surf.get_width() + 16, msg_surf.get_height() + 8))
            bg.set_alpha(180)
            bg.fill(BLACK)
            screen.blit(bg, (mx - 8, my - 4))
            screen.blit(msg_surf, (mx, my))

        pygame.display.flip()

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manual play for FourLockedRoomEnv")
    parser.add_argument("--size", type=int, default=19,
                        help="Tamaño del grid (por defecto 19)")
    args = parser.parse_args()
    main(size=args.size)
