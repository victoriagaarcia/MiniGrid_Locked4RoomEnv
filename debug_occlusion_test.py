from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minigrid.core.world_object import Wall, Key
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

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

    # 1) Crear entorno base
    env = FourLockedRoomEnv(
        size=19,
        render_mode="rgb_array",
        agent_view_size=7,
    )

    obs, info = env.reset(seed=seed)

    # 2) Forzar posición y orientación del agente
    #    direction: 0=right, 1=down, 2=left, 3=up en MiniGrid
    env.unwrapped.agent_pos = (9, 16)
    env.unwrapped.agent_dir = 3  # mirando hacia arriba

    # 3) Colocar una pared delante del agente
    #    Si el agente está en (9,16) mirando arriba, delante está (9,15)
    wall_pos = (9, 15)
    env.unwrapped.grid.set(*wall_pos, Wall())

    # 4) Colocar una llave detrás de la pared
    #    Por ejemplo en (9,14), una celda más arriba
    key_pos = (9, 14)
    env.unwrapped.grid.set(*key_pos, Key("red"))

    # 5) Render completo del entorno para referencia
    full_render = env.render()

    # 6) Crear el wrapper parcial RGB correcto
    wrapped_env = FourLockedRoomEnv(
        size=19,
        render_mode="rgb_array",
        agent_view_size=7,
    )
    wrapped_env = RGBImgPartialObsWrapper(wrapped_env)
    wrapped_env = ImgObsWrapper(wrapped_env)

    wrapped_obs, wrapped_info = wrapped_env.reset(seed=seed)

    # 7) Repetir la misma manipulación sobre el entorno envuelto
    wrapped_base = wrapped_env.unwrapped
    wrapped_base.agent_pos = (9, 16)
    wrapped_base.agent_dir = 3
    wrapped_base.grid.set(*wall_pos, Wall())
    wrapped_base.grid.set(*key_pos, Key("red"))

    # 8) Obtener observación parcial actualizada
    #    Hacemos un "fake refresh" con render del wrapper
    # Entorno base real
    wrapped_base = wrapped_env.unwrapped    

    # Wrapper intermedio RGB    
    rgb_wrapper = wrapped_env.env   

    # 1) obs parcial simb�lica del entorno base 
    raw_obs = wrapped_base.gen_obs()    

    # 2) convertirla a dict con imagen RGB parcial  
    rgb_obs = rgb_wrapper.observation(raw_obs)  

    # 3) extraer solo la imagen 
    partial_obs = wrapped_env.observation(rgb_obs)  

    print("Render completo:", full_render.shape)
    print("Observación parcial:", partial_obs.shape)

    save_image(full_render, "debug_occlusion/full_render.png")
    save_image(partial_obs, "debug_occlusion/partial_obs.png")

    print("Imágenes guardadas en:")
    print(" - debug_occlusion/full_render.png")
    print(" - debug_occlusion/partial_obs.png")

    env.close()
    wrapped_env.close()


if __name__ == "__main__":
    main()