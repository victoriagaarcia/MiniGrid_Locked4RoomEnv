"""
eval.py
-------
Carga un modelo PPO ya entrenado, ejecuta N episodios y guarda cada uno
como un fichero de vídeo MP4 en la carpeta `eval_videos/`.

Uso
---
    python eval.py                                         # modelo por defecto
    python eval.py --model models/ppo_fourlocked_final     # ruta explícita
    python eval.py --episodes 5 --size 13                  # 5 episodios, grid 13
    python eval.py --fps 4                                 # vídeo más lento

Requisitos
----------
    pip install stable-baselines3[extra] minigrid opencv-python
"""

import argparse
import os

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from envs.four_locked_room_env import FourLockedRoomEnv
from train_old import RGBFlatWrapper, ShapedRewardWrapper   # reutiliza los wrappers


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

TILE_SIZE = 32   # resolución de cada celda en el vídeo (px)


def make_eval_env(size: int) -> DummyVecEnv:
    """Crea un único entorno envuelto, listo para evaluación."""
    def _init():
        env = FourLockedRoomEnv(size=size, render_mode="rgb_array", tile_size=TILE_SIZE)
        env = ShapedRewardWrapper(env)
        env = RGBFlatWrapper(env)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecTransposeImage(vec_env)   # (H,W,C) → (C,H,W) para SB3
    return vec_env


def get_raw_frame(env_unwrapped: FourLockedRoomEnv) -> np.ndarray:
    """
    Renderiza el entorno sin wrappers para obtener una imagen más grande
    y visualmente clara para el vídeo.
    """
    return env_unwrapped.render()   # (H, W, 3) RGB uint8


def add_hud(
    frame: np.ndarray,
    episode: int,
    step: int,
    reward_total: float,
    carrying: str,
    mission: str,
    done: bool,
    success: bool,
) -> np.ndarray:
    """
    Añade una banda informativa en la parte inferior del frame.
    Usa OpenCV (BGR internamente, convierte antes de escribir).
    """
    h, w = frame.shape[:2]
    hud_h = 72
    canvas = np.zeros((h + hud_h, w, 3), dtype=np.uint8)
    canvas[:h] = frame   # frame ya está en RGB

    # Fondo HUD oscuro
    canvas[h:] = (20, 20, 20)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    small      = 0.45
    medium     = 0.55
    white      = (230, 230, 230)
    yellow     = (50,  220, 255)   # BGR
    green_bgr  = (80,  200,  80)
    red_bgr    = (80,   80, 220)

    # Línea 1: episodio, paso, reward
    cv2.putText(canvas, f"Episode {episode}   Step {step:>4}   Reward {reward_total:+.2f}",
                (8, h + 18), font, medium, white, 1, cv2.LINE_AA)

    # Línea 2: qué lleva el agente
    carry_color = yellow if carrying != "nothing" else white
    cv2.putText(canvas, f"Carrying: {carrying}",
                (8, h + 38), font, small, carry_color, 1, cv2.LINE_AA)

    # Línea 3: misión (truncada si es muy larga)
    mission_short = mission if len(mission) <= 70 else mission[:67] + "..."
    cv2.putText(canvas, mission_short,
                (8, h + 56), font, small, white, 1, cv2.LINE_AA)

    # Línea 4: estado final (solo cuando termina)
    if done:
        if success:
            label, color = "GOAL REACHED!", green_bgr
        else:
            label, color = "TIME OUT / FAILED", red_bgr
        cv2.putText(canvas, label,
                    (w - 200, h + 38), font, medium, color, 2, cv2.LINE_AA)

    return canvas   # sigue en RGB


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluación principal
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Device: {device}")

    # ── Cargar modelo ─────────────────────────────────────────────────────
    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    print(f"[eval] Cargando modelo: {model_path}")
    vec_env = make_eval_env(args.size)
    model   = PPO.load(model_path, env=vec_env, device=device)
    print("[eval] Modelo cargado correctamente.")

    # Referencia al entorno sin wrappers para render y metadatos
    # raw_env: FourLockedRoomEnv = vec_env.envs[0].env.env.env  # desenvuelve wrappers
    raw_env: FourLockedRoomEnv = vec_env.venv.envs[0].unwrapped

    # ── Bucle de episodios ────────────────────────────────────────────────
    for ep in range(1, args.episodes + 1):
        obs = vec_env.reset()

        frames      = []
        step        = 0
        total_rew   = 0.0
        done        = False
        success     = False
        terminated  = False

        print(f"\n[eval] ── Episodio {ep}/{args.episodes} ──────────────────")

        while not done:
            # Frame del entorno (renderizado con tile_size grande para el vídeo)
            frame = get_raw_frame(raw_env)

            # Info de HUD
            carrying_obj = raw_env.carrying
            carrying_str = (
                f"{carrying_obj.color} {carrying_obj.type}"
                if carrying_obj else "nothing"
            )
            mission = getattr(raw_env, "mission", "")

            frame_hud = add_hud(
                frame, ep, step, total_rew,
                carrying_str, mission,
                done=False, success=False,
            )
            frames.append(frame_hud)

            # Acción del agente (determinista)
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = vec_env.step(action)

            total_rew += float(rewards[0])
            step      += 1
            done       = bool(dones[0])

            if done:
                success = not infos[0].get("TimeLimit.truncated", False) and \
                          total_rew > 0.9

        # Frame final con resultado
        frame = get_raw_frame(raw_env)
        frame_hud = add_hud(
            frame, ep, step, total_rew,
            carrying_str, mission,
            done=True, success=success,
        )
        # Congela el frame final unos segundos
        freeze_frames = int(args.fps * args.freeze_secs)
        for _ in range(freeze_frames):
            frames.append(frame_hud)

        result_str = "✓ ÉXITO" if success else "✗ FALLO"
        print(f"[eval] {result_str}  |  Pasos: {step}  |  Reward: {total_rew:.3f}")

        # ── Guardar vídeo ─────────────────────────────────────────────────
        h, w = frames[0].shape[:2]
        out_path = os.path.join(args.out_dir, f"episode_{ep:02d}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))

        for f in frames:
            writer.write(rgb_to_bgr(f))   # OpenCV escribe en BGR

        writer.release()
        print(f"[eval] Vídeo guardado: {out_path}  ({len(frames)} frames @ {args.fps} fps)")

    vec_env.close()
    print(f"\n[eval] Evaluación completada. Vídeos en '{args.out_dir}/'")


# ═══════════════════════════════════════════════════════════════════════════════
# Argparse
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evalúa un modelo PPO y guarda vídeos")
    p.add_argument("--model",        type=str,   default="models/ppo_fourlocked_final",
                   help="Ruta al modelo guardado (con o sin .zip)")
    p.add_argument("--episodes",     type=int,   default=3,
                   help="Número de episodios a grabar (default: 3)")
    p.add_argument("--size",         type=int,   default=19,
                   help="Tamaño del grid (debe coincidir con el entrenamiento)")
    p.add_argument("--fps",          type=int,   default=6,
                   help="Frames por segundo del vídeo (default: 6)")
    p.add_argument("--freeze_secs",  type=float, default=2.0,
                   help="Segundos congelados en el frame final (default: 2)")
    p.add_argument("--out_dir",      type=str,   default="eval_videos",
                   help="Carpeta de salida para los vídeos (default: eval_videos/)")
    p.add_argument("--no_deterministic", dest="deterministic", action="store_false",
                   help="Usa política estocástica en vez de determinista")
    p.set_defaults(deterministic=True)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
