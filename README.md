# 🔑 Deep RL Navigation in Locked Room Environments

Entrenamiento de agentes PPO/RecurrentPPO en entornos de navegación con habitaciones y llaves, construidos sobre [MiniGrid](https://github.com/Farama-Foundation/Minigrid).

---

## 📋 Descripción

Este proyecto implementa dos entornos de navegación personalizados basados en MiniGrid, en los que un agente debe recoger una llave roja, abrir la puerta cerrada correspondiente y alcanzar la meta. Se estudian cuatro configuraciones experimentales cruzando dos entornos (4 y 6 habitaciones) con dos regímenes de observación (observación completa y parcial), comparando PPO estándar con RecurrentPPO.

### Entornos implementados

| Entorno | Habitaciones | Descripción |
|---|---|---|
| `FourLockedRoomEnv` | 4 | Pasillo central vertical, 2 habitaciones a cada lado. |
| `SixLockedRoomEnv` | 6 | Pasillo central vertical, 3 habitaciones a cada lado (layout del LockedRoom original de MiniGrid). |

**Lógica de puertas y colores:**
- Puertas **abiertas** → color **verde** (`ColorDoor`)
- Puerta **cerrada** (locked) → color **rojo**, contiene la meta
- Llave → color **rojo**, colocada aleatoriamente en una habitación abierta

---

## 🏗️ Estructura del repositorio

```
.
├── envs/
│   ├── four_locked_room_env.py   # Entorno de 4 habitaciones
│   ├── six_locked_room_env.py    # Entorno de 6 habitaciones
│   └── color_door.py             # Clase ColorDoor: puertas con lógica de color
├── utils/
│   └── run_logger.py             # Registro de experimentos en Excel
├── configs/
│   └── ppo_fourlocked.yaml       # Hiperparámetros (leídos por ExperimentConfig)
├── train_fullmap.py              # Entrenamiento PPO con observación completa (MDP)
├── train_partialobs.py           # Entrenamiento RecurrentPPO con observación parcial (POMDP)
├── config.py                     # Carga/guardado de configs YAML
└── eval.py                       # Evaluación y grabación de vídeos del agente
```

---

## 🧪 Configuraciones experimentales

| Experimento | Entorno | Observación | Algoritmo | Política |
|---|---|---|---|---|
| Recurrente completo | 4 rooms | Full map | RecurrentPPO | `CnnLstmPolicy` |
| Parcial 7×7 | 4 rooms | FOV 7×7 (`RGBImgPartialObsWrapper`) | RecurrentPPO | `CnnLstmPolicy` |
| Parcial 5×5 | 4 rooms | FOV 5×5 | RecurrentPPO | `CnnLstmPolicy` |
| 6 habitaciones | 6 rooms | Full / Parcial | PPO / RecurrentPPO | `CnnPolicy` / `CnnLstmPolicy` |

**Resultados principales:**
- PPO con observación completa alcanza ~80–100% de tasa de éxito en evaluación.
- RecurrentPPO rinde por debajo de PPO en observación completa.
- Observación parcial 7×7 logra rendimiento moderado (~70%).
- Observación parcial 5×5 no converge dentro del presupuesto de entrenamiento.
- El entorno de 6 habitaciones muestra rendimiento sustancialmente inferior en todos los escenarios.

---

## ⚙️ Formulación del problema

**Observación completa → MDP:**  
El agente recibe como observación la imagen RGB completa del grid (`ImgObsWrapper`). El estado es totalmente observable, lo que satisface la propiedad de Markov.

**Observación parcial → POMDP:**  
El agente solo percibe su campo de visión local (cono de `agent_view_size` tiles delante de él, renderizado como imagen RGB). Al no ver el mapa completo, el historial importa. Se usa `RecurrentPPO` con `CnnLstmPolicy`, que mantiene un estado LSTM entre pasos del episodio y lo resetea en `done`.

**Espacio de acciones (discreto, 6 acciones):**

| Acción | Descripción |
|---|---|
| `left` | Girar 90° a la izquierda |
| `right` | Girar 90° a la derecha |
| `forward` | Avanzar una casilla |
| `pickup` | Recoger objeto |
| `toggle` | Interactuar con puerta/objeto |

**Reward shaping:**
- Recompensa alta al alcanzar la meta (`+1.0`)
- Recompensas intermedias por recoger la llave (`+0.3`) y abrir la puerta cerrada (`+0.5`)
- Penalización por pasos (`-0.001`, incentiva eficiencia)

---

## 🚀 Instalación

```bash
# Dependencias base
pip install stable-baselines3[extra] shimmy gymnasium minigrid pygame pyyaml openpyxl

# Para entrenamiento con observación parcial (RecurrentPPO)
pip install sb3-contrib

# Para grabación de vídeos de evaluación
pip install opencv-python
```

> **Nota GPU:** Si entrenas con GPU y experimentas `CUDA out of memory`, prueba:
> ```bash
> export PYTORCH_ALLOC_CONF=expandable_segments:True
> ```
> También puedes reducir `n_envs` (de 8 a 4) y `n_steps` (de 1024 a 512) en el config.

---

## 🏋️ Entrenamiento

### Observación completa (MDP) — PPO

```bash
python train_fullmap.py
# Con config personalizado:
python train_fullmap.py --config configs/ppo_fourlocked.yaml
# Opciones adicionales:
python train_fullmap.py --timesteps 3000000 --size 19 --n_envs 8
```

Lanza TensorBoard para monitorizar:

```bash
tensorboard --logdir runs
```

### Observación parcial (POMDP) — RecurrentPPO

```bash
python train_partialobs.py
# Con config personalizado:
python train_partialobs.py --config configs/ppo_fourlocked.yaml
```

El `agent_view_size` (tamaño del FOV) se configura en el archivo YAML. Los valores experimentados son `7` (default MiniGrid) y `5`.

---

## 🎬 Evaluación y vídeos

```bash
# Evaluación básica (modelo final)
python eval.py

# Opciones
python eval.py --model models/best_model       # mejor checkpoint
python eval.py --episodes 5 --fps 4            # más lento para visualización
python eval.py --size 19                       # debe coincidir con el entrenamiento
python eval.py --no_deterministic              # política estocástica
```


---

## 📊 Monitorización de experimentos

El script de entrenamiento registra automáticamente cada experimento en un archivo Excel (`runs/experiments.xlsx`) mediante `utils/run_logger.py`. Cada fila incluye: timestamp, configuración de hiperparámetros, métricas de evaluación y ruta al checkpoint.

---

## 🔧 Hiperparámetros clave (config YAML)

```yaml
env:
  size: 19           # Tamaño del grid NxN
  n_envs: 8          # Entornos paralelos (SubprocVecEnv)
  agent_view_size: 7 # FOV para observación parcial

ppo:
  total_timesteps: 3_000_000
  n_steps: 1024
  batch_size: 256
  n_epochs: 10
  learning_rate: 3e-4
  ent_coef: 0.01
  clip_range: 0.2

model:
  features_dim: 512
  lstm_hidden_size: 256   # Solo RecurrentPPO
  pi_layers: [256, 256]
  vf_layers: [256, 256]
```

---

## 📁 Tecnologías utilizadas

- [MiniGrid](https://github.com/Farama-Foundation/Minigrid) — Entornos de navegación en grid
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO con `CnnPolicy`
- [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) — RecurrentPPO con `CnnLstmPolicy`
- [Gymnasium](https://gymnasium.farama.org/) — Interfaz de entornos RL
- PyTorch · NumPy · OpenCV · TensorBoard

---

## 📌 Trabajo futuro

- Ablación sistemática de distintos tamaños de FOV con presupuestos de entrenamiento ajustados proporcionalmente.
- Entrenamiento más prolongado en el entorno de 6 habitaciones (tanto observación completa como parcial).
- Introducción de lógica de colores más compleja, inspirada en el entorno LockedRoom original de MiniGrid, para aumentar la dificultad de la tarea.

---

## 📄 Autores

Elena Ardura y Victoria García 

Deep Reinforcement Learning (MUIA) — Universidad Pontificia Comillas
