# LRS: Low Resource Simulation

[한국어](README_ko.md)

**Status: early research prototype. This version is incomplete, may fail at runtime, and still contains unresolved bugs.**

LRS explores driving simulation and behavior learning using BEV observations. The purpose is to train driving policies, use them to control NPC vehicles with varied behavior, and collect richer driving data through their interactions. The longer-term goal was to build an autonomous-driving policy for NPU deployment and use NPU inference to control these NPCs. Improved data quality and lower resource cost are research goals; neither has been measured in this release.

**This is the version in which all active NPC vehicles in the RL environment are controlled by a reinforcement-learning policy.** Each NPC receives its own BEV and vehicle state, while the policy weights are shared. Pedestrians use a separate scripted model. The NPCs currently run deterministic inference from a pretrained checkpoint; the training script contains PPO updates for the ego agent, rather than independent online training for every NPC.

## Implemented prototype

- Local bicycle-model vehicle dynamics, route generation, traffic-light logic, and pedestrian motion using CARLA/OpenDRIVE map assets.
- Ego-aligned BEV observations with road, route, lane, vehicle history, pedestrian history, and traffic-light stop-line history.
- Batched RL policy inference for all active NPC vehicles in `loop.py`.
- An experimental PPO training script, BEV viewer, logging, video recording, and checkpoint output.
- An older, separate structured-data export script retained for reference.

NPU export/runtime integration, behavior-diversity controls, and measured data-quality improvements remain future work. The current policy code uses PyTorch on CUDA or CPU.

## Code layout

| Path | Role |
| --- | --- |
| [`train_w.py`](train_w.py) | Main experimental PPO entrypoint; run from the repository root |
| [`loop.py`](loop.py) | Current BEV environment and shared RL policy control of NPC vehicles |
| [`lrs_config.py`](lrs_config.py) | Local checkpoint path configuration |
| [`roach/ppo.py`](roach/ppo.py) | Ego actor-critic, rollout buffer, PPO updates, and checkpoint handling |
| [`roach/models/`](roach/models/) | Checkpoint policy, CNN features, distributions, and retained ROACH training components |
| [`roach/config/config_agent.yaml`](roach/config/config_agent.yaml) | Retained ROACH policy/agent configuration |
| [`sim/`](sim/) | Vehicle/pedestrian dynamics, routing, traffic lights, BEV maps, and geometry helpers |
| [`data_gan.py`](data_gan.py) | Older BEV/trajectory export prototype using scripted vehicle control; requires compatibility fixes |
| [`env.py`](env.py), [`model/ppo.py`](model/ppo.py) | Earlier environment and PPO variants; unused by the main entrypoint |
| `sim_using_data/` | Required external map assets, supplied separately and excluded from Git |

The ROACH wrappers and criteria are retained with the source. Some depend on an external `carla_gym` package, which is not included.

## RL observation and control

| Item | Current environment contract |
| --- | --- |
| `birdview` | `float32`, shape `(15, 192, 192)` at the default size, values in `0..255` |
| BEV channels | Road, route, lane, four vehicle-history masks, four pedestrian-history masks, four traffic-light-history masks |
| `state` | Six values: throttle, steer, brake, gear, ego-frame longitudinal velocity, ego-frame lateral velocity |
| Policy normalization | `PpoPolicy` divides `birdview` by 255 internally |
| Action | Acceleration command and steer, each in `[-1, 1]`; acceleration is mapped to throttle/brake |
| NPC policy | One shared pretrained policy, separate observations per NPC, batched deterministic inference |
| Simulation step | Main training configuration advances the local dynamics at `1/30` second per step |

The BEV inputs are constructed directly from map assets and simulated actor states. They are privileged simulation observations, not estimates from cameras. Traffic-light channels encode stop lines and simulated signal state; they do not perform visual traffic-light recognition.

## Local setup

The original Python, CARLA, PyTorch, and Gym versions were not recorded. `requirements.txt` lists direct imports and is **not a verified dependency lockfile**. CARLA's Python API must be compatible with the local Python environment and supplied maps. Both `gym` and `gymnasium` are imported by the retained code.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The main simulation reads offline map assets and advances local vehicle dynamics. Install the CARLA Python API even when using this offline loop.

### Map assets supplied separately

Place `sim_using_data/` under the repository root. It is intentionally absent from this code release. The current RL environment reads:

| Location | Required content |
| --- | --- |
| `sim_using_data/data5/<Town>/` | `world_offset.npy`, `das_full.png`, `lane_full.png`; `das_full_high.png` and `lane_full_high.png` for Town04/Town05 |
| `sim_using_data/Town/` | `<Town>.xodr`, `traffic_<Town>.json`, `<Town>_ped_graph.json` |
| `sim_using_data/height_estimator/<Town>/` | `height_low.png`, `height_high.png`, used for height/layer handling |

The earlier `env.py` also refers to `data20/`. The older `data_gan.py` refers to both `data20/` and `data2/`; its constants assume 2.5 pixels/meter even though the folder is named `data2`. Confirm the actual raster scale before using that exporter.

### Policy checkpoint

Checkpoints are excluded from Git. Place a compatible pretrained ROACH policy at `roach/checkpoints/ckpt_11833344.pth`, or point to a local file:

```bash
export LRS_CHECKPOINT=/absolute/path/to/ckpt_11833344.pth
```

If using the original archive, its checkpoint is under `roach/log/ckpt_11833344.pth`; move it to the default location or set `LRS_CHECKPOINT` to that path. The NPC loader expects `policy_init_kwargs` with observation/action spaces, `policy_state_dict`, and `train_init_kwargs`. The custom trainer's saved checkpoint format is different, so it cannot be assumed to work directly with the NPC loader.

### Prototype training entrypoint

After providing map assets and a compatible checkpoint, run from the repository root:

```bash
WANDB=0 VIS=0 TOWN=Town03 TOTAL_TIMESTEPS=1000 python train_w.py
```

This is an entrypoint example, **not a successful end-to-end training result**. Resolve the limitations below before using the output as training evidence. Set `VIS=1` to enable the viewer. The script writes checkpoints under `roach_run/<Town>/`, videos under `artifacts/<Town>/`, and a log to `roach/results.log`; these outputs are ignored by Git.

W&B is disabled by default. To enable it, install `wandb`, authenticate locally using `wandb login` or `WANDB_API_KEY`, and set `WANDB=1`. `WANDB_PROJECT` and `WANDB_ENTITY` configure the destination. No API key is included in the public source.

## Known limitations

- **PPO rollout/update mismatch:** `train_w.py` collects ego actions, values, and log probabilities from the frozen checkpoint policy while updating a separate `PPOAgent`. The trained agent does not drive the subsequent rollout. This connection needs correction before claiming valid on-policy PPO training.
- **Checkpoint compatibility:** the two policy implementations use different parameter layouts and save metadata. Loading reports missing/unexpected keys; partial loading does not establish a valid resumed model. Older checkpoints may also require a compatible PyTorch serialization environment.
- **Older exporter:** `data_gan.py` uses scripted controls and older `BicycleModel` calls that omit the current `id` and `town` arguments. It is not the RL-controlled NPC data pipeline and is not ready for end-to-end use.
- **Video path:** the retained recording logic queries a `bev` field, while the current environment returns `birdview`. Video export needs further adaptation.
- **Reproducibility:** dependency versions, supplied map assets, full simulation execution, learning convergence, dataset splits, and NPU behavior have not been validated for this public release.

This cleanup removes caches, logs, generated outputs, and embedded credentials; makes the checkpoint path configurable; reuses the already loaded inference policy; and aligns the declared observation space with the actual `birdview`/six-value state. Verification covered Python syntax and isolated configuration checks. Full simulation/training was not run because the external assets and simulator/ML dependencies were unavailable in the review environment.

## Research direction

The intended workflow is BEV-based driving learning, interaction among policy-controlled NPCs, NPU deployment of driving policies, and collection of diverse driving trajectories. An additional planned stage would generate camera observations from structured BEV/occupancy states and train camera-based driving models from the paired data. That image-generation stage is not implemented here.

[SDV / FMTC Studio](https://github.com/junwoomin/SDV) was intended to provide the configuration and experiment interface. End-to-end integration remains planned.

Related image-generation reference: [UniScene paper](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UniScene_Unified_Occupancy-centric_Driving_Scene_Generation_CVPR_2025_paper.html) and [author repository](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation). UniScene is an external reference, not a dependency integrated into this release.
