# Required assets and environment setup

**English** | [한국어](ASSET_SETUP_ko.md)

[Project overview](../README.md)

Prepare assets from the repository root. A fresh clone has no virtual environment, extracted maps, checkpoint, or private GUI settings. The preparation commands below were not executed during the original publishing work.

## Original policy checkpoint

`loop.py` reads the ego/NPC policy from `roach/log/ckpt_11833344.pth`. `ppo_train.py --checkpoint` changes only the ego initialization file; the environment's NPC path remains unchanged. `--from-scratch` also requires the NPC file. The earlier `LRS_CHECKPOINT` setting is not used by this version's execution path.

The verified original file is 8,890,031 bytes, with this SHA256:

```text
1fecec0c8a206a9ac07a6a9b0be77a5b7e95a073b20d9dfc2a7397354355094a
```

The checker and restricted loader share the hash in `roach/utils/asset_manifest.py`; it does not establish provenance or redistribution rights. Weights are excluded from public commits. A public download URL, original publisher/signature, and redistribution permission have not been confirmed. No guessed download URL is provided.

Users who lawfully hold the original file can copy it separately as a regular file into `roach/log/` and verify its hash. A symlink does not satisfy the restricted loader's exact-path condition. This relative-path example assumes the file is in an adjacent `local_assets/` directory:

```bash
mkdir -p roach/log
cp ../local_assets/ckpt_11833344.pth roach/log/ckpt_11833344.pth
sha256sum roach/log/ckpt_11833344.pth
```

The unresolved public acquisition procedure remains a reproduction blocker for new users. Do not assume an arbitrary Roach model or a generated ego PPO checkpoint can replace the NPC policy.

## Maps

The existing [`sim_using_data/sim_using_data.zip`](../sim_using_data/sim_using_data.zip) is preserved unchanged. It was not downloaded or extracted during publishing preparation, so its contents, integrity, and possible checkpoint inclusion were not verified.

Inspect its layout with `unzip -l sim_using_data/sim_using_data.zip`, extract into an empty temporary directory, and arrange the required paths below. Alternatively, copy existing local extracted files separately into the new checkout. Do not overwrite originals or the existing ZIP. The source does not read a nested `sim_using_data/sim_using_data/` directory.

```bash
unzip -l sim_using_data/sim_using_data.zip
asset_tmp="$(mktemp -d)"
unzip sim_using_data/sim_using_data.zip -d "$asset_tmp"
# Inspect the extracted layout, then copy each required file to the paths below.
# The archive's internal layout/content has not been verified; do not assume it is ready to run.
```

| Default Town03 path | Purpose |
| --- | --- |
| `sim_using_data/data5/Town03/world_offset.npy` | World-coordinate offset |
| `sim_using_data/data5/Town03/das_full.png` | Road mask |
| `sim_using_data/data5/Town03/lane_full.png` | Lane mask |
| `sim_using_data/Town/Town03.xodr` | OpenDRIVE map |
| `sim_using_data/Town/traffic_Town03.json` | Traffic-light data |
| `sim_using_data/Town/Town03_ped_graph.json` | Pedestrian graph |

These files were confirmed in the original local assets; equivalence with the ZIP contents was not verified. Town04/Town05 also require `data5/<Town>/das_full_high.png`, `lane_full_high.png`, `height_estimator/<Town>/height_low.png`, and `height_high.png`. Earlier `env.py` and `data_gan.py` also reference `data20/` and `data2/`; their full execution was not revalidated.

## Compatible environment and GUI settings

The short validation used Python 3.9, torch 2.8, Gym 0.26.2/Gymnasium 1.1.1, stable-baselines3 2.7.0, and Pygame 2.6.1 as key compatibility references. The CARLA 0.9.15 Python API and map-processing dependencies are required. Execution uses an offline loop without an actual CARLA server.

`requirements.txt` lists direct core dependencies; it is not a verified installation lockfile. OpenCV 4.13 metadata requires NumPy >=2, but the short validation used NumPy 1.26.4. Because of this inconsistency, the observed versions are not offered as a fully pinned installation specification. Clean installation and NumPy 2 checkpoint compatibility remain untested. Choose a PyTorch build appropriate for your OS/CPU/CUDA.

With a compatible existing Python environment activated, create the project virtual environment from the repository root as follows. `requirements-local.txt` contains only two additional packages, not all base dependencies.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -r requirements-local.txt
cp gui_settings.example.json gui_settings.json
```

`gui_settings.json` is the private settings file read by the GUI and is ignored by Git. Set the example's `font_path` to an installed Korean font. Dependencies of some older Roach wrappers/criteria, including `carla_gym`, hydra, and h5py, are outside the primary execution validation scope.

## Check preparation before execution

The checker uses only the Python standard library. Run it even before a project virtual environment exists to see missing assets and environment instructions:

```bash
python3 preflight.py gui.py

# The launcher runs the same checker; --check stops after the checks.
./run.sh --check
./run.sh --check simulate.py --headless --town Town03
./run.sh --check ppo_train.py --town Town04 --from-scratch
```

| Preparation item | Checks and failure behavior |
| --- | --- |
| Selected town's maps | All six default files must be readable, nonempty regular files at the exact paths; Town04/Town05 also require the four layered files. A ZIP or nested directory is insufficient. |
| Original Roach checkpoint | Always required, including with `--from-scratch` or custom ego `--checkpoint`. SHA256 must match the fixed original identity. The checker reads bytes without deserializing them. |
| Policy YAML / custom ego weights | The tracked policy YAML must exist; a supplied ego checkpoint must exist when the entrypoint uses it. Custom ego weights do not change the original NPC requirement. |
| Project virtual environment | `.venv/pyvenv.cfg`, executable `.venv/bin/python`, actual venv prefix, and Python >=3.9. |
| Major dependencies | Imports `numpy`, `torch`, `cv2`, `pygame`, `shapely`, `networkx`, `carla`, `gym`, `gymnasium`, `omegaconf`, `tqdm`, and `stable_baselines3` in the project venv; checks selected critical APIs, including `torch.serialization.safe_globals`. |
| GUI settings and font | Required for the default GUI and GUI-delegating simulation; omitted for headless simulation/CLI PPO. JSON dimensions and the configured font file are checked. |

Failures print Korean paths, reasons, and repair commands, then return nonzero. No download, extraction, package installation, policy loading, driving, or training starts. After preparation passes, run `./run.sh` or the specific command in the README; the gate runs again before launch.

**Passing preparation does not establish successful driving or learning.** Map contents, checkpoint deserialization/NumPy compatibility, package-wide version compatibility, CUDA, and display initialization remain outside this check. The unresolved public checkpoint source remains a blocker for users who do not already have the original file.

```bash
# Standard-library tests with synthetic assets/dependency stand-ins; no real policy execution
python3 validation/tests/test_preflight.py
```
