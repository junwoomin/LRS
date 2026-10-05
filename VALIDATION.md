# Validation results

**English** | [한국어](VALIDATION_ko.md)

[Project overview](README.md)

This document summarizes previously completed short execution and contract checks. Publishing preparation did not rerun the GUI, simulation, or training. Raw logs, tracebacks, personal paths, and environment dumps are excluded from public distribution; original local records are preserved.

## Implementation and validation scope

- Removed wandb imports, authentication, remote tracking, and uploads; replaced them with local JSON/JSONL records through `LocalRun`.
- Separated `simulate.py`, `TaskEnv`/`task_reward.py`, `ppo_train.py`, and the Korean Pygame GUI. Default `train_w.py` opens the GUI without automatically launching large-scale training.
- Actual observations are `birdview` (15,192,192) and `state` (6,); actions are two acceleration/braking and steering values. Collection and PPO updates use the same ego policy.
- Legacy checkpoint loading allows restricted Gym/NumPy types only for the exact path/hash and retains `weights_only=True`. There is no unrestricted-pickle fallback.

| Check | Recorded result |
| --- | --- |
| Reward counterexamples, bounded Beta, terminated/truncated GAE | 12 contract tests passed |
| Original policy and PPO initialization | Equivalent head semantics/shapes, BEV normalization, Beta parameters, values, and deterministic actions |
| Actual PPO regression | Pre-update probability ratios near one, finite loss/KL/log probabilities, 34 changed parameter tensors, exact save/reload equality |
| Restricted checkpoint loading | Succeeded for the original allowed path/hash; rejected legacy metadata at another path |
| Fixed-policy headless/original viewer | Eight steps each, actual movement of 1.111 m, eight decoded BEV video frames, clean exit |
| CPU PPO | 64 steps/two updates; pre-update ratio errors 9.11e-5/6.03e-5, KL .01314/.01235 |
| CUDA PPO | Four steps/one update; ratio error .000122, finite metrics, 34 changed tensors, identical save/reload |
| New GUI | Passed start/pause, halted physics, single step, repeated same-seed reset, mode changes, two actual PPO updates, model saving, error recovery, resize, keyboard focus, screenshots, exit/reopen |
| Execution without tracking | Succeeded with wandb imports blocked |

Reproduction tests remain in `validation/tests/` and `validation/no_tracker/`. Actual GUI screenshots are linked in the [GUI guide](UI_GUIDE.md). Tests require prepared maps, checkpoint, and a compatible environment; some open a GUI or perform short policy updates.

## Observed failures and corrections

| Issue | Correction / verification |
| --- | --- |
| Failed Gym/NumPy metadata loading in legacy checkpoint | Applied a restricted type allowlist for the exact path/hash |
| Original training head mismatch, non-finite KL, empty video | Corrected same-policy collection/updates and original head semantics/shapes, inputs, and distribution; used actual RGB BEV video |
| Synthetic mouse-event corruption in older Pygame | Actual GUI control checks passed on Pygame 2.6.1 |
| Different routes on repeated same-seed reset | Applied seed immediately before actual reset; route equivalence passed |
| Initial CUDA single/batch ratio error .001126 | Disabling cuDNN TF32 within the process reduced it to .000122; the actual short entrypoint passed |

GUI recovery testing intentionally injected NaN actions. The [error screenshot](validation/gui_smoke/error.png) does not represent a failure observed during ordinary use.

## Independent-seed evaluation

Training seed was 0; evaluation seeds were 101/102/103. Policies used routes/NPCs from the same seeds; random actions used a separate RNG. Each policy ran for at most 32 steps, about 1.07 seconds of simulation time. No learning occurred during evaluation.

| Policy | Mean task return | Mean new route progress (m) | Mean speed (m/s) | Mean route progress | Collisions | Route-departure terminations | Completions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64-step learned | 3.463 | 5.615 | 5.147 | 1.156% | 0/3 | 0/3 | 0/3 |
| Original fixed | 3.851 | 6.027 | 5.549 | 1.242% | 0/3 | 0/3 | 0/3 |
| Random | -5.144 | 3.658 | 4.630 | 0.767% | 0/3 | 2/3 | 0/3 |

The random policy terminated for route-distance error >2.5 m at steps 24/29 for seeds 102/103 respectively. Road-mask departure and route-departure termination are distinct.

**This short training did not improve on the original policy.** These results confirm bounded execution of learning, termination, and saving contracts. They do not establish generalization, collision safety, or full-route completion, and must not be interpreted as profitability or performance gains.

## Remaining limits and reproduction conditions

- Long training, full routes, multi-town generalization, and all traffic-light crossing/collision scenarios were not evaluated. A real CARLA server, vehicle control, and NPU integration remain unvalidated.
- Rewards depend on offline maps and existing collision masks. Red-light violations are assessed only when route-linked IDs/stop lines exist.
- Restarting PPO from a checkpoint recreates optimizer/rollout state. CPU updates may briefly halt GUI rendering.
- The original checkpoint's public provenance, signature, and redistribution permission are unconfirmed. Matching a file hash does not authenticate its source. Separate preparation is required; see [Asset and environment setup](ASSET_SETUP.md).
- The preserved map ZIP was not downloaded, extracted, or verified during publishing preparation.
- Clean installation and NumPy 2 legacy-checkpoint compatibility remain untested. Recorded OpenCV 4.13 metadata requires NumPy >=2, but execution used NumPy 1.26.4. The complete observed version set is therefore not presented as a verified installation lockfile.

Publishing preparation statically checked syntax, document links, secret/personal-path patterns, byte-for-byte code identity, and protected-tree identity. Excluding raw material summarizes the public evidence and does not add new execution results.
