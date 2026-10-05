# Pygame GUI guide

**English** | [한국어](UI_GUIDE_ko.md)

[Project overview](README.md)

```bash
./run.sh gui.py
```

First prepare maps, checkpoint, and environment using [Asset and environment setup](ASSET_SETUP.md), then create local settings with `cp gui_settings.example.json gui_settings.json`. A Korean font must be installed separately.

The defaults are fixed-policy driving, paused execution, and CPU. Choose the mode in the upper-right corner; actual execution status appears beside the title. The left pane shows routes, vehicles, and traffic lights. The upper-right pane shows the policy's BEV input; below are speed and driving/reward/training tabs. The orange vehicle is ego; blue vehicles are NPCs. Interface labels remain Korean.

| Control | Actual behavior |
| --- | --- |
| Start / Space | Starts physics and policy execution in the current mode |
| Pause / Space | Stops physics, collection, and policy updates |
| New episode / R | Resets route and vehicle state with the same seed; retains GUI-trained weights |
| Single step / N | Advances one step while paused |
| Fixed-policy driving | Uses the original Roach policy without PPO updates |
| PPO training | Performs real collection, probability-ratio checks, PPO updates, and local model saving |
| Map wheel / drag / arrow keys | Zooms and pans the map |
| Follow vehicle / F | Recenters the camera on ego |
| Screenshot / S | Saves only this window as a PNG in the current output directory |
| Tab / Enter | Moves button focus and activates the focused control |
| Exit / Esc / window close | Closes environment/window and saves updated GUI weights |

Reward/training metrics initially show a waiting state. Modes without training do not display fabricated losses. After an episode ends, press Start again or create a new episode. Single step is disabled while running; Start is disabled during an error. Errors halt physics and show the actual error; New episode can retry. Diagnostics remain in local `metrics.jsonl` error events.

GUI PPO defaults are a 32-step rollout, one epoch, maximum minibatch 16, and learning rate 1e-5. The training tab displays collection steps, update count, policy/value losses, entropy, KL, and learning rate. For a brief check, use the command below, then press Start manually:

```bash
./run.sh gui.py --mode ppo --steps 64 --rollout 4
```

Pausing retains the current rollout. New episode or mode change discards an unfinished rollout while retaining trained weights. Changing mode does not start execution automatically. Closing the GUI does not train on a partial rollout. To reuse trained weights after reopening, specify them explicitly:

```bash
./run.sh gui.py --mode ppo --checkpoint <previous-output-directory>/ppo_latest.pth
```

`window_size`, `minimum_window_size`, and `font_path` in `gui_settings.json` are applied. The default Korean font is an already-installed NanumGothic. If changing the font, choose a file with Korean support. The minimum window is 1024×760, with map, BEV, and text areas arranged to avoid overlap.

## Recorded validation screenshots

- [Ready](validation/gui_smoke/ready.png)
- [Paused](validation/gui_smoke/paused.png)
- [Two actual PPO updates](validation/gui_smoke/learning.png)
- [Reward components](validation/gui_smoke/rewards.png)
- [Error display](validation/gui_smoke/error.png)
- [1024×760 window](validation/gui_smoke/resized.png)
- [Reopened after exit](validation/gui_restarted/restarted.png)

`validation/tests/test_gui.py` sends click/key events through an actual desktop window's Pygame event queue to verify control connections to physics and learning. Its NaN-action error is intentionally injected for recovery testing, not a failure observed during ordinary use. See [Validation results](VALIDATION.md). Raw execution logs and environment information are excluded from public distribution.
