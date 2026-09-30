# LRS: Low Resource Simulation

[한국어](README_ko.md)

**Status: research concept.** LRS means **Low Resource Simulation**. It remained an architecture sketch because the full research stack required more time than was available.

The idea is to learn driving behavior first in a simulator that exposes BEV or occupancy states without producing RGB camera images. A later stage would generate camera observations from those structured states and use them for a second round of learning. The goal is a complete workflow from inexpensive behavioral exploration to image-based driving research.

## Two-stage learning design

| Stage | Input and operation | Intended outcome |
| --- | --- | --- |
| 1. Low-resource simulation | Simulate driving with BEV or occupancy observations, without RGB image rendering in the learning loop | Learn driving behavior and collect structured trajectories |
| 2. Image generation | Use an occupancy-conditioned generator, such as the driving-scene generation work UniScene, to synthesize camera observations | Build paired structured-state and image data |
| 3. Secondary learning | Train or adapt an image-based driving model using the generated observations | Transfer the behavior learned from structured states to camera inputs |
| 4. Evaluation | Compare behavior in the original simulator and image-based environments | Measure the total resource cost and transfer quality |

Different visual styles or domains are a project goal. Support for arbitrary styles is not established by the sketch or by naming a generator.

## Relation to SDV / FMTC Studio

SDV was intended to be the user-facing configuration and experiment interface. LRS would provide the simulation and learning backend. The SDV UI prototype and FMTC integration work have their own archive, [SDV](https://github.com/junwoomin/SDV); end-to-end integration with LRS was planned.

## Technical questions left open

- How will BEV/OCC encode road geometry, actor motion, signals, and the action-dependent next state?
- How will generated images preserve geometry, camera calibration, traffic-light state, and temporal consistency?
- Which parts of the first policy transfer to the second model, and which require retraining?
- Does the total cost, including image generation and secondary learning, improve over a rendered baseline?

No simulator implementation, generator integration, resource reduction, or driving score is claimed here.

## Related work

- [UniScene: Unified Occupancy-centric Driving Scene Generation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UniScene_Unified_Occupancy-centric_Driving_Scene_Generation_CVPR_2025_paper.html).
- [UniScene author repository](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation). This work uses occupancy as an intermediate representation for driving-scene generation. It is an external reference, not an implemented dependency of LRS.
