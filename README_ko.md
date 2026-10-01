# LRS: Low Resource Simulation

[English](README.md)

**상태: 초기 연구 프로토타입. 아직 미완성이며, 실행 중 오류가 발생할 수 있고 해결되지 않은 버그가 남아 있습니다.**

LRS는 BEV 관측을 이용해 주행 시뮬레이션과 행동 학습을 연구하는 프로젝트입니다. 주행 정책을 학습하고, 다양한 움직임을 보이는 NPC 차량의 제어에 활용해 차량 간 상호작용에서 더 풍부한 주행 데이터를 얻으려 했습니다. 장기적으로는 자율주행 정책을 NPU에 배포하고, NPU 추론으로 NPC들을 제어하는 구상이었습니다. 데이터 품질 향상과 자원 절감은 연구 목표이며, 이 공개 버전에서 측정된 성과는 아닙니다.

**이 버전의 RL 환경에서는 활성 상태인 모든 NPC 차량을 강화학습 정책으로 제어합니다.** 각 NPC가 자신의 BEV와 차량 상태를 입력으로 받으며 정책 가중치는 공유합니다. 보행자는 별도의 규칙 기반 모델로 움직입니다. 현재 NPC는 사전 학습된 체크포인트로 결정적 추론을 수행하고, 학습 스크립트는 ego 에이전트의 PPO 업데이트를 포함합니다. 모든 NPC가 실행 중 각각 독립적으로 학습하는 구조는 아닙니다.

## 구현된 프로토타입

- CARLA/OpenDRIVE 지도 자산을 활용하는 로컬 bicycle-model 차량 동역학, 경로 생성, 신호등 로직, 보행자 이동
- 도로·경로·차선, 차량 이력, 보행자 이력, 신호등 정지선 이력을 포함하는 ego 기준 BEV 관측
- `loop.py`에서 활성 NPC 차량 전체에 대한 강화학습 정책 배치 추론
- 실험용 PPO 학습 스크립트, BEV 뷰어, 로그, 영상 기록, 체크포인트 저장 코드
- 참고용으로 유지한 이전 구조화 데이터 생성 스크립트

NPU 변환·런타임 연동, 행동 다양성 제어, 데이터 품질 향상 검증은 앞으로 구현·검증할 부분입니다. 현재 정책 코드는 PyTorch의 CUDA 또는 CPU에서 실행됩니다.

## 코드 구성

| 경로 | 역할 |
| --- | --- |
| [`train_w.py`](train_w.py) | 주요 실험용 PPO 실행 파일. 저장소 루트에서 실행 |
| [`loop.py`](loop.py) | 현재 BEV 환경과 공유 강화학습 정책 기반 NPC 차량 제어 |
| [`lrs_config.py`](lrs_config.py) | 로컬 체크포인트 경로 설정 |
| [`roach/ppo.py`](roach/ppo.py) | Ego actor-critic, rollout buffer, PPO 업데이트, 체크포인트 처리 |
| [`roach/models/`](roach/models/) | 체크포인트 정책, CNN 특징 추출, 행동 분포, 유지된 ROACH 학습 구성 요소 |
| [`roach/config/config_agent.yaml`](roach/config/config_agent.yaml) | 유지된 ROACH 정책·에이전트 설정 |
| [`sim/`](sim/) | 차량·보행자 동역학, 경로, 신호등, BEV 맵, 기하 유틸리티 |
| [`data_gan.py`](data_gan.py) | 규칙 기반 차량 제어를 사용하는 이전 BEV·궤적 생성 프로토타입. 호환성 수정 필요 |
| [`env.py`](env.py), [`model/ppo.py`](model/ppo.py) | 이전 환경·PPO 구현. 주요 실행 파일에서는 사용하지 않음 |
| `sim_using_data/` | 필수 지도 자산. 업로드된 [ZIP 파일](sim_using_data/sim_using_data.zip)을 내려받아 압축을 풀고 사용 |

ROACH wrapper와 주행 기준 관련 코드는 유지했습니다. 일부는 외부 `carla_gym` 패키지에 의존하며, 해당 패키지는 포함되어 있지 않습니다.

## RL 관측과 제어

| 항목 | 현재 환경의 입출력 |
| --- | --- |
| `birdview` | 기본 크기에서 `float32`, `(15, 192, 192)`, 값 범위 `0..255` |
| BEV 채널 | 도로·경로·차선, 차량 이력 4장, 보행자 이력 4장, 신호등 이력 4장 |
| `state` | Throttle, steer, brake, gear, ego 기준 종방향 속도, ego 기준 횡방향 속도 총 6개 |
| 정책 정규화 | `PpoPolicy` 내부에서 `birdview`를 255로 나눔 |
| 행동 | 가속 명령과 조향. 각각 `[-1, 1]`이며 가속 명령을 throttle/brake로 변환 |
| NPC 정책 | 하나의 사전 학습 정책 공유, NPC별 관측, 배치 단위 결정적 추론 |
| 시뮬레이션 간격 | 주요 학습 설정에서 로컬 동역학을 스텝당 `1/30`초 진행 |

BEV는 지도 자산과 시뮬레이션 내부 객체 상태에서 직접 구성합니다. 카메라로 추정한 관측이 아니라 시뮬레이션에서 제공하는 특권 정보입니다. 신호등 채널은 정지선과 시뮬레이션 신호 상태를 표현하며 영상 기반 신호등 인식 결과가 아닙니다.

## 로컬 실행 준비

원래 사용한 Python·CARLA·PyTorch·Gym 버전은 기록되어 있지 않습니다. `requirements.txt`는 직접 import하는 패키지 목록이며 **검증된 버전 고정 파일이 아닙니다.** CARLA Python API는 로컬 Python 환경과 제공된 지도에 맞아야 합니다. 유지된 코드에서 `gym`과 `gymnasium`을 모두 사용합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

주요 시뮬레이션은 오프라인 지도 자산을 읽고 로컬 차량 동역학을 진행합니다. 이 루프에서도 CARLA Python API 설치가 필요합니다.

### 지도 자산 다운로드 및 압축 해제

지도 자산은 [`sim_using_data/sim_using_data.zip`](sim_using_data/sim_using_data.zip)으로 업로드되어 있습니다. 실행 전에 ZIP을 내려받아 압축을 풀어 주세요. 저장소 루트 아래에 `sim_using_data/data5/<Town>/`, `sim_using_data/Town/` 등의 경로가 생기도록 배치합니다. `sim_using_data/sim_using_data/`처럼 폴더가 중복되지 않도록 확인하세요. 현재 RL 환경은 다음 자산을 읽습니다.

| 위치 | 필요한 내용 |
| --- | --- |
| `sim_using_data/data5/<Town>/` | `world_offset.npy`, `das_full.png`, `lane_full.png`. Town04/Town05는 `das_full_high.png`, `lane_full_high.png`도 필요 |
| `sim_using_data/Town/` | `<Town>.xodr`, `traffic_<Town>.json`, `<Town>_ped_graph.json` |
| `sim_using_data/height_estimator/<Town>/` | 높이·레이어 처리용 `height_low.png`, `height_high.png` |

이전 `env.py`는 `data20/`도 참조합니다. 이전 `data_gan.py`는 `data20/`과 `data2/`를 참조하며, 폴더 이름은 `data2`지만 코드 상수는 2.5 pixels/meter를 가정합니다. 해당 생성기를 사용하기 전에 실제 래스터 해상도를 확인해야 합니다.

### 정책 체크포인트

체크포인트는 Git에서 제외했습니다. 호환되는 사전 학습 ROACH 정책을 `roach/checkpoints/ckpt_11833344.pth`에 배치하거나 로컬 경로를 지정합니다.

```bash
export LRS_CHECKPOINT=/absolute/path/to/ckpt_11833344.pth
```

원본 압축파일의 체크포인트는 `roach/log/ckpt_11833344.pth`에 있습니다. 기본 위치로 옮기거나 `LRS_CHECKPOINT`에 그 경로를 지정하면 됩니다. NPC 로더는 관측·행동 공간이 들어 있는 `policy_init_kwargs`, `policy_state_dict`, `train_init_kwargs`를 요구합니다. 커스텀 학습기가 저장하는 체크포인트 형식은 다르므로 NPC 로더에서 바로 사용할 수 있다고 가정하면 안 됩니다.

### 프로토타입 학습 실행 파일

지도 자산과 호환 체크포인트를 준비한 뒤 저장소 루트에서 실행합니다.

```bash
WANDB=0 VIS=0 TOWN=Town03 TOTAL_TIMESTEPS=1000 python train_w.py
```

이는 실행 진입점 예시이며, **학습이 정상 완료됐다는 검증 결과가 아닙니다.** 아래 문제를 해결한 뒤 학습 결과를 판단해야 합니다. `VIS=1`이면 뷰어를 사용합니다. 체크포인트는 `roach_run/<Town>/`, 영상은 `artifacts/<Town>/`, 로그는 `roach/results.log`에 기록하며 Git에서 제외됩니다.

W&B는 기본적으로 꺼져 있습니다. 사용하려면 `wandb`를 설치하고 `wandb login` 또는 `WANDB_API_KEY`로 로컬 인증한 뒤 `WANDB=1`을 설정합니다. 저장 대상은 `WANDB_PROJECT`와 `WANDB_ENTITY`로 지정합니다. 공개 소스에는 API 키를 포함하지 않습니다.

## 알려진 제한 사항

- **PPO rollout·업데이트 정책 불일치:** `train_w.py`는 고정 체크포인트 정책에서 ego 행동·value·log probability를 수집하면서 별도의 `PPOAgent`를 업데이트합니다. 업데이트된 모델이 다음 rollout을 제어하지 않습니다. 정상적인 on-policy PPO 학습을 주장하려면 이 연결을 수정해야 합니다.
- **체크포인트 호환성:** 두 정책 구현의 파라미터 배치와 저장 메타데이터가 다릅니다. 로딩 중 missing/unexpected key가 출력될 수 있으며, 부분 로딩만으로 올바른 학습 재개가 보장되지는 않습니다. 오래된 체크포인트는 PyTorch 직렬화 환경의 호환성도 확인해야 합니다.
- **이전 데이터 생성기:** `data_gan.py`는 규칙 기반 제어를 사용하며, 현재 `BicycleModel`에 필요한 `id`·`town` 인자를 생략한 호출이 남아 있습니다. RL 기반 NPC 데이터 생성 파이프라인과 별개이며 전체 실행을 위한 수정이 필요합니다.
- **영상 기록:** 유지된 기록 로직은 `bev` 필드를 읽지만 현재 환경은 `birdview`를 반환합니다. 영상 출력 연결은 추가 수정이 필요합니다.
- **재현성:** 의존성 버전, 제공 지도 자산, 전체 시뮬레이션 실행, 학습 수렴, 데이터 분할, NPU 동작은 이번 공개 버전에서 검증하지 못했습니다.

이번 정리에서는 캐시·로그·생성물·하드코딩 인증 정보를 제외하고, 체크포인트 경로를 설정 가능하게 만들고, 이미 로드된 추론 정책을 재사용하도록 했습니다. 관측 공간 선언도 실제 `birdview`와 6개 상태값에 맞췄습니다. Python 문법과 독립적인 설정 검사를 수행했으며, 검토 환경에 외부 지도 자산과 시뮬레이터·ML 의존성이 없어 전체 시뮬레이션·학습은 실행하지 않았습니다.

## 연구 방향

BEV 기반 주행 학습, 정책으로 제어되는 NPC 간 상호작용, 주행 정책의 NPU 배포, 다양한 주행 궤적 수집으로 이어지는 흐름을 목표로 했습니다. 추가 단계로는 구조화된 BEV/occupancy 상태에서 카메라 관측을 생성하고, 대응하는 데이터로 카메라 기반 주행 모델을 학습하는 구상이 있었습니다. 이미지 생성 단계는 현재 코드에 구현되어 있지 않습니다.

[SDV / FMTC Studio](https://github.com/junwoomin/SDV)는 설정과 실험을 다루는 인터페이스로 연결하려 했습니다. 전체 통합은 계획 단계입니다.

이미지 생성 관련 참고 연구: [UniScene 논문](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UniScene_Unified_Occupancy-centric_Driving_Scene_Generation_CVPR_2025_paper.html), [저자 저장소](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation). UniScene은 이번 버전에 통합된 의존성이 아니라 외부 참고 연구입니다.
