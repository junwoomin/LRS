# 필수 자산과 실행 환경 준비

저장소 루트에서 준비하세요. 새 clone에는 가상환경, 추출된 지도, checkpoint와 개인 GUI 설정이 없습니다. 아래 준비 명령은 이번 게시 작업에서 실행하지 않았습니다.

## 원본 정책 checkpoint

`loop.py`는 ego/NPC 정책을 `roach/log/ckpt_11833344.pth`에서 읽습니다. `ppo_train.py --checkpoint`는 ego 초기화 파일만 바꾸며 환경의 NPC 경로는 그대로입니다. `--from-scratch`도 NPC 파일이 필요합니다. 이전 `LRS_CHECKPOINT` 설정은 이 버전의 실행 경로에서 사용하지 않습니다.

확인된 원본 파일의 크기는 8,890,031 bytes, SHA256는 다음과 같습니다.

```text
1fecec0c8a206a9ac07a6a9b0be77a5b7e95a073b20d9dfc2a7397354355094a
```

이 해시는 제한 로더의 상수와 일치하며 출처 인증이나 배포 권한을 증명하지는 않습니다. 가중치는 공개 커밋에서 제외했습니다. 공개 다운로드 주소·원본 발행자/서명·배포 권한은 확인되지 않았으며 추측한 주소를 제공하지 않습니다.

원본 파일을 적법하게 보유한 사용자는 프로젝트의 `roach/log/`에 별도 복사하고 해시를 확인하세요. 아래는 인접한 `local_assets/`에 파일을 준비한 경우의 상대경로 예시입니다.

```bash
mkdir -p roach/log
cp ../local_assets/ckpt_11833344.pth roach/log/ckpt_11833344.pth
sha256sum roach/log/ckpt_11833344.pth
```

공개 확보 절차가 확인되지 않아 새 사용자의 재현 blocker로 남습니다. 임의 Roach 모델이나 생성된 ego PPO 파일로 NPC 정책을 대체할 수 있다고 가정하지 마세요.

## 지도

기존 [`sim_using_data/sim_using_data.zip`](sim_using_data/sim_using_data.zip)을 변경 없이 보존합니다. 이번 준비에서 그 내용을 다운로드하거나 압축해제하지 않았으므로 내용·정상성·checkpoint 포함 여부는 확인하지 않았습니다.

`unzip -l sim_using_data/sim_using_data.zip`으로 구조를 확인하고 빈 임시 폴더에 풀어 아래 경로로 배치하세요. 기존 로컬 추출 파일을 새 checkout으로 별도 복사할 수도 있습니다. 원본과 기존 ZIP은 덮어쓰지 마세요. 중첩된 `sim_using_data/sim_using_data/`는 소스가 읽지 않습니다.

| 기본 Town03 경로 | 용도 |
| --- | --- |
| `sim_using_data/data5/Town03/world_offset.npy` | 세계 좌표 오프셋 |
| `sim_using_data/data5/Town03/das_full.png` | 도로 마스크 |
| `sim_using_data/data5/Town03/lane_full.png` | 차선 마스크 |
| `sim_using_data/Town/Town03.xodr` | OpenDRIVE 지도 |
| `sim_using_data/Town/traffic_Town03.json` | 신호등 데이터 |
| `sim_using_data/Town/Town03_ped_graph.json` | 보행자 그래프 |

위 파일은 기존 로컬 자료에 존재함을 확인했으며 ZIP 내부의 동일성은 검증하지 않았습니다. Town04/Town05에는 `data5/<Town>/das_full_high.png`, `lane_full_high.png`, `height_estimator/<Town>/height_low.png`, `height_high.png`도 필수입니다. 이전 `env.py`와 `data_gan.py`는 `data20/`, `data2/`도 참조하며 전체 실행은 재검증하지 않았습니다.

## 호환 환경과 GUI 설정

짧은 검증의 주요 호환 기준은 Python 3.9, torch 2.8, Gym 0.26.2/Gymnasium 1.1.1, stable-baselines3 2.7.0, Pygame 2.6.1입니다. CARLA 0.9.15 Python API와 지도 처리 의존성이 필요하며 실행 시 실제 CARLA 서버는 사용하지 않는 오프라인 루프입니다.

`requirements.txt`는 코어 직접 의존성 목록이며 검증된 설치 lockfile이 아닙니다. OpenCV 4.13의 metadata는 NumPy>=2를 요구하지만 기존 짧은 검증은 NumPy 1.26.4에서 수행됐습니다. 이 불일치 때문에 전체 실측 버전을 그대로 pin한 설치 명세를 제공하지 않습니다. 깨끗한 환경 설치와 NumPy 2 checkpoint 호환성은 미검증입니다. PyTorch 빌드는 OS/CPU/CUDA에 맞아야 합니다.

호환되는 기존 Python 환경이 활성화돼 있는 경우 저장소 루트에서 다음처럼 프로젝트 가상환경을 만들 수 있습니다. `requirements-local.txt`는 추가 패키지 두 개만 담으며, 기본 의존성을 전부 설치하는 파일이 아닙니다.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -r requirements-local.txt
cp gui_settings.example.json gui_settings.json
```

`gui_settings.json`은 GUI가 읽는 개인 파일이며 Git에서 제외합니다. 예제의 `font_path`를 설치된 한글 글꼴 경로로 맞추세요. 일부 이전 Roach wrapper/criteria의 `carla_gym`, hydra/h5py 등은 주 실행 검증 범위 밖입니다. 의존성·checkpoint·추출 지도·글꼴을 모두 준비한 후 `./run.sh`를 실행하세요.
