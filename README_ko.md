# LRS_V1 실행 안내

오프라인 BEV 주행 시뮬레이터, 고정 Roach 정책 주행, 별도 PPO 학습을 실행합니다. wandb 로그인·설치·업로드는 필요하지 않습니다. 지도는 기존 원격 `sim_using_data/`를 보존합니다. 원본의 지도·모델·과거 실행 기록은 로컬에 보존하며 새 커밋에는 포함하지 않습니다.

[English](README.md)

## 실행 전 필수 준비

새 clone에는 가상환경, 추출된 지도, 모델 가중치와 개인 설정이 없습니다. 먼저 [ASSET_SETUP.md](ASSET_SETUP.md)에 따라 필수 자산을 준비하세요. 특히 원본 Roach 체크포인트의 공개 출처는 확인되지 않아, 그 파일 없이 GUI·시뮬레이션·PPO를 실행할 수 없습니다. `--from-scratch`도 NPC의 원본 정책 로딩을 제거하지 않습니다.

```bash
cp gui_settings.example.json gui_settings.json
```

개인 `gui_settings.json`은 Git에서 제외됩니다. 한글 글꼴 경로를 로컬 환경에 맞게 수정하세요. `requirements.txt`는 코어 직접 의존성 목록이며 검증된 설치 명세/lockfile이 아닙니다. 실측 OpenCV 4.13의 metadata는 NumPy>=2를 요구하지만 기존 실행 환경은 NumPy 1.26.4이므로 패키지 의존성 불일치가 있습니다. 새 환경 전체 설치와 NumPy 2 checkpoint 호환성은 검증하지 않았습니다. 아래 절차는 호환되는 기존 Python 환경이 있는 경우에만 사용합니다.

## 바로 실행

```bash
./run.sh
```

Pygame 창이 **일시정지 상태**로 열립니다. `시작`을 누르면 고정 정책이 주행합니다. 우측 상단의 `PPO 학습`을 선택하면 실제 정책 수집·갱신 모드로 바뀝니다. 모드를 바꾼 뒤 시작을 누르세요. GUI는 CPU를 사용합니다.

```bash
# GUI를 명시적으로 실행
./run.sh gui.py

# 화면 없이 고정 정책 시뮬레이션 60스텝 (학습 없음)
./run.sh simulate.py --headless --steps 60 --device cpu

# 짧은 PPO 학습과 독립 seed 평가
./run.sh ppo_train.py --steps 64 --rollout 32 --epochs 2 --batch-size 16 \
  --episode-steps 128 --eval-steps 32 --device cpu

# GPU에서 짧은 PPO 실행만 확인
./run.sh ppo_train.py --steps 4 --rollout 4 --epochs 1 --batch-size 4 \
  --episode-steps 8 --skip-eval --device cuda
```

`train_w.py`의 기본 명령도 새 시뮬레이션 GUI로 연결됩니다. 새로운 학습에는 `ppo_train.py`를 사용하세요. 내부의 이전 `train()`/`experimental_main()`은 과거 구조 비교용으로 남겼으며 권장 실행 경로가 아닙니다. 자동으로 5백만 스텝 학습을 시작하지 않습니다.

## 조작과 저장

- 시작/일시정지: 버튼 또는 `Space`. 일시정지 중에는 물리·정책 수집·갱신이 모두 멈춥니다.
- 새 에피소드: 버튼 또는 `R`. 같은 seed의 경로를 다시 만들고 차량 상태를 초기화합니다. GUI에서 이미 갱신한 모델 가중치는 유지합니다.
- 한 스텝: 버튼 또는 `N`. 일시정지 중 한 번만 진행합니다.
- 지도: 휠로 확대, 드래그/방향키로 이동, `F`로 차량 따라가기.
- 화면 저장: 버튼 또는 `S`. 현재 창만 PNG로 저장합니다.
- 키보드 포커스: `Tab`, `Enter`. 종료: 버튼/창 닫기/`Esc`.

기본 출력은 `roach_run/<Town>/<시각>/`입니다. `--output`으로 새 출력 폴더를 지정할 수 있습니다. GUI는 `metrics.jsonl`, `summary.json`, 화면 PNG를 저장하고 PPO 갱신마다 `ppo_latest.pth`를 저장합니다. Headless 시뮬레이션은 실제 BEV 영상 `simulation.mp4`를 저장합니다. CLI 학습은 `ppo_final.pth`, 설정, 지표, 독립 평가 결과를 저장합니다. 기존 `roach/results.log`와 원본 체크포인트를 덮어쓰지 않습니다.

창은 기본 1280×820, 최소 1024×760입니다. 글꼴·크기는 `gui_settings.example.json`을 복사한 개인 `gui_settings.json`에서 바꿀 수 있습니다. UI 상세 설명과 캡처는 [UI_GUIDE.md](UI_GUIDE.md)에 있습니다.

## PPO와 보상

PPO 수집, old logprob, 가치 추정, 부트스트랩, 갱신은 같은 `PPOAgent`를 사용합니다. 원본 Roach의 head 의미/shape, BEV `/255` 정규화, Softplus Beta 분포와 결정론적 행동을 비교해 검증했습니다. 행동은 가속/제동과 조향의 2개 값이며 범위는 `[-1, 1]`입니다. logprob에는 이 범위로 변환하는 Jacobian을 반영합니다.

`TaskEnv`는 기존 시뮬레이션과 관측을 유지하면서 학습용 보상과 5-tuple 종료 계약을 별도로 제공합니다. 고정 정책 시뮬레이션의 기존 보상은 그대로입니다.

| 보상 신호 | 단위와 동작 |
| --- | --- |
| 새 경로 진행 | 처음 도달한 경로 거리(m)에 보상. 왕복 반복으로 같은 거리를 다시 보상받지 않음 |
| 시간·경로/방향 오차 | 실제 `FIXED_DT`(기본 1/30초)에 비례한 비용 |
| 과속·후진·불필요한 정지 | 속도만 높이거나 중앙에서 정지해 보상을 얻지 않음 |
| 제어 변화 | 연속 가속·조향 값 변화의 제곱 비용 |
| 충돌·도로/경로 이탈·빨간 신호 통과·장기 정체 | 하나의 우선 실패 비용으로 종료. 같은 스텝의 진행/완료 보너스 제거 |
| 경로 완료 | 끝 지점 거리·오차·방향 조건을 함께 만족하면 보너스와 종료 |

보상은 환경의 실제 위치, 경로, 충돌/도로 마스크, 신호등 ID/정지선, NPC 위치를 사용합니다. 별도 센서를 가정하지 않습니다. 빨간 신호 앞이나 가까운 전방 NPC 때문에 정지해야 할 때는 불필요한 정지 비용/정체 종료를 면제하지만 양의 정지 보상은 주지 않습니다. 매 스텝의 보상 항목과 raw task metrics를 JSONL로 남깁니다.

```bash
# reward_config.json 값을 바꾼 뒤 사용
./run.sh ppo_train.py --steps 64 --reward-config reward_config.json

# 저장된 가중치에서 다시 시작 (optimizer 상태는 새로 생성)
./run.sh ppo_train.py --steps 64 --checkpoint roach_run/<실행폴더>/ppo_final.pth
```

`reward_config.json`은 전체 기본 보상 설정을 담습니다. CLI의 `--episode-steps`가 지정되면 파일의 스텝 제한보다 우선합니다. 평가에는 별도의 `--eval-steps` 제한을 적용합니다. GUI는 현재 기본 보상 설정을 사용하며 실제 항목은 보상 탭에 표시합니다. `--from-scratch`는 ego PPO 가중치를 새로 초기화합니다. NPC 정책은 기존 Roach 체크포인트를 계속 사용합니다.

진짜 terminal은 가치 부트스트랩을 끄고, 시간 제한 truncated는 종료 직전 관측의 가치를 사용합니다. GAE는 에피소드 경계를 넘어 누적하지 않습니다. CUDA 수집/배치 재계산 정합성을 위해 해당 학습 프로세스에서 cuDNN TF32를 끕니다. 드라이버나 시스템 설정을 변경하지 않습니다.

## 검증과 현재 한계

CPU 64스텝/2갱신, GPU 4스텝/1갱신, 원본 정책 동등성, 갱신 전 확률비≈1, 유한 손실/KL/logprob, 34개 파라미터 텐서 변경, 저장/재로드 동등성을 확인했습니다. 보상 반례와 종료/부트스트랩 12개 테스트, 실제 GUI 반복 조작·오류 복구·창 크기 변경·종료/재실행도 통과했습니다. 검증에서는 wandb import를 강제로 막았습니다.

독립 seed 101/102/103으로 각 최대 32스텝(약 1.07초)의 짧은 평가를 했습니다. 학습 정책의 평균 보상은 3.463, 기존 고정 정책은 3.851로 **이번 짧은 학습이 원본 정책을 개선하지는 않았습니다.** 이 평가는 경로 완주나 충돌 안전성을 판단할 길이가 아닙니다. 장시간 학습/전체 경로 평가, 실제 CARLA 서버·실차 제어는 실행하지 않았습니다.

실행 결과와 실패 수정의 요약, 평가 표는 [VALIDATION.md](VALIDATION.md)에 있습니다. 원시 로그와 환경 덤프는 공개하지 않습니다.

```bash
# CPU 계약/실제 시뮬레이션 회귀
CUDA_VISIBLE_DEVICES='' ./run.sh validation/tests/test_contracts.py
CUDA_VISIBLE_DEVICES='' ./run.sh validation/tests/test_regression.py

# 실제 데스크톱 Pygame 창을 열어 짧게 조작 검증
CUDA_VISIBLE_DEVICES='' SDL_AUDIODRIVER=dummy \
  PYTHONPATH=validation/no_tracker:. ./run.sh validation/tests/test_gui.py
```

## 환경 준비

[ASSET_SETUP.md](ASSET_SETUP.md)의 checkpoint·지도·환경·글꼴 준비를 완료하세요. 호환되는 기존 Python 3.9 환경을 사용한다면 저장소 루트에서 다음처럼 프로젝트 가상환경을 만들 수 있습니다. 깨끗한 환경 전체 설치는 검증하지 않았습니다.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -r requirements-local.txt
cp gui_settings.example.json gui_settings.json
```

원본 체크포인트는 `roach/log/ckpt_11833344.pth`에 별도로 준비합니다. 외부 원본 출처는 미확인입니다. 동일 경로와 고정 SHA256가 일치할 때만 제한된 Gym/NumPy 메타데이터 타입을 허용하며 `weights_only=True`를 유지합니다. 다른 경로/해시에는 예외를 적용하지 않고 unrestricted pickle로 재시도하지 않습니다.

## 공개 소스 범위

`loop.py`의 NPC들은 공유된 고정 Roach 정책으로 결정론적 배치 추론을 수행합니다. PPO는 ego 정책을 갱신합니다. BEV는 지도와 시뮬레이션 객체 상태에서 생성되는 특권 관측이며 카메라 인식 결과가 아닙니다. NPU 통합·학습 수렴·데이터 품질 향상·자원 절감은 이 버전의 검증 성과가 아닙니다. `env.py`, `model/ppo.py`, `data_gan.py`와 이전 Roach 통합은 비교용 소스이며 전체 실행은 재검증하지 않았습니다. 일부 이전 wrapper/criteria는 포함되지 않은 `carla_gym`을 요구합니다.

[VALIDATION.md](VALIDATION.md)의 결과는 원본 환경에서 이미 수행한 기록입니다. 이번 Git 준비에서는 GUI·학습을 재실행하지 않았습니다. 공개 폴더에는 검증 요약·테스트 코드와 프로젝트 화면만 보이는 GUI 캡처를 포함합니다. 원시 로그·환경 덤프·생성 모델·영상은 원본 로컬에 보존하며 공개 대상에서 제외합니다. [LICENSE](LICENSE)를 유지했습니다.
