# Pygame GUI 사용법

[English](UI_GUIDE.md) | **한국어**

[프로젝트 소개](README_ko.md)

```bash
./run.sh gui.py
```

먼저 [자산 및 환경 준비](ASSET_SETUP_ko.md)의 지도·체크포인트·환경 준비를 마치고 `cp gui_settings.example.json gui_settings.json`으로 로컬 설정을 만드세요. 한글 글꼴은 별도 설치가 필요합니다.

기본은 고정 정책 주행, 일시정지, CPU입니다. 모드는 우측 상단에서 선택하고, 실제 실행 상태는 제목 옆에 표시합니다. 왼쪽은 경로·차량·신호등 지도, 오른쪽 위는 정책 입력 BEV, 아래는 속도와 주행/보상/학습 탭입니다. 주황색 차량은 ego, 파란색 차량은 NPC입니다.

| 조작 | 실제 동작 |
| --- | --- |
| 시작 / Space | 현재 모드에서 물리와 정책 실행 시작 |
| 일시정지 / Space | 물리·수집·정책 갱신을 멈춤 |
| 새 에피소드 / R | 동일 seed 경로와 차량 상태 초기화, GUI 학습 가중치는 유지 |
| 한 스텝 / N | 일시정지 상태에서 한 스텝 진행 |
| 고정 정책 주행 | 원본 Roach 정책 사용, PPO 갱신 없음 |
| PPO 학습 | 실제 수집·확률비 확인·PPO 갱신, 모델 로컬 저장 |
| 지도 휠 / 드래그 / 방향키 | 확대·지도 이동 |
| 차량 따라가기 / F | ego 중심으로 카메라 복귀 |
| 화면 저장 / S | 이 창만 현재 출력 폴더에 PNG 저장 |
| Tab / Enter | 버튼 포커스 이동·실행 |
| 종료 / Esc / 창 닫기 | 환경·창 종료, 갱신된 GUI 모델 저장 |

처음에는 보상/학습 지표를 기다림으로 표시합니다. 학습하지 않는 모드에 가짜 손실을 표시하지 않습니다. 에피소드 완료 뒤 다시 시작을 누르거나 새 에피소드를 만들 수 있습니다. 실행 중 한 스텝 버튼, 오류 상태의 시작 버튼은 비활성화됩니다. 오류가 나면 물리 진행을 멈추고 실제 오류를 표시하며 새 에피소드로 재시도할 수 있습니다. 자세한 진단은 로컬 `metrics.jsonl`의 error 이벤트에 남습니다.

GUI의 PPO 롤아웃 기본은 32스텝, 1 epoch, 최대 minibatch 16, lr 1e-5입니다. 수집 스텝과 갱신 횟수, 정책/가치 손실, entropy, KL, lr을 학습 탭에서 표시합니다. 짧게 확인하려면 다음처럼 실행하세요. 시작은 사용자가 누릅니다.

```bash
./run.sh gui.py --mode ppo --steps 64 --rollout 4
```

중간 일시정지는 현재 롤아웃을 유지합니다. 새 에피소드/모드 변경은 미완성 롤아웃을 버리며 이미 학습된 가중치는 유지합니다. mode를 바꾼 뒤에는 자동 실행하지 않습니다. GUI를 종료하면 부분 롤아웃에 추가 학습을 수행하지 않습니다. 재실행 때 기존 학습 가중치를 쓰려면 직접 지정합니다.

```bash
./run.sh gui.py --mode ppo --checkpoint <이전출력폴더>/ppo_latest.pth
```

`gui_settings.json`의 `window_size`, `minimum_window_size`, `font_path`가 실제 적용됩니다. 기본 한글 글꼴은 이미 설치된 NanumGothic입니다. 글꼴을 바꾸면 한글 지원 파일을 지정하세요. 창 최소 크기는 1024×760이며 지도와 BEV/텍스트 영역이 겹치지 않도록 구성했습니다.

## 실제 검증 캡처

- [준비 화면](validation/gui_smoke/ready.png)
- [일시정지](validation/gui_smoke/paused.png)
- [실제 PPO 2회 갱신](validation/gui_smoke/learning.png)
- [보상 항목](validation/gui_smoke/rewards.png)
- [오류 표시](validation/gui_smoke/error.png)
- [1024×760 크기](validation/gui_smoke/resized.png)
- [종료 후 재실행](validation/gui_restarted/restarted.png)

`validation/tests/test_gui.py`가 실제 데스크톱 창의 Pygame 이벤트 큐에 클릭/키 이벤트를 보내 제어와 실제 물리/학습 연결을 확인합니다. GUI 테스트의 NaN 행동 오류는 의도적으로 주입한 복구 검증이며 정상 사용 중 발생한 실패가 아닙니다. 결과는 [검증 결과](VALIDATION_ko.md)에 요약했습니다. 원시 실행 로그와 환경 정보는 공개하지 않습니다.
