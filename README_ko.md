# LRS: Low Resource Simulation

[English](README.md)

**상태: 연구 아이디어 단계.** LRS는 **Low Resource Simulation**의 약자입니다. 전체 연구 범위가 커서 구조 스케치까지 진행했습니다.

처음부터 RGB 카메라 영상을 생성하며 학습하는 대신, BEV 또는 occupancy 상태만 제공하는 시뮬레이션에서 주행을 먼저 학습하는 구상입니다. 이후 구조화된 상태로부터 이미지를 생성하고, 그 이미지로 2차 학습을 수행하는 전체 연구 흐름을 목표로 했습니다.

## 학습 단계

| 단계 | 구상한 내용 | 목적 |
| --- | --- | --- |
| 1차 학습 | RGB 렌더링 없이 BEV 또는 OCC 관측으로 시뮬레이션과 주행 학습 | 적은 자원으로 주행 행동과 궤적을 학습·수집 |
| 이미지 생성 | UniScene 같은 occupancy 기반 생성 모델로 구조화된 상태를 카메라 관측으로 변환 | 상태와 이미지가 대응하는 학습 자료 확보 |
| 2차 학습 | 생성한 이미지로 카메라 기반 주행 모델 학습 또는 적응 | 구조화된 상태에서 학습한 행동을 이미지 입력으로 연결 |
| 평가 | 원래 시뮬레이션과 이미지 기반 환경에서 주행 및 전체 자원 비용 비교 | 비용 감소와 전이 효과 검증 |

원하는 그림체나 도메인의 이미지를 생성하는 것은 프로젝트의 목표입니다. 임의의 그림체 지원이 구현되었다는 의미는 아닙니다.

## SDV와의 연결

SDV는 설정과 실험을 다루는 사용자 인터페이스, LRS는 시뮬레이션과 학습을 담당하는 백엔드로 연결하려 했습니다. SDV UI 프로토타입과 FMTC 연동 자료는 [SDV](https://github.com/junwoomin/SDV)에 분리했습니다. LRS까지 포함한 전체 통합은 계획 단계였습니다.

## 미해결 과제

- 도로 형상·차량 이동·신호 상태와 행동에 따른 상태 변화를 BEV/OCC로 표현하는 방법
- 생성 이미지의 기하·캘리브레이션·신호 상태·시간적 일관성
- 1차 정책의 어떤 부분을 2차 모델로 이전할지에 대한 설계
- 이미지 생성과 2차 학습까지 포함한 전체 계산 비용 비교

현재 자료로는 시뮬레이터 구현, 생성 모델 연동, 자원 절감률, 주행 성능을 주장할 수 없습니다.

## 참고 연구

- [UniScene: Unified Occupancy-centric Driving Scene Generation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UniScene_Unified_Occupancy-centric_Driving_Scene_Generation_CVPR_2025_paper.html).
- [UniScene 저자 저장소](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation). LRS에 실제로 통합된 의존성이 아니라 구상을 설명하기 위한 외부 참고 연구입니다.
