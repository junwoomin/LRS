"""Check local launch prerequisites without loading a policy or starting a simulator."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from roach.utils.asset_manifest import ORIGINAL_CHECKPOINT_PATH, ORIGINAL_CHECKPOINT_SHA256

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = (
    ('numpy', 'numpy'), ('torch', 'torch'), ('cv2', 'opencv-python'),
    ('pygame', 'pygame==2.6.1'), ('shapely', 'shapely'), ('networkx', 'networkx'),
    ('carla', 'carla'), ('gym', 'gym'), ('gymnasium', 'gymnasium'),
    ('omegaconf', 'omegaconf'), ('tqdm', 'tqdm'), ('stable_baselines3', 'stable-baselines3==2.7.0'),
)
ENVIRONMENT_REMEDY = (
    '저장소 루트에서 호환되는 기존 Python 환경을 활성화한 뒤 다음을 실행하세요:\n'
    '  python -m venv --system-site-packages .venv\n'
    '  .venv/bin/python -m pip install --no-deps -r requirements-local.txt\n'
    '기존 .venv가 손상됐다면 별도 보관한 뒤 다시 만드세요. requirements-local.txt는 기본 의존성을 설치하지 않습니다.\n'
    '깨끗한 환경의 설치/버전 호환성은 미검증입니다. docs/ASSET_SETUP_ko.md의 환경 안내를 확인하세요.'
)
MAP_REMEDY = (
    '저장소 루트에서 unzip -l sim_using_data/sim_using_data.zip으로 내부 경로를 확인하세요.\n'
    '  asset_tmp="$(mktemp -d)"\n'
    '  unzip sim_using_data/sim_using_data.zip -d "$asset_tmp"\n'
    '압축을 푼 임시 폴더에서 위에 표시된 파일을 각각 정확한 경로로 복사하세요.\n'
    'ZIP만 있거나 sim_using_data/sim_using_data/ 아래에 중첩된 상태는 준비 완료가 아닙니다.\n'
    'ZIP에 없는 파일은 별도 원본 지도 자산에서 확보하세요. ZIP 내용/동일성은 미검증입니다.\n'
    '필수 파일 목록: docs/ASSET_SETUP_ko.md'
)
CHECKPOINT_REMEDY = (
    '원본 Roach 체크포인트의 공개 다운로드 경로는 미확인입니다. 원본 파일을 별도로 확보해야 합니다.\n'
    '원본을 보유한 경우 저장소 루트에서 (아래 입력 경로는 예시):\n'
    '  mkdir -p roach/log\n'
    '  cp ../local_assets/ckpt_11833344.pth roach/log/ckpt_11833344.pth\n'
    '  sha256sum roach/log/ckpt_11833344.pth\n'
    '--from-scratch와 --checkpoint는 ego 정책만 변경하며 원본 NPC 체크포인트를 대체하지 않습니다.\n'
    '임의 모델로 교체하거나 해시 상수를 변경하지 마세요. docs/ASSET_SETUP_ko.md를 확인하세요.'
)


@dataclass(frozen=True)
class Check:
    label: str
    ok: bool
    detail: str
    remedy: str = ''


@dataclass(frozen=True)
class Launch:
    town: str
    needs_gui: bool
    ego_checkpoint: Path | None = None


def parse_launch(command, environ=None):
    """Mirror town/GUI selection for the documented entrypoints without importing them."""
    environ = os.environ if environ is None else environ
    command = list(command) or ['gui.py']
    target = Path(command[0]).name
    simulation = target in ('simulate.py', 'train_w.py')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--town', default=environ.get('TOWN', 'Town03') if simulation else 'Town03')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--headless', action='store_true', default=simulation and environ.get('VIS', '1') == '0')
    parser.add_argument('--legacy-gui', action='store_true')
    options, _ = parser.parse_known_args(command[1:])
    if not re.fullmatch(r'Town[0-9]{2}(?:HD)?(?:_Opt)?', options.town):
        raise ValueError('잘못된 --town: Town03 같은 지도 이름을 사용하세요. 경로 구분자는 허용하지 않습니다.')
    gui = target == 'gui.py' or (simulation and not options.headless and not options.legacy_gui)
    ego = options.checkpoint if target == 'gui.py' or (target == 'ppo_train.py' and not options.from_scratch) else None
    return Launch(options.town, gui, ego)


def required_map_paths(town):
    """Primary entrypoints use BEVPathFollowEnv's default 5 pixels per meter."""
    data = Path('sim_using_data/data5') / town
    paths = [data / 'world_offset.npy', data / 'das_full.png', data / 'lane_full.png',
             Path('sim_using_data/Town') / (town + '.xodr'),
             Path('sim_using_data/Town') / ('traffic_' + town + '.json'),
             Path('sim_using_data/Town') / (town + '_ped_graph.json')]
    if town in ('Town04', 'Town05'):
        height = Path('sim_using_data/height_estimator') / town
        paths += [data / 'das_full_high.png', data / 'lane_full_high.png',
                  height / 'height_low.png', height / 'height_high.png']
    return paths


def file_problem(path):
    try:
        if not path.is_file():
            return '필수 파일이 없거나 일반 파일이 아닙니다'
        if path.stat().st_size == 0:
            return '파일이 비어 있습니다'
        with path.open('rb') as handle:
            handle.read(1)
    except OSError as error:
        return '파일을 읽을 수 없습니다: ' + str(error)
    return ''


def check_gui_settings(root):
    path = root / 'gui_settings.json'
    remedy = ('저장소 루트에서 cp gui_settings.example.json gui_settings.json을 실행하고, '
              'font_path를 설치된 한글 글꼴 파일의 경로로 수정하세요.')
    problem = file_problem(path)
    if problem:
        return Check('GUI 설정', False, 'gui_settings.json: ' + problem, remedy)
    try:
        settings = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(settings, dict):
            raise ValueError('JSON 최상위 값은 객체여야 합니다')
        for key in ('window_size', 'minimum_window_size'):
            value = settings.get(key, [1024, 760] if key == 'minimum_window_size' else None)
            if (not isinstance(value, list) or len(value) != 2
                    or any(type(size) is not int or size <= 0 for size in value)):
                raise ValueError(key + '는 양의 정수 2개를 담은 배열이어야 합니다')
        font = settings.get('font_path')
        if not isinstance(font, str) or not font:
            raise ValueError('font_path에 설치된 한글 글꼴 파일을 지정하세요')
        problem = file_problem(root / font)
        if problem:
            raise ValueError('font_path=' + font + ': ' + problem)
    except (OSError, UnicodeError, ValueError) as error:
        return Check('GUI 설정', False, 'gui_settings.json JSON/설정 오류: ' + str(error), remedy)
    return Check('GUI 설정', True, '설정과 글꼴 파일 경로 확인 (글꼴 내용/화면 출력 미검증)')


def check_assets(root, launch):
    paths = required_map_paths(launch.town)
    problems = []
    for relative in paths:
        problem = file_problem(root / relative)
        if problem:
            problems.append(str(relative) + ': ' + problem)
    checks = [Check(launch.town + ' 지도', not problems,
                    '\n'.join(problems) if problems else str(len(paths)) + '개 필수 파일 경로/읽기 확인 (내용 미검증)',
                    MAP_REMEDY if problems else '')]
    path = root / ORIGINAL_CHECKPOINT_PATH
    problem = file_problem(path)
    if path.is_symlink():
        problem = '심볼릭 링크는 원본 제한 로더의 고정 경로 조건을 충족하지 않습니다. 실제 파일을 복사하세요'
    if not problem:
        try:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != ORIGINAL_CHECKPOINT_SHA256:
                problem = 'SHA256 불일치\n  실제: ' + actual
        except OSError as error:
            problem = 'SHA256 계산 실패: ' + str(error)
    detail = (ORIGINAL_CHECKPOINT_PATH + ': ' + (problem or 'SHA256 일치')
              + '\n  기대: ' + ORIGINAL_CHECKPOINT_SHA256)
    checks.append(Check('원본 Roach 체크포인트 (NPC 필수)', not problem, detail,
                        CHECKPOINT_REMEDY if problem else ''))
    config = Path('roach/config/config_agent.yaml')
    problem = file_problem(root / config)
    checks.append(Check('정책 설정', not problem, str(config) + ': ' + (problem or '파일 확인'),
                        'Git에 포함된 roach/config/config_agent.yaml 파일을 복원하세요.' if problem else ''))
    if launch.ego_checkpoint is not None and launch.ego_checkpoint != Path(ORIGINAL_CHECKPOINT_PATH):
        problem = file_problem(root / launch.ego_checkpoint)
        checks.append(Check('지정된 ego 체크포인트', not problem, str(launch.ego_checkpoint) + ': ' + (problem or '파일 확인 (내용 미검증)'),
                            '--checkpoint를 실제 ego 가중치 파일로 수정하세요. PPO를 새로 초기화하려면 --from-scratch를 사용하세요. 원본 NPC 파일은 계속 필요합니다.' if problem else ''))
    if launch.needs_gui:
        checks.append(check_gui_settings(root))
    return checks


# Imports execute in the project's interpreter, including any inherited system-site-packages.
# -I excludes the invoking host's user site/PYTHONPATH; no project/ML code is imported here.
PROBE_CODE = r'''
import contextlib, importlib, io, json, sys
results = []
required_api = {'torch': ('serialization.safe_globals',), 'carla': ('Map',),
                'cv2': ('imread',), 'pygame': ('Surface',)}
for name in json.loads(sys.argv[1]):
    result = {'module': name, 'ok': False}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module = importlib.import_module(name)
            for attribute in required_api.get(name, ()):
                value = module
                for part in attribute.split('.'):
                    value = getattr(value, part)
            result.update(ok=True, version=str(getattr(module, '__version__', '버전 정보 없음')))
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    results.append(result)
print('__LRS_PREFLIGHT_JSON__' + json.dumps({
    'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
    'python': list(sys.version_info[:3]), 'dependencies': results,
}, ensure_ascii=False))
'''


def check_environment(root, timeout=45):
    environment = root / '.venv'
    python = environment / 'bin/python'
    if not (environment / 'pyvenv.cfg').is_file() or not python.is_file() or not os.access(python, os.X_OK):
        return [Check('Python 가상환경', False, '.venv/pyvenv.cfg와 실행 가능한 .venv/bin/python이 필요합니다. 의존성 검사도 .venv를 준비한 후 수행됩니다.', ENVIRONMENT_REMEDY)]
    try:
        probe = subprocess.run([str(python), '-I', '-B', '-c', PROBE_CODE,
                                json.dumps([module for module, _ in DEPENDENCIES])],
                               cwd=root, text=True, encoding='utf-8', errors='replace',
                               capture_output=True, timeout=timeout)
        if probe.returncode:
            detail = '가상환경 Python/의존성 검사 프로세스 종료 코드: ' + str(probe.returncode)
            detail += '\n' + (probe.stderr or probe.stdout).strip()[-2000:]
            return [Check('Python 가상환경', False, detail, ENVIRONMENT_REMEDY)]
        data = json.loads(probe.stdout.rsplit('__LRS_PREFLIGHT_JSON__', 1)[1])
        valid = (Path(data['prefix']).resolve() == environment.resolve()
                 and data['prefix'] != data['base_prefix'] and tuple(data['python']) >= (3, 9))
        if not valid:
            return [Check('Python 가상환경', False,
                          '프로젝트 .venv의 Python 3.9 이상이 필요합니다. 감지: ' + str(data['python']) + ', prefix=' + data['prefix'],
                          ENVIRONMENT_REMEDY)]
        records = {item['module']: item for item in data['dependencies']}
    except subprocess.TimeoutExpired:
        return [Check('Python 가상환경/의존성', False, '의존성 import 검사 제한 시간(' + str(timeout) + '초)을 초과했습니다.',
                      '멈춘 Python/패키지 설치를 확인하고 python3 preflight.py로 다시 검사하세요. ' + ENVIRONMENT_REMEDY)]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        return [Check('Python 가상환경', False, 'Python 검사 실행/응답 오류: ' + str(error), ENVIRONMENT_REMEDY)]
    checks = [Check('Python 가상환경', True, '.venv / Python ' + '.'.join(map(str, data['python'])))]
    for module, package in DEPENDENCIES:
        record = records.get(module, {})
        ok = record.get('ok', False)
        detail = ('import 성공 / ' + record.get('version', '버전 정보 없음')) if ok else record.get('error', '의존성 검사 응답 누락')
        remedy = ''
        if not ok:
            remedy = ('프로젝트 가상환경에 설치/복구하세요:\n  .venv/bin/python -m pip install ' + shlex.quote(package)
                      + '\nOS/CPU/CUDA 및 NumPy/OpenCV 호환 버전은 docs/ASSET_SETUP_ko.md를 확인하세요. 설치 명령의 성공은 미검증입니다.')
            if 'libGL.so' in detail:
                remedy += '\nUbuntu/Debian에서 libGL.so.1 누락은 시스템 libgl1 패키지를 설치한 뒤 다시 확인하세요: sudo apt-get install libgl1'
        checks.append(Check('의존성 ' + module, ok, detail, remedy))
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description='LRS 실행 전 필수 준비 검사 (시뮬레이션/학습을 실행하지 않음)')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='검사할 실행 파일과 인자; 기본 gui.py')
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        launch = parse_launch(command)
    except ValueError as error:
        print('[실패] 실행 인자: ' + str(error), file=sys.stderr)
        return 1
    print('LRS 실행 전 준비 검사 / 지도: ' + launch.town)
    checks = check_assets(ROOT, launch) + check_environment(ROOT)
    for check in checks:
        print(('\n[통과] ' if check.ok else '\n[실패] ') + check.label + '\n' + check.detail)
        if not check.ok:
            print('해결 방법:\n' + check.remedy)
    if any(not check.ok for check in checks):
        print('\n필수 준비 항목이 충족되지 않아 실행을 중단합니다. docs/ASSET_SETUP_ko.md를 확인하세요.')
        return 1
    print('\n실행 전 준비 검사를 통과했습니다. 경로·읽기 가능 여부·원본 해시와 의존성 import를 확인했습니다.\n'
          '실제 주행·학습 성공, 지도 내용의 정확성, CUDA/디스플레이 동작은 검증하지 않았습니다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
