"""Preparation checks use synthetic assets; no driving or learning is exercised."""
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

FIXTURE_BYTES = b'synthetic checkpoint, never a real Roach policy'
FIXTURE_SHA256 = hashlib.sha256(FIXTURE_BYTES).hexdigest()
MAP_PATHS = (
    'sim_using_data/data5/{town}/world_offset.npy',
    'sim_using_data/data5/{town}/das_full.png',
    'sim_using_data/data5/{town}/lane_full.png',
    'sim_using_data/Town/{town}.xodr',
    'sim_using_data/Town/traffic_{town}.json',
    'sim_using_data/Town/{town}_ped_graph.json',
)
LAYER_PATHS = (
    'sim_using_data/data5/{town}/das_full_high.png',
    'sim_using_data/data5/{town}/lane_full_high.png',
    'sim_using_data/height_estimator/{town}/height_low.png',
    'sim_using_data/height_estimator/{town}/height_high.png',
)
# These stand-ins only test the subprocess/import diagnostics, not package compatibility.
DEPENDENCY_STUBS = {
    'numpy': '', 'torch': 'from types import SimpleNamespace\nserialization = SimpleNamespace(safe_globals=lambda x: None)\n',
    'cv2': 'imread = lambda *args: None\n', 'pygame': 'Surface = object\n',
    'shapely': '', 'networkx': '', 'carla': 'Map = object\n',
    'gym': '', 'gymnasium': '', 'omegaconf': '', 'tqdm': '', 'stable_baselines3': '',
}


def write_file(root, relative, content=b'fixture'):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def prepare_assets(root, town='Town03'):
    for template in MAP_PATHS + (LAYER_PATHS if town in ('Town04', 'Town05') else ()):
        write_file(root, template.format(town=town))
    write_file(root, 'roach/log/ckpt_11833344.pth', FIXTURE_BYTES)
    write_file(root, 'roach/config/config_agent.yaml')
    write_file(root, 'fixture-font.ttf')
    write_file(root, 'gui_settings.json', json.dumps({
        'window_size': [1280, 820], 'font_path': 'fixture-font.ttf',
    }).encode())


def prepare_venv(root, stubs=True):
    venv.EnvBuilder(with_pip=False).create(root / '.venv')
    python = root / '.venv/bin/python'
    site = Path(subprocess.check_output([
        str(python), '-I', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))',
    ], text=True).strip())
    if stubs:
        for module, source in DEPENDENCY_STUBS.items():
            (site / (module + '.py')).write_text(source)
    return site


def copy_checker(root):
    shutil.copy2(PROJECT / 'preflight.py', root / 'preflight.py')
    manifest = (PROJECT / 'roach/utils/asset_manifest.py').read_text()
    from roach.utils.asset_manifest import ORIGINAL_CHECKPOINT_SHA256
    # Only the sandbox copy trusts the synthetic fixture; production has no hash override.
    write_file(root, 'roach/utils/asset_manifest.py',
               manifest.replace(ORIGINAL_CHECKPOINT_SHA256, FIXTURE_SHA256).encode())


def failure_text(checks):
    return '\n'.join(check.label + '\n' + check.detail + '\n' + check.remedy
                     for check in checks if not check.ok)


class AssetChecks(unittest.TestCase):
    def setUp(self):
        import preflight
        self.preflight = preflight
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        prepare_assets(self.root)
        self.hash_patch = patch.object(preflight, 'ORIGINAL_CHECKPOINT_SHA256', FIXTURE_SHA256)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def checks(self, command=None):
        launch = self.preflight.parse_launch(command or ['gui.py'], {})
        return self.preflight.check_assets(self.root, launch)

    def test_prepared_town03_assets_pass_without_deserializing_weights(self):
        self.assertEqual(failure_text(self.checks()), '')

    def test_each_town03_required_file_is_reported(self):
        for template in MAP_PATHS:
            with self.subTest(path=template):
                path = self.root / template.format(town='Town03')
                path.unlink()
                errors = failure_text(self.checks())
                self.assertIn(str(path.relative_to(self.root)), errors)
                self.assertIn('unzip -l', errors)
                path.write_bytes(b'fixture')

    def test_zip_alone_does_not_count_as_extracted_maps(self):
        shutil.rmtree(self.root / 'sim_using_data')
        write_file(self.root, 'sim_using_data/sim_using_data.zip')
        errors = failure_text(self.checks())
        for template in MAP_PATHS:
            self.assertIn(template.format(town='Town03'), errors)

    def test_nested_map_directory_is_rejected_with_layout_help(self):
        data = self.root / 'sim_using_data/data5'
        nested = self.root / 'sim_using_data/sim_using_data'
        nested.mkdir()
        shutil.move(str(data), nested / 'data5')
        self.assertIn('sim_using_data/sim_using_data/', failure_text(self.checks()))

    def test_empty_map_file_fails(self):
        (self.root / MAP_PATHS[0].format(town='Town03')).write_bytes(b'')
        self.assertIn('비어', failure_text(self.checks()))

    def test_directory_cannot_replace_a_map_file(self):
        path = self.root / MAP_PATHS[0].format(town='Town03')
        path.unlink()
        path.mkdir()
        self.assertIn('파일', failure_text(self.checks()))

    def test_layered_towns_require_all_four_extra_files(self):
        for town in ('Town04', 'Town05'):
            prepare_assets(self.root, town)
            self.assertEqual(failure_text(self.checks(['simulate.py', '--headless', '--town', town])), '')
            for template in LAYER_PATHS:
                with self.subTest(town=town, path=template):
                    path = self.root / template.format(town=town)
                    path.unlink()
                    self.assertIn(template.format(town=town), failure_text(self.checks(['simulate.py', '--town', town])))
                    path.write_bytes(b'fixture')

    def test_town03_does_not_require_optional_height_or_stopline_masks(self):
        self.assertEqual(failure_text(self.checks()), '')

    def test_missing_original_checkpoint_is_a_reproduction_blocker(self):
        (self.root / 'roach/log/ckpt_11833344.pth').unlink()
        errors = failure_text(self.checks())
        self.assertIn('roach/log/ckpt_11833344.pth', errors)
        self.assertIn('공개 다운로드', errors)
        self.assertIn('미확인', errors)

    def test_mismatched_hash_reports_expected_and_actual(self):
        wrong = b'wrong policy'
        (self.root / 'roach/log/ckpt_11833344.pth').write_bytes(wrong)
        errors = failure_text(self.checks())
        self.assertIn(FIXTURE_SHA256, errors)
        self.assertIn(hashlib.sha256(wrong).hexdigest(), errors)
        self.assertIn('불일치', errors)

    def test_checkpoint_symlink_is_not_compatible_with_exact_path_loader(self):
        path = self.root / 'roach/log/ckpt_11833344.pth'
        path.rename(self.root / 'elsewhere.pth')
        path.symlink_to(self.root / 'elsewhere.pth')
        self.assertIn('심볼릭 링크', failure_text(self.checks()))

    def test_from_scratch_keeps_original_npc_requirement(self):
        (self.root / 'roach/log/ckpt_11833344.pth').unlink()
        self.assertIn('ckpt_11833344.pth', failure_text(self.checks(['ppo_train.py', '--from-scratch'])))

    def test_custom_ego_checkpoint_does_not_replace_original(self):
        write_file(self.root, 'ego.pth')
        (self.root / 'roach/log/ckpt_11833344.pth').unlink()
        self.assertIn('ckpt_11833344.pth', failure_text(self.checks(['ppo_train.py', '--checkpoint', 'ego.pth'])))

    def test_missing_custom_ego_checkpoint_is_reported(self):
        self.assertIn('missing ego.pth', failure_text(self.checks(['ppo_train.py', '--checkpoint', 'missing ego.pth'])))

    def test_from_scratch_does_not_require_unused_ego_checkpoint(self):
        self.assertEqual(failure_text(self.checks(['ppo_train.py', '--from-scratch', '--checkpoint', 'unused.pth'])), '')

    def test_missing_policy_configuration_is_reported(self):
        (self.root / 'roach/config/config_agent.yaml').unlink()
        self.assertIn('config_agent.yaml', failure_text(self.checks()))

    def test_gui_requires_private_settings_and_font(self):
        (self.root / 'gui_settings.json').unlink()
        self.assertIn('cp gui_settings.example.json gui_settings.json', failure_text(self.checks()))

    def test_invalid_gui_json_is_reported(self):
        (self.root / 'gui_settings.json').write_text('{invalid')
        self.assertIn('JSON', failure_text(self.checks()))

    def test_invalid_gui_dimensions_are_reported(self):
        (self.root / 'gui_settings.json').write_text(json.dumps({'window_size': 'bad', 'font_path': 'fixture-font.ttf'}))
        self.assertIn('window_size', failure_text(self.checks()))

    def test_missing_font_is_reported(self):
        (self.root / 'fixture-font.ttf').unlink()
        self.assertIn('font_path', failure_text(self.checks()))

    def test_headless_execution_does_not_require_gui_settings(self):
        (self.root / 'gui_settings.json').unlink()
        self.assertEqual(failure_text(self.checks(['simulate.py', '--headless'])), '')
        self.assertEqual(failure_text(self.checks(['ppo_train.py'])), '')

    def test_launch_town_matches_entrypoint_defaults_environment_and_equals_form(self):
        parse = self.preflight.parse_launch
        self.assertEqual(parse([], {'TOWN': 'Town05'}).town, 'Town03')
        self.assertEqual(parse(['simulate.py'], {'TOWN': 'Town05'}).town, 'Town05')
        self.assertEqual(parse(['train_w.py'], {'TOWN': 'Town04'}).town, 'Town04')
        self.assertEqual(parse(['ppo_train.py'], {'TOWN': 'Town05'}).town, 'Town03')
        self.assertEqual(parse(['gui.py', '--town=Town04'], {}).town, 'Town04')
        self.assertEqual(parse(['gui.py', '--town', 'Town04', '--town', 'Town05'], {}).town, 'Town05')
        self.assertFalse(parse(['simulate.py'], {'VIS': '0'}).needs_gui)
        self.assertFalse(parse(['simulate.py', '--legacy-gui'], {}).needs_gui)

    def test_town_cannot_escape_asset_directory(self):
        with self.assertRaises(ValueError):
            self.preflight.parse_launch(['gui.py', '--town', '../elsewhere'], {})

    def test_town_abbreviation_matches_runtime_argparse(self):
        self.assertEqual(self.preflight.parse_launch(['gui.py', '--to', 'Town04'], {}).town, 'Town04')


class EnvironmentChecks(unittest.TestCase):
    def setUp(self):
        import preflight
        self.preflight = preflight
        self.tmp = tempfile.TemporaryDirectory(prefix='lrs environment ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_missing_virtual_environment_has_creation_commands(self):
        errors = failure_text(self.preflight.check_environment(self.root))
        self.assertIn('python -m venv --system-site-packages .venv', errors)
        self.assertIn('requirements-local.txt', errors)

    def test_executable_alone_is_not_a_virtual_environment(self):
        path = write_file(self.root, '.venv/bin/python', b'#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
        self.assertIn('pyvenv.cfg', failure_text(self.preflight.check_environment(self.root)))

    def test_broken_interpreter_is_reported_without_traceback(self):
        write_file(self.root, '.venv/pyvenv.cfg')
        path = write_file(self.root, '.venv/bin/python', b'#!/bin/sh\necho broken >&2\nexit 99\n')
        path.chmod(0o755)
        errors = failure_text(self.preflight.check_environment(self.root))
        self.assertIn('99', errors)
        self.assertIn('broken', errors)

    def test_dependency_imports_run_in_the_project_venv(self):
        prepare_venv(self.root)
        self.assertEqual(failure_text(self.preflight.check_environment(self.root)), '')

    def test_missing_dependencies_are_aggregated_with_package_names(self):
        prepare_venv(self.root, stubs=False)
        errors = failure_text(self.preflight.check_environment(self.root))
        for module in DEPENDENCY_STUBS:
            self.assertIn(module, errors)
        self.assertIn('opencv-python', errors)
        self.assertIn('stable-baselines3==2.7.0', errors)

    def test_broken_import_reports_the_original_reason(self):
        site = prepare_venv(self.root)
        (site / 'cv2.py').write_text("raise ImportError('libGL.so.1 cannot be loaded')\n")
        errors = failure_text(self.preflight.check_environment(self.root))
        self.assertIn('libGL.so.1', errors)
        self.assertIn('libgl1', errors)

    def test_old_torch_without_restricted_loader_api_is_rejected(self):
        site = prepare_venv(self.root)
        (site / 'torch.py').write_text('serialization = object()\n')
        self.assertIn('safe_globals', failure_text(self.preflight.check_environment(self.root)))

    def test_host_python_wrapper_cannot_impersonate_the_project_venv(self):
        prepare_venv(self.root)
        python = self.root / '.venv/bin/python'
        python.unlink()
        python.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        python.chmod(0o755)
        self.assertIn('prefix=', failure_text(self.preflight.check_environment(self.root)))

    def test_dependency_probe_timeout_is_reported(self):
        site = prepare_venv(self.root)
        (site / 'numpy.py').write_text('import time\ntime.sleep(10)\n')
        self.assertIn('시간', failure_text(self.preflight.check_environment(self.root, timeout=.1)))


class LauncherRegression(unittest.TestCase):
    def test_missing_assets_stop_before_the_entrypoint(self):
        with tempfile.TemporaryDirectory(prefix='lrs preparation ') as directory:
            root = Path(directory)
            shutil.copy2(PROJECT / 'run.sh', root / 'run.sh')
            copy_checker(root)
            # A real venv satisfies the old launcher, but there are no map/weight assets.
            venv.EnvBuilder(with_pip=False).create(root / '.venv')
            (root / 'gui.py').write_text("from pathlib import Path\nPath('launched').touch()\n")
            result = subprocess.run([str(root / 'run.sh')], cwd=PROJECT.parent,
                                    text=True, capture_output=True, timeout=30)
            self.assertFalse((root / 'launched').exists(), 'run.sh launched with missing mandatory assets')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('world_offset.npy', result.stdout + result.stderr)
            self.assertIn('ckpt_11833344.pth', result.stdout + result.stderr)

    def prepared_launcher(self, root):
        shutil.copy2(PROJECT / 'run.sh', root / 'run.sh')
        copy_checker(root)
        prepare_assets(root)
        prepare_venv(root)
        (root / 'gui.py').write_text("import json, sys\nfrom pathlib import Path\nPath('launched').write_text(json.dumps(sys.argv[1:]))\n")

    def test_check_only_never_launches_and_limits_the_success_claim(self):
        with tempfile.TemporaryDirectory(prefix='lrs launcher ') as directory:
            root = Path(directory)
            self.prepared_launcher(root)
            result = subprocess.run([str(root / 'run.sh'), '--check'], cwd=PROJECT.parent,
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((root / 'launched').exists())
            self.assertIn('실제 주행·학습 성공', result.stdout)
            self.assertIn('검증하지', result.stdout)

    def test_success_launches_from_repository_root_preserving_arguments(self):
        with tempfile.TemporaryDirectory(prefix='lrs launcher ') as directory:
            root = Path(directory)
            self.prepared_launcher(root)
            arguments = ['--town=Town03', '--output', 'a directory with spaces']
            result = subprocess.run([str(root / 'run.sh'), 'gui.py', *arguments],
                                    cwd=PROJECT.parent, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((root / 'launched').read_text()), arguments)

    def test_from_scratch_launcher_still_blocks_missing_original_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepared_launcher(root)
            (root / 'roach/log/ckpt_11833344.pth').unlink()
            (root / 'ppo_train.py').write_text("from pathlib import Path\nPath('launched').touch()\n")
            result = subprocess.run([str(root / 'run.sh'), 'ppo_train.py', '--from-scratch'],
                                    text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'launched').exists())
            self.assertIn('ckpt_11833344.pth', result.stdout + result.stderr)

    def test_hash_mismatch_blocks_launch_after_other_preparation_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepared_launcher(root)
            (root / 'roach/log/ckpt_11833344.pth').write_bytes(b'wrong checkpoint')
            result = subprocess.run([str(root / 'run.sh')], text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'launched').exists())
            self.assertIn('SHA256 불일치', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
