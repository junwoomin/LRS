#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export MPLCONFIGDIR="$project_dir/validation/cache/matplotlib"
export XDG_CACHE_HOME="$project_dir/validation/cache"
check_only=false
if [[ "${1:-}" == "--check" ]]; then
  check_only=true
  shift
fi
if (( $# == 0 )); then set -- gui.py; fi
# The checker uses only the standard library and can diagnose a missing/broken .venv.
if command -v python3 >/dev/null 2>&1; then
  checker_python="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  checker_python="$(command -v python)"
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
  checker_python="$project_dir/.venv/bin/python"
else
  echo "[실패] Python을 찾을 수 없습니다. Python 3.9 이상을 설치하고 docs/ASSET_SETUP_ko.md에 따라 .venv를 준비하세요." >&2
  exit 1
fi
"$checker_python" -B "$project_dir/preflight.py" -- "$@"
if [[ "$check_only" == true ]]; then exit 0; fi
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
exec "$project_dir/.venv/bin/python" -B "$@"
