#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export MPLCONFIGDIR="$project_dir/validation/cache/matplotlib"
export XDG_CACHE_HOME="$project_dir/validation/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  echo "프로젝트 .venv가 없습니다. README.md의 환경 재구성 안내를 확인하세요." >&2
  exit 1
fi
if (( $# == 0 )); then set -- gui.py; fi
exec "$project_dir/.venv/bin/python" -B "$@"
