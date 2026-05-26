#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$APP_DIR/build"
DIST_DIR="$APP_DIR/dist"
VENV_DIR="$APP_DIR/.venv-build"
APP_NAME="Xper Hemodynamic Viewer"

cd "$APP_DIR"

remove_build_path() {
  local path="$1"
  local attempt

  rm -rf "$path" 2>/dev/null || true
  for attempt in 1 2 3 4 5; do
    if [[ ! -e "$path" ]]; then
      return 0
    fi
    find "$path" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
    rmdir "$path" 2>/dev/null && return 0
    sleep 0.2
  done

  rm -rf "$path"
}

remove_build_path "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

remove_build_path "$BUILD_DIR"
remove_build_path "$DIST_DIR"

pyinstaller \
  --name "$APP_NAME" \
  --windowed \
  --onedir \
  --clean \
  --add-data "app.py:." \
  --add-data "assets:assets" \
  --collect-all streamlit \
  --collect-all plotly \
  --collect-all scipy \
  --hidden-import streamlit.web.bootstrap \
  --hidden-import streamlit.runtime.scriptrunner.magic_funcs \
  --hidden-import scipy.signal \
  "packaging/macos_launcher.py"

echo
echo "Built: $DIST_DIR/$APP_NAME.app"
echo "You can zip and share that .app with another Mac user."
