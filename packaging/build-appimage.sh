#!/bin/bash
# Build a self-contained facecomp-x86_64.AppImage: the compiled binary, its
# shared library dependencies (BLAS/LAPACK/dlib/libjpeg/libpng/...), and both
# dlib model files, bundled into one file that runs without needing any of
# those installed on the target machine.
#
# Usage:
#   ./packaging/build-appimage.sh /path/to/shape_predictor_68_face_landmarks.dat /path/to/dlib_face_recognition_resnet_model_v1.dat
#
# Requires: cargo build --release already run (or this script runs it for
# you), and network access the first time to fetch linuxdeploy/appimagetool.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <landmark-model.dat> <encoder-model.dat>" >&2
  exit 1
fi

landmark_model="$1"
encoder_model="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

tools_dir="$repo_root/packaging/.tools"
mkdir -p "$tools_dir"

fetch_tool() {
  local name="$1"
  local url="$2"
  local dest="$tools_dir/$name"
  if [[ ! -x "$dest" ]]; then
    echo "Fetching $name..."
    curl -sS -L -o "$dest" "$url"
    chmod +x "$dest"
  fi
  echo "$dest"
}

linuxdeploy="$(fetch_tool linuxdeploy-x86_64.AppImage \
  https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage)"
appimagetool="$(fetch_tool appimagetool-x86_64.AppImage \
  https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage)"

echo "Building facecomp in release mode..."
(cd "$repo_root" && cargo build --release)

appdir="$work_dir/AppDir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" \
  "$appdir/usr/share/icons/hicolor/256x256/apps" "$appdir/usr/share/facecomp/models"

cp "$repo_root/target/release/facecomp" "$appdir/usr/bin/facecomp"
cp "$landmark_model" "$appdir/usr/share/facecomp/models/shape_predictor_68_face_landmarks.dat"
cp "$encoder_model" "$appdir/usr/share/facecomp/models/dlib_face_recognition_resnet_model_v1.dat"

cat > "$appdir/usr/share/applications/facecomp.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=facecomp
Exec=facecomp
Icon=facecomp
Categories=Utility;
Terminal=true
EOF

# A small pre-baked placeholder icon (64x64 PNG), embedded directly so this
# script doesn't depend on Pillow/ImageMagick being installed.
base64 -d > "$appdir/usr/share/icons/hicolor/256x256/apps/facecomp.png" <<'ICON'
iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAACiElEQVR4nO2bsYrqQBSG/4jxLdzCQvYJBHu1umAhbGMtKPgKaWxUfICUwUpYsLuljTY+gYKtYGGljUiw2FtcXFzRTTJzTk6C83UGzfnP75mZOHO03t7ev/DCZKQFSGMMkBYgjTFAWoA0xgBpAdJk4w74+fk38D0fH39iUPIfi/tBKEzCQXAawmYAReL3cBhBbkBQ4mGSoLhHWEgNeCZcRzDHPW8hMeCRSI5y5YijvQzGlfyz++rONVoVcB88zuWLKrZyBUgm/yieaiUoGSCd/LO4KiZozwFSyVPFj2zArcvSyV+51RG1CiIZwPF0x0EUncpDICnf/hX2VSCJpX+PylB4+f2AUAakZezfE0Z35ApIavlfiarPDAFpAdIE/hgKM/vP53Os1+vv14vFApPJBPV6HY1GA6fTCefzGcPhEPv9nkj674RdtUg2RS+XCzqdzo9rpVIJtVoNrVYLvu+jXC7DcRx0u12KkGSwDYFmswnXdeH7PgBguVxit9shm419I/pX2NQUCgVsNpsf1/r9Plc4ZUgMsG0brut+vx4MBshk0jG/ss0B2+0WxWIRq9UKAGBZFhzHQa/XowhJBtvXNJ1O0W63Yds2AKBarSKXy3GFU4ZtDpjNZsjn8xiPxzgejzgcDhiNRlzhlCExoFKpPLzueR48z6MIwUbgENDZbZEiyk/3dEzVjBgDon4g6cOAZVM06XsAzwij2wyBsG9Mw2qgsnFLdjYoTSxng2mZC6Lo1NoUTUoV6JxZsDRIxIlufCUDqM7mdaE4pic7G4zbBKoeBe0mqTh7hDjisTRIcFUDh9mmT9B0ippeYdMtHuv/Bl/u/wJJx+wHSAuQxhggLUAaY4C0AGn+ARBmDWl4CUP4AAAAAElFTkSuQmCC
ICON

echo "Bundling shared library dependencies..."
"$linuxdeploy" --appimage-extract-and-run \
  --appdir "$appdir" \
  --executable "$appdir/usr/bin/facecomp" \
  --desktop-file "$appdir/usr/share/applications/facecomp.desktop" \
  --icon-file "$appdir/usr/share/icons/hicolor/256x256/apps/facecomp.png"

# Replace linuxdeploy's default AppRun (a plain symlink to the binary) with a
# wrapper that points facecomp at the bundled model files automatically.
rm -f "$appdir/AppRun"
cat > "$appdir/AppRun" <<'EOF'
#!/bin/sh
set -e
HERE="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$HERE/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FACECOMP_LANDMARK_MODEL="${FACECOMP_LANDMARK_MODEL:-$HERE/usr/share/facecomp/models/shape_predictor_68_face_landmarks.dat}"
export FACECOMP_ENCODER_MODEL="${FACECOMP_ENCODER_MODEL:-$HERE/usr/share/facecomp/models/dlib_face_recognition_resnet_model_v1.dat}"
exec "$HERE/usr/bin/facecomp" "$@"
EOF
chmod +x "$appdir/AppRun"

echo "Packaging AppImage..."
ARCH=x86_64 "$appimagetool" --appimage-extract-and-run "$appdir" "$repo_root/facecomp-x86_64.AppImage"

echo "Done: $repo_root/facecomp-x86_64.AppImage"
