#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

set -e

THIS_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUILD_DIR="$(mktemp -d /tmp/ChituBoot.XXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT

#################################################################
# Parse arguments

usage() {
    echo "Usage 1 (apptainer): $0 <apptainer_image.sif> -o <output_file>" >&2
    echo "    <apptainer_image.sif>: Path to the .sif image file to bundle." >&2
    echo "Usage 2 (docker): $0 <docker_image:tag> -o <output_file>" >&2
    echo "    <docker_image:tag>: Docker image to bundle." >&2
    echo "" >&2
    echo "Options:" >&2
    echo "    -o, --output-file <file>: Path to the AppImage output file (required)." >&2
    echo "    --online: Make a smaller bundle without container image inside. Users will pull from online resouce." >&2
    echo "    -h, --help: Show this help message." >&2
}

OUTPUT_FILE=""
USE_ONLINE_DOCKER=0

if ! PARSED=$(getopt -o ho: --long help,output-file:,online -n "$0" -- "$@"); then
    usage
    exit 1
fi
eval set -- "$PARSED"

while true; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -o|--output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --online)
            USE_ONLINE_DOCKER=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal parsing error" >&2
            usage
            exit 1
            ;;
    esac
done

# After getopt reordering, the image name is the remaining positional argument
if [[ $# -lt 1 ]]; then
    echo "Error: missing image argument" >&2
    usage
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: output file is required (-o/--output-file)" >&2
    usage
    exit 1
fi

IMAGE_NAME="$1"
IMAGE_IS_APPTAINER=0
IMAGE_IS_DOCKER=0
if [[ -f "$IMAGE_NAME" ]]; then
    echo "This bundle will include Apptainer image $IMAGE_NAME"
    IMAGE_IS_APPTAINER=1
else
    echo "This bundle will include Docker image $IMAGE_NAME"
    IMAGE_IS_DOCKER=1
fi

#################################################################
# Pack boot scripts with PyInstaller

# 0) Select a Python interpreter (>= 3.9)
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: no suitable Python interpreter found (Python >= 3.9 is required)" >&2
    exit 1
fi
echo "Using Python interpreter: $("$PYTHON_BIN" --version 2>&1) ($PYTHON_BIN)"

# 1) Initialize a Python venv in the temporary directory
"$PYTHON_BIN" -m venv "$BUILD_DIR/venv"
source "$BUILD_DIR/venv/bin/activate"

# 2) Install dependencies and PyInstaller
pip install --upgrade pip
pip install -r "$THIS_SCRIPT_DIR"/requirements.txt
pip install pyinstaller

# 3) Pack the boot script as a single file into the temporary directory
pyinstaller --onefile \
    --name chitu-boot \
    --path "$THIS_SCRIPT_DIR"/.. \
    --distpath "$BUILD_DIR/dist" \
    --workpath "$BUILD_DIR/work" \
    --specpath "$BUILD_DIR" \
    "$THIS_SCRIPT_DIR"/../chitu/boot/main.py

PACKED_BINARY="$BUILD_DIR/dist/chitu-boot"

# 4) Deactivate the venv
deactivate

#################################################################
# Put files into AppDir

APPDIR="$BUILD_DIR"/AppDir

mkdir -p "$APPDIR"/usr/bin
mkdir -p "$APPDIR"/usr/share/chitu

# 1) The packed binary that will run when the AppImage is executed
mv "$PACKED_BINARY" "$APPDIR"/AppRun
chmod +x "$APPDIR"/AppRun

# 2) The data file we want to bundle and print
cp -r "$THIS_SCRIPT_DIR"/../chitu/config "$APPDIR"/usr/share/chitu/config
if [ "$IMAGE_IS_APPTAINER" -eq 1 ]; then
    cp "$IMAGE_NAME" "$APPDIR"/usr/share/chitu/image.sif
fi
if [ "$IMAGE_IS_DOCKER" -eq 1 ]; then
    if [ "$USE_ONLINE_DOCKER" -eq 0 ]; then
        docker save "$IMAGE_NAME" -o "$APPDIR"/usr/share/chitu/image.docker
    fi
    echo "$IMAGE_NAME" > "$APPDIR"/usr/share/chitu/image_name.txt
fi

# 3) Minimal desktop file (required by AppImage spec)
cat > "$APPDIR"/chitu.desktop << 'EOF'
[Desktop Entry]
Name=chitu
Exec=chitu
Icon=chitu
Type=Application
Categories=Utility;
Terminal=true
EOF

# 4) Placeholder for icon
touch "$APPDIR"/chitu.png

#################################################################
# Detect CPU architecture

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    aarch64|arm64)
        ARCH="aarch64"
        ;;
    *)
        echo "Error: unsupported architecture '$ARCH'" >&2
        exit 1
        ;;
esac

#################################################################
# Download appimagetool

TOOL_DIR="$HOME/.chitu"
APPIMAGETOOL="$TOOL_DIR/appimagetool-$ARCH.AppImage"
RUNTIME_FILE="$TOOL_DIR/runtime-$ARCH"

mkdir -p "$TOOL_DIR"

if [ ! -x "$APPIMAGETOOL" ]; then
    wget -O "$APPIMAGETOOL" "https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-$ARCH.AppImage"
    chmod +x "$APPIMAGETOOL"
fi
if [ ! -f "$RUNTIME_FILE" ]; then
    wget -O "$RUNTIME_FILE" "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-$ARCH"
fi

#################################################################
# Build AppImage

rm "$OUTPUT_FILE" 2>/dev/null || true

# NOTE: We set `--runtime-file` even if it is optional, because if it is omitted,
# it will be downloaded every time this command is executed.
ARCH="$ARCH" "$APPIMAGETOOL" --runtime-file "$RUNTIME_FILE" "$APPDIR" "$OUTPUT_FILE"
