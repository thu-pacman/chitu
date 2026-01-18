#!/bin/bash

COLORS=(
    $'\033[0;31m'  # Red (ID 0)
    $'\033[0;32m'  # Green (ID 1)
    $'\033[0;33m'  # Yellow (ID 2)
    $'\033[0;34m'  # Blue (ID 3)
    $'\033[0;35m'  # Magenta (ID 4)
    $'\033[0;36m'  # Cyan (ID 5)
    $'\033[1;31m'  # Bright Red/Bold Red (ID 6)
    $'\033[1;32m'  # Bright Green/Bold Green (ID 7)
    $'\033[1;33m'  # Bright Yellow/Bold Yellow (ID 8)
    $'\033[1;34m'  # Bright Blue/Bold Blue (ID 9)
    $'\033[1;35m'  # Bright Magenta/Bold Magenta (ID 10)
    $'\033[1;36m'  # Bright Cyan/Bold Cyan (ID 11)
    $'\033[1;37m'  # Bright White/Bold White (ID 12)
)
RESET=$'\033[0m'

COLOR_ID=${LOCAL_RANK}
if [[ ! "$LOCAL_RANK" =~ ^[0-9]+$ ]] || [ "$LOCAL_RANK" -ge "${#COLORS[@]}" ]; then
    COLOR_ID=0
fi

# NOTE: There is an unknown buffering issue in `awk`, so use `sed` instead
python -u -m pytest --color=yes "$@" 2>&1 | sed "s/^/${COLORS[$COLOR_ID]}[LOCAL_RANK ${LOCAL_RANK}]${RESET} /"
exit ${PIPESTATUS[0]}  # Preserve the original command's exit code
