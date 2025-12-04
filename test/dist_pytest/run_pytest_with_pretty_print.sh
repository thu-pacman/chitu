#!/bin/bash

python -u -m pytest --color=yes "$@" 2>&1 | awk -v r="$LOCAL_RANK" '{print "[LOCAL_RANK " r "] " $0}'
exit ${PIPESTATUS[0]}  # Preserve the original command's exit code
