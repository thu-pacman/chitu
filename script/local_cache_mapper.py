# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import logging

logger = logging.getLogger(__name__)

LOCAL_DIR = "/data/local"
NFS_DIRS = ["/data/nfs", "/data/nfs2"]


def _find_nfs_source(path):
    for nfs_dir in NFS_DIRS:
        if path.startswith(nfs_dir + "/"):
            return nfs_dir
    return None


def _get_local_path(path):
    nfs_source = _find_nfs_source(path)
    if nfs_source is None:
        return None

    rel_path = os.path.relpath(path, nfs_source)
    return os.path.join(LOCAL_DIR, rel_path)


def replace_with_cached_path(nfs_dir_path):
    nfs_source = _find_nfs_source(nfs_dir_path)
    if nfs_source is None:
        print(
            f"Warning: {nfs_dir_path} is not under any NFS directory {NFS_DIRS}, returning original path"
        )
        return nfs_dir_path

    if not os.path.exists(nfs_dir_path):
        raise FileNotFoundError(f"Source path does not exist: {nfs_dir_path}")

    local_cache_path = _get_local_path(nfs_dir_path)

    if os.path.exists(local_cache_path):
        return local_cache_path

    return nfs_dir_path


def main():
    global LOCAL_DIR, NFS_DIRS, MAX_CACHE_SIZE_BYTES

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="NFS model path")

    args = parser.parse_args()
    cached_path = replace_with_cached_path(args.model_path)
    print(f"CACHED_PATH={cached_path}")


if __name__ == "__main__":
    main()
