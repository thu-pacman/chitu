# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import os
import sys

ZH_FILE = os.path.join("docs", "zh", "SUPPORTED_MODELS.md")
EN_FILE = os.path.join("docs", "en", "SUPPORTED_MODELS.md")


def _collect_model_names() -> list[str]:
    config_dir = os.path.join("chitu", "config", "models")
    names: list[str] = []
    for model_file in glob.iglob(os.path.join(config_dir, "*.yaml")):
        base = os.path.basename(model_file)
        if base.endswith(".yaml"):
            names.append(base[:-5])
    names.sort()
    return names


def _render_full_doc(names: list[str], lang: str) -> str:
    if lang == "zh":
        title = "# 支持的模型\n\n"
        intro = (
            "本页面由脚本自动生成。数据源: `chitu/config/models/*.yaml`。"
            "更新命令: `python3 script/generate_supported_models_docs.py`。\n\n"
        )
    else:
        title = "# Supported Models\n\n"
        intro = (
            "This page is auto-generated. Data source: `chitu/config/models/*.yaml`. "
            "To update, run: `python3 script/generate_supported_models_docs.py`.\n\n"
        )

    lines: list[str] = []
    for name in names:
        if lang == "zh":
            lines.append(f"- {name}")
            lines.append(f"  用法: 启动赤兔时追加 `models={name}` 启动参数")
        else:
            lines.append(f"- {name}")
            lines.append(
                f"  Usage: Append `models={name}` command line argument when starting Chitu"
            )
    lines.append("")

    return title + intro + "\n".join(lines)


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check drift; exit non-zero if update is needed",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print supported models to stdout (English) and exit",
    )
    args = parser.parse_args()

    names = _collect_model_names()

    if args.print:
        print("Supported models:")
        for name in names:
            print(f"- {name}")
            print(
                f"  Usage: Append `models={name}` command line argument when starting Chitu"
            )
        return 0

    zh_new = _render_full_doc(names, "zh")
    en_new = _render_full_doc(names, "en")

    zh_old = _read(ZH_FILE)
    en_old = _read(EN_FILE)

    changed = (zh_old != zh_new) or (en_old != en_new)

    if args.check:
        if changed:
            print(
                "Supported models docs are outdated. Please run:\n  python3 script/generate_supported_models_docs.py",
                file=sys.stderr,
            )
            return 2
        print("Supported models docs are up-to-date.")
        return 0

    if changed:
        _write(ZH_FILE, zh_new)
        _write(EN_FILE, en_new)
        print("Updated supported models docs.")
    else:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
