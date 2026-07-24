# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
import argparse
import glob
import os
import sys
import hydra

ZH_FILE = os.path.join("docs", "zh", "SUPPORTED_MODELS.md")
EN_FILE = os.path.join("docs", "en", "SUPPORTED_MODELS.md")


def _collect_models() -> list[tuple[str, Any]]:
    config_dir = os.path.join(os.getcwd(), "chitu", "config", "models")
    ret: list[tuple[str, Any]] = []
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        for model_file in glob.iglob(os.path.join(config_dir, "*.yaml")):
            base = os.path.basename(model_file)
            if base.endswith(".yaml"):
                filename = base[:-5]
                if filename == "none":
                    # This is a placeholder only for testing
                    continue
                cfg = hydra.compose(config_name=filename)
                ret.append((filename, cfg))
    return ret


def _sort_models(models: list[tuple[str, Any]], is_pro: bool) -> list[tuple[str, Any]]:
    return sorted(
        filter(lambda x: getattr(x[1], "is_pro", False) == is_pro, models),
        key=lambda x: x[0],
    )


def _render_full_doc(models: list[tuple[str, Any]], lang: str) -> str:
    if lang == "zh":
        title = "# 支持的模型\n\n"
        intro = (
            "> 本页面由脚本自动生成。数据源: `chitu/config/models/*.yaml`。"
            "更新命令: `python3 script/generate_supported_models_docs.py`。\n\n"
        )
    else:
        title = "# Supported Models\n\n"
        intro = (
            "> This page is auto-generated. Data source: `chitu/config/models/*.yaml`. "
            "To update, run: `python3 script/generate_supported_models_docs.py`.\n\n"
        )

    # Generate a table, example:
    #
    # | Name | Support tool calling (featuring constrained decoding) | Usage (append the argument below when starting Chitu) | How to obtain the model |
    # |------|------------------------------------------------------|-------------------------------------------------------|-------------------------|
    # | name | ✓                                                    | `models=file_name`                                    | http://...              |

    lines: list[str] = []
    if lang == "zh":
        lines.append("## 开源模型")
        lines.append("")
        lines.append(
            "| 名称 | 支持工具调用（内含约束解码） | 用法（启动赤兔时追加下列参数） | 获取方法 |"
        )
    else:
        lines.append("## Open-source models")
        lines.append("")
        lines.append(
            "| Name | Support tool calling (featuring constrained decoding) | Usage (append the argument below when starting Chitu) | How to obtain the model |"
        )
    lines.append("|---|---|---|---|")
    for filename, cfg in _sort_models(models, is_pro=False):
        if hasattr(cfg, "tool_parser"):
            support_tool_calling = "✓"
        else:
            support_tool_calling = ""
        lines.append(
            f"| {cfg.name} | {support_tool_calling} | `models={filename}` | {cfg.source} |"
        )
    lines.append("")
    lines.append("")
    open_source_models = "\n".join(lines)

    lines: list[str] = []
    if lang == "zh":
        lines.append("## 赤兔-pro 模型")
        lines.append("")
        lines.append(
            "以下模型随赤兔-pro提供，请联系 [solution@chitu.ai](solution@chitu.ai) 进行商务咨询。"
        )
        lines.append("")
        lines.append(
            "| 名称 | 支持工具调用（内含约束解码） | 用法（启动赤兔时追加下列参数） |"
        )
    else:
        lines.append("## Chitu-pro models")
        lines.append("")
        lines.append(
            "The following models are part of chitu-pro. Please concat [solution@chitu.ai](solution@chitu.ai) for business inquiries."
        )
        lines.append("")
        lines.append(
            "| Name | Support tool calling (featuring constraint decoding) | Usage (append the argument below when starting Chitu) |"
        )
    lines.append("|---|---|---|")
    for filename, cfg in _sort_models(models, is_pro=True):
        if hasattr(cfg, "tool_parser"):
            support_tool_calling = "✓"
        else:
            support_tool_calling = ""
        lines.append(f"| {cfg.name} | {support_tool_calling} | `models={filename}` |")
    lines.append("")
    lines.append("")
    chitu_pro_models = "\n".join(lines)

    return title + intro + open_source_models + chitu_pro_models


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

    models = _collect_models()

    if args.print:
        print("Supported models:")
        for filename, cfg in models:
            print(f"- {cfg.name}")
            print(
                f"  Usage: Append `models={filename}` command line argument when starting Chitu"
            )
        return 0

    zh_new = _render_full_doc(models, "zh")
    en_new = _render_full_doc(models, "en")

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
