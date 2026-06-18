# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Generate CLI documentation from `chitu/config/serve_config.yaml`.

This script parses the Hydra config file `chitu/config/serve_config.yaml`, which
describes Chitu's CLI arguments, and generates a human-readable English document
at `docs/en/CLI.md`.

Only items whose immediately-preceding comment block contains lines starting with
`#:` are included in the document. Those `#:` lines (with the marker stripped) are
used as the description of the item. All other items (and items without a `#:`
comment block directly above them) are ignored entirely, and neither their value
nor any description is emitted.

Nested items are rendered with multi-level Markdown titles, and each documented
item is rendered as:

```markdown
**Argument `<argument-name>`**:

<description>

*Default: `<default-value>`.*
```

Usage:
    python script/generate_cli_docs.py

This script reads `chitu/config/serve_config.yaml` directly as text with a small
comment-aware line parser, so it has no third-party dependencies.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Marker that distinguishes a documentation comment from an ordinary one.
DOC_COMMENT_MARKER = "#:"

# Resolve paths relative to the repository root (the parent of `script/`).
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "chitu" / "config" / "serve_config.yaml"
OUTPUT_PATH = REPO_ROOT / "docs" / "en" / "CLI.md"

# Matches a YAML mapping key at the start of a line (after indentation), e.g.
# `  max_batch_size: 8` or `boot:`. It also matches a mapping key nested inside a
# list item, e.g. `  - models: ???`; in that case the key's effective indentation
# is the column of the key itself (after the `- ` marker), so list-item keys nest
# under the list's owning key. The trailing `value` group captures the inline
# value (if any), which we use as the item's default.
KEY_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dash>-\s+)?(?P<key>[A-Za-z_][\w-]*)\s*:(?:\s+(?P<value>.*))?$"
)

# Matches an inline (end-of-line) comment in a value, e.g. ` # some note`. We only
# strip comments that are clearly separated by whitespace, to avoid mangling `#`
# characters that are part of a quoted string value.
INLINE_COMMENT_RE = re.compile(r"\s+#.*$")


@dataclass
class DocItem:
    """A single documented config item discovered while parsing the YAML."""

    key: str
    indent: int
    description: str
    default: Optional[str] = None
    children: List["DocItem"] = field(default_factory=list)


def _is_comment_line(line: str) -> bool:
    return line.lstrip().startswith("#")


def _doc_lines_from_block(block: List[str]) -> Optional[List[str]]:
    """Extract `#:` description lines from a contiguous comment block.

    `block` is the list of raw comment lines directly above a key. We only treat
    the block as documentation if it contains at least one `#:` line. Returns the
    cleaned description lines, or `None` if the block has no `#:` lines.
    """
    doc_lines: List[str] = []
    for raw in block:
        stripped = raw.strip()
        if not stripped.startswith(DOC_COMMENT_MARKER):
            continue
        content = stripped[len(DOC_COMMENT_MARKER) :]
        # Drop a single leading space after the marker, if present.
        if content.startswith(" "):
            content = content[1:]
        doc_lines.append(content)

    if not doc_lines:
        return None
    return doc_lines


def parse_doc_items(text: str) -> List[DocItem]:
    """Parse the YAML text and return a tree of documented items.

    The parser walks line by line, tracking the contiguous comment block that
    immediately precedes each mapping key. If that block contains `#:` lines, the
    key becomes a documented item; otherwise it is ignored. Nesting is recovered
    from indentation, and documented children attach to their nearest documented
    ancestor.
    """
    lines = text.splitlines()

    # Roots of the documented-item forest.
    roots: List[DocItem] = []
    # Stack of DocItems for the current documented-ancestor chain.
    stack: List[DocItem] = []

    # The contiguous comment block immediately above the current line. It is reset
    # whenever a blank line or a non-comment, non-key line breaks the contiguity.
    pending_comments: List[str] = []

    for line in lines:
        if line.strip() == "":
            # Blank line breaks comment contiguity.
            pending_comments = []
            continue

        if _is_comment_line(line):
            pending_comments.append(line)
            continue

        match = KEY_RE.match(line)
        if not match:
            # Some other YAML content (e.g. list item, scalar continuation). It
            # breaks comment contiguity but is not itself documented.
            pending_comments = []
            continue

        # For a key nested inside a list item (`- key:`), the effective
        # indentation is the column of the key, i.e. after the `- ` marker. This
        # makes such keys nest under the list's owning key.
        indent = len(match.group("indent")) + len(match.group("dash") or "")
        key = match.group("key")

        # Capture the inline default value, stripping any end-of-line comment.
        raw_value = match.group("value")
        default: Optional[str] = None
        if raw_value is not None:
            cleaned = INLINE_COMMENT_RE.sub("", raw_value).strip()
            if cleaned:
                default = cleaned

        doc_lines = _doc_lines_from_block(pending_comments)
        pending_comments = []

        if doc_lines is None:
            # Undocumented key: ignored entirely. Documented descendants (if any)
            # will attach to the nearest documented ancestor instead.
            continue

        item = DocItem(
            key=key,
            indent=indent,
            description="\n".join(doc_lines),
            default=default,
        )

        # Pop ancestors that are not shallower than the current item.
        while stack and stack[-1].indent >= indent:
            stack.pop()

        if stack:
            stack[-1].children.append(item)
        else:
            roots.append(item)

        stack.append(item)

    return roots


def _render_items(
    items: List[DocItem], lines: List[str], heading_level: int, prefix: str = ""
) -> None:
    """Render documented items (and their documented children) into `lines`.

    `prefix` is the dotted path of the parent item, so that every item is
    rendered with its fully-qualified name (e.g. `boot.n_nodes`, not `n_nodes`).
    """
    for item in items:
        full_name = f"{prefix}.{item.key}" if prefix else item.key
        heading_prefix = "#" * min(heading_level, 6)
        if item.children:
            # A documented item that also has documented children becomes a
            # plain section heading (without the `Argument` label, which is
            # reserved for leaf arguments), with its description directly below.
            lines.append(f"{heading_prefix} `{full_name}`")
            lines.append("")
            lines.append(item.description)
            lines.append("")
            _render_items(item.children, lines, heading_level + 1, full_name)
        else:
            # A leaf argument is rendered as a heading too, but keeps the
            # `Argument` label, and shows its default value (if any) below.
            lines.append(f"{heading_prefix} Argument `{full_name}`")
            lines.append("")
            lines.append(item.description)
            lines.append("")
            if item.default is not None:
                lines.append(f"*Default: `{item.default}`.*")
                lines.append("")


def render_docs(input_path: Path) -> str:
    """Parse `input_path` and return the generated Markdown as a string."""
    text = input_path.read_text(encoding="utf-8")

    items = parse_doc_items(text)

    lines: List[str] = [
        "# Chitu CLI Arguments",
        "",
        (
            "This document is automatically generated from "
            "`chitu/config/serve_config.yaml` by `script/generate_cli_docs.py`. "
            "Do not edit it by hand."
        ),
        "",
    ]

    _render_items(items, lines, heading_level=2)

    # Collapse any trailing blank lines into a single newline at EOF.
    return "\n".join(lines).rstrip() + "\n"


def generate_docs(input_path: Path, output_path: Path) -> None:
    """Parse `input_path` and write the generated Markdown to `output_path`."""
    output_text = render_docs(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")

    print(f"Generated {output_path} from {input_path}")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the CLI documentation from "
            "`chitu/config/serve_config.yaml`. By default the doc is updated in "
            "place if it is out of date."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check drift; exit non-zero if update is needed",
    )
    args = parser.parse_args()

    new_doc = render_docs(INPUT_PATH)
    old_doc = _read(OUTPUT_PATH)
    changed = old_doc != new_doc

    if args.check:
        if changed:
            print(
                "CLI docs are outdated. Please run:\n"
                "  python3 script/generate_cli_docs.py",
                file=sys.stderr,
            )
            return 2
        print("CLI docs are up-to-date.")
        return 0

    if changed:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(new_doc, encoding="utf-8")
        print(f"Updated CLI docs at {OUTPUT_PATH}.")
    else:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
