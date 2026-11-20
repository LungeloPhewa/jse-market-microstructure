"""Helpers for the book: show the exact source code (with docstrings) of the
functions that produce a result, and a few formatting utilities.

Showing the real source -- rather than a copy pasted into the text -- means
the code in the report can never drift away from the code that ran.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap


def show_source(obj, max_lines: int | None = None) -> None:
    """Print a function's or class's source as a fenced Markdown code block
    (use in a chunk with ``#| output: asis``)."""
    src = textwrap.dedent(inspect.getsource(obj))
    lines = src.rstrip().split("\n")
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + ["    ...  # (truncated; full source on GitHub)"]
    mod = inspect.getmodule(obj).__name__
    print(f"\n```python\n# {mod}.{obj.__qualname__}\n" + "\n".join(lines) + "\n```\n")


def module_summary(module_name: str) -> str:
    """Markdown summary of a module: its docstring's first paragraph and a
    table of its public functions with the first line of each docstring."""
    mod = importlib.import_module(module_name)
    doc = inspect.getdoc(mod) or ""
    first = doc.split("\n\n")[0].replace("\n", " ")
    rows = []
    tree = ast.parse(inspect.getsource(mod))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            d = (ast.get_docstring(node) or "").strip().split("\n")[0]
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            rows.append(f"| `{node.name}` | {kind} | {d.replace('|', '/')} |")
    out = f"**`{module_name}`** — {first}\n\n"
    if rows:
        out += "| Name | Kind | Purpose |\n|---|---|---|\n" + "\n".join(rows) + "\n"
    return out


def package_modules(package: str = "jsemicro") -> list[str]:
    """All module names in the package, in a stable order."""
    pkg = importlib.import_module(package)
    names = [m.name for m in pkgutil.walk_packages(pkg.__path__, package + ".") if not m.ispkg]
    return sorted(n for n in names if not n.endswith("__main__"))


def record(**values) -> None:
    """Store headline results in ``<data root>/out/summary.json`` so that the
    concluding chapter can quote numbers computed in earlier chapters (each
    chapter runs in its own kernel)."""
    import json

    import numpy as np

    from .config import PATHS

    path = PATHS.out_dir / "summary.json"
    data = json.loads(path.read_text()) if path.exists() else {}

    def plain(v):
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    data.update(plain(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def recall() -> dict:
    """Headline results stored by :func:`record`."""
    import json

    from .config import PATHS

    path = PATHS.out_dir / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}
