"""
DEPRECATED wrapper.

This file lives under `_tmp/` and is kept only for backward compatibility with
older commands. The maintained implementation is:
  tools/porto/porto_od_diversity_scan.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "tools" / "porto" / "porto_od_diversity_scan.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
