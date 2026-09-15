#!/usr/bin/env python3
"""Run PersonalOS without installing it.

    python main.py ask "what is in my Downloads folder?"

Installing the package (`pip install -e .`) gives you the `agent` command,
which is the same entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running straight from a checkout, before `pip install -e .`.
SOURCE = Path(__file__).parent / "src"
if SOURCE.is_dir() and str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from personalos.interface.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
