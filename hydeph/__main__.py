"""CLI entry point when invoked via `python -m hydeph`."""

from __future__ import annotations

import sys
from hydeph.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
