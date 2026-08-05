#!/usr/bin/env python3
"""Compatibility entry point: ``python infer.py IMAGE ...``."""

from eyewear_localization.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
