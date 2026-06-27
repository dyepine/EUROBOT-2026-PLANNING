"""Compatibility CLI wrapper for `poc.cli.main`."""

from poc.cli.main import *  # noqa: F401,F403
from poc.cli.main import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
