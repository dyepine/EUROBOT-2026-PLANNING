"""Compatibility CLI wrapper for `poc.cli.rl_dataset`."""

from poc.cli.rl_dataset import *  # noqa: F401,F403
from poc.cli.rl_dataset import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
