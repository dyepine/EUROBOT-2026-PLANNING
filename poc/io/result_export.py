from __future__ import annotations

import json
from pathlib import Path

from poc.simulation.history import plain_value
from poc.simulation.simulator import MatchResult


def save_result(result: MatchResult, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plain_value(result), indent=2), encoding="utf-8")
    return output
