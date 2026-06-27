from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None


_DEFAULT = object()


def require_torch(torch_module=_DEFAULT, *, purpose: str = "self-play PPO") -> None:
    if torch_module is _DEFAULT:
        torch_module = torch
    if torch_module is None:
        raise ModuleNotFoundError(
            f"PyTorch is required for {purpose}. Install project dependencies with torch."
        )
