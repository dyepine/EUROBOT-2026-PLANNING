from __future__ import annotations

from typing import Any, Iterable


def _summary_from_result(result: Any) -> dict[str, float]:
    if hasattr(result, "summary"):
        return dict(result.summary)
    return dict(result["summary"])


def summarize_batch(results: Iterable[Any]) -> dict[str, float]:
    summaries = [_summary_from_result(result) for result in results]
    if not summaries:
        return {}

    numeric_keys = {
        key
        for summary in summaries
        for key, value in summary.items()
        if isinstance(value, (int, float))
    }
    return {
        key: round(sum(float(summary.get(key, 0.0)) for summary in summaries) / len(summaries), 4)
        for key in sorted(numeric_keys)
    }
