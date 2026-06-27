from __future__ import annotations

from pathlib import Path

from poc.map_config import (
    DEFAULT_SEMANTIC_MAP_CONFIG,
    FIELD_HEIGHT,
    FIELD_WIDTH,
    SemanticMap,
    SemanticMapConfig,
    load_semantic_map_config,
)


def build_semantic_map(config: SemanticMapConfig) -> SemanticMap:
    return config.build()


def build_semantic_map_from_yaml(path: str | Path) -> SemanticMap:
    return build_semantic_map(load_semantic_map_config(path))


def build_default_semantic_map() -> SemanticMap:
    return build_semantic_map(DEFAULT_SEMANTIC_MAP_CONFIG)
