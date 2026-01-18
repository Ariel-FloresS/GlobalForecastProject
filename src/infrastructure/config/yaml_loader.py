from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config_loader_interface import ConfigLoaderInterface


class YamlConfigLoader(ConfigLoaderInterface):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = {}

    @property
    def path(self) -> Path:
        return self._path

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def load(self) -> dict[str, Any]:
        with self._path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
        if not isinstance(content, dict):
            raise ValueError(f"YAML root must be a mapping. Got: {type(content).__name__}")
        self._data = content
        return self._data

    def get(self, key: str, default: Any | None = None) -> Any:
        return self._data.get(key, default)

    def get_required(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"Missing required key: {key}")
        return self._data[key]

    def get_section(self, key: str) -> dict[str, Any]:
        value = self.get_required(key)
        if not isinstance(value, dict):
            raise ValueError(f"Section '{key}' must be a mapping.")
        return value

    def get_policy(self, section: str, name: str) -> dict[str, Any]:
        policies = self.get_section(section)
        if name not in policies:
            raise KeyError(f"Missing '{name}' in section '{section}'.")
        policy = policies[name]
        if not isinstance(policy, dict):
            raise ValueError(f"Policy '{name}' in '{section}' must be a mapping.")
        return policy

    def build_object(self, spec: Any, registry: Mapping[str, type] | None = None) -> Any:
        return build_object(spec, registry=registry)

    def build_object_list(
        self,
        specs: Iterable[Any],
        registry: Mapping[str, type] | None = None,
    ) -> list[Any]:
        return [build_object(spec, registry=registry) for spec in specs]


def build_object(spec: Any, registry: Mapping[str, type] | None = None) -> Any:
    if spec is None or isinstance(spec, (int, float, bool, str, list, tuple)):
        if isinstance(spec, str) and (registry or "." in spec):
            return _instantiate(spec, [], {}, registry)
        return spec
    if isinstance(spec, dict):
        class_path = spec.get("class") or spec.get("type")
        if not class_path:
            return spec
        params = spec.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ValueError("Object params must be a mapping.")
        args = spec.get("args", []) or []
        if not isinstance(args, list):
            raise ValueError("Object args must be a list.")
        return _instantiate(class_path, args, params, registry)
    return spec


def _instantiate(
    class_path: str,
    args: list[Any],
    params: Mapping[str, Any],
    registry: Mapping[str, type] | None,
) -> Any:
    cls = _resolve_class(class_path, registry)
    return cls(*args, **params)


def _resolve_class(class_path: str, registry: Mapping[str, type] | None) -> type:
    if registry and class_path in registry:
        return registry[class_path]
    if "." not in class_path:
        raise ValueError(
            "Class path must be fully qualified or present in registry: "
            f"'{class_path}'."
        )
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
