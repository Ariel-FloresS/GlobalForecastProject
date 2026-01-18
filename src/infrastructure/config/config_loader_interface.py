from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConfigLoaderInterface(ABC):
    @abstractmethod
    def load(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str, default: Any | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_required(self, key: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_section(self, key: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_policy(self, section: str, name: str) -> dict[str, Any]:
        raise NotImplementedError
