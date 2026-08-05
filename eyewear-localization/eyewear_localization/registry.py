"""Small model registry used to keep optional model dependencies replaceable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Factory = Callable[..., Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Factory
    fallback: Factory | None = None
    reliability: float = 1.0


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {}

    def register(
        self,
        name: str,
        factory: Factory,
        *,
        fallback: Factory | None = None,
        reliability: float = 1.0,
    ) -> None:
        if not name.strip():
            raise ValueError("model name must not be empty")
        if name in self._models:
            raise ValueError(f"model already registered: {name}")
        if not 0 <= reliability <= 1:
            raise ValueError("model reliability must be between 0 and 1")
        self._models[name] = ModelSpec(name, factory, fallback, reliability)

    def build(self, name: str, **kwargs: Any) -> Any:
        try:
            spec = self._models[name]
        except KeyError:
            raise KeyError(f"unknown model {name!r}; available: {', '.join(self.names())}") from None
        try:
            return spec.factory(**kwargs)
        except Exception:
            if spec.fallback is None:
                raise
            return spec.fallback(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._models)

    def spec(self, name: str) -> ModelSpec:
        return self._models[name]
