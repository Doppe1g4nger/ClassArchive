"""A deliberately thin name -> class registry.

Hydra's ``_target_`` does the actual instantiation, so this registry exists only
for the three things ``_target_`` cannot do: enumerate what exists (for CLI
listing and ``pytest.mark.parametrize`` over every method), resolve a short name
to a class, and fail loudly on a typo with the valid options in the message.

Deliberately *not* a dependency-injection container. Building a second DI system
on top of Hydra is how config systems become unreadable.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a short string key to a class."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, type[T]] = {}

    def register(self, key: str | None = None) -> Callable[[type[T]], type[T]]:
        """Decorator. Defaults to the lower-cased class name when ``key`` is omitted."""

        def deco(cls: type[T]) -> type[T]:
            k = key if key is not None else cls.__name__.lower()
            if k in self._entries and self._entries[k] is not cls:
                raise ValueError(
                    f"{self.name} registry: duplicate key {k!r} "
                    f"(already bound to {self._entries[k].__qualname__})"
                )
            self._entries[k] = cls
            return cls

        return deco

    def get(self, key: str) -> type[T]:
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(
                f"unknown {self.name} {key!r}; available: {sorted(self._entries)}"
            ) from None

    def build(self, key: str, /, **kwargs: object) -> T:
        return self.get(key)(**kwargs)  # type: ignore[call-arg]

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, {len(self._entries)} entries)"


METHODS: Registry = Registry("methods")
ENCODERS: Registry = Registry("encoders")
AUG_OPS: Registry = Registry("aug_ops")
MODULATORS: Registry = Registry("modulators")

_DISCOVERED = False


def autodiscover() -> None:
    """Import every submodule of the pluggable packages so decorators fire.

    Idempotent, so tests and CLI entry points can both call it unconditionally.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    for pkg_name in ("iqssl.dsp", "iqssl.models", "iqssl.methods", "iqssl.augment"):
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        for mod in pkgutil.iter_modules(pkg.__path__):
            if not mod.name.startswith("_"):
                importlib.import_module(f"{pkg_name}.{mod.name}")
