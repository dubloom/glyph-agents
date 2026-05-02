"""Core types and registry for Glyph artifacts."""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any
from typing import Literal
from typing import TypeVar


ArtifactCapability = Literal["read_fs", "write_fs", "execute", "network"]
ArtifactHandler = Callable[["ArtifactContext"], Any | Awaitable[Any]]
F = TypeVar("F", bound=ArtifactHandler)


@dataclass(frozen=True)
class ArtifactContext:
    """Runtime context passed to artifact handlers."""

    workflow_path: Path | None = None
    workflow_dir: Path | None = None
    step_id: str | None = None
    step_input: Any = None
    markdown_context: dict[str, Any] | None = None
    args: dict[str, Any] | None = None

    def arg(self, name: str, default: Any = None) -> Any:
        """Return an artifact argument by name."""

        return (self.args or {}).get(name, default)


@dataclass(frozen=True)
class Artifact:
    """Registered executable artifact."""

    name: str
    handler: ArtifactHandler
    description: str | None = None
    capabilities: frozenset[ArtifactCapability] = frozenset()

    async def run(self, context: ArtifactContext) -> Any:
        """Execute the artifact handler."""

        result = self.handler(context)
        if inspect.isawaitable(result):
            return await result
        return result


_REGISTRY: dict[str, Artifact] = {}


def register_artifact(
    name: str,
    *,
    description: str | None = None,
    capabilities: set[ArtifactCapability] | frozenset[ArtifactCapability] | None = None,
) -> Callable[[F], F]:
    """Register ``func`` as a named artifact."""

    normalized_name = _normalize_artifact_name(name)

    def _decorate(func: F) -> F:
        _REGISTRY[normalized_name] = Artifact(
            name=normalized_name,
            handler=func,
            description=description,
            capabilities=frozenset(capabilities or ()),
        )
        return func

    return _decorate


def get_artifact(name: str) -> Artifact:
    """Return a registered artifact by name."""

    normalized_name = _normalize_artifact_name(name)
    try:
        return _REGISTRY[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(f"Unknown Glyph artifact {name!r}. Available artifacts: {available}.") from error


def list_artifacts() -> tuple[Artifact, ...]:
    """Return all registered artifacts sorted by name."""

    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def _normalize_artifact_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Artifact name must be a non-empty string.")
    normalized = name.strip()
    if any(part == "" for part in normalized.split(".")):
        raise ValueError(f"Invalid artifact name {name!r}.")
    return normalized


__all__ = [
    "Artifact",
    "ArtifactCapability",
    "ArtifactContext",
    "get_artifact",
    "list_artifacts",
    "register_artifact",
]
