import json
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path.home() / ".glyph" / "glyphs.json"

class GlyphRegistryError(Exception):
    """Raised when the named glyph registry cannot satisfy a request."""

def _load_registry() -> dict[str, str]:
    path = REGISTRY_PATH
    if not path.exists():
        return {}

    try:
        raw_registry: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_registry, dict):
            raise TypeError("Glyphs registry should be a dict")
    except Exception as exc:
        raise GlyphRegistryError(f"Invalid glyphs registry at {path}: {exc}") from exc


    registry: dict[str, str] = {}
    for name, workflow_path in raw_registry.items():
        if not isinstance(name, str) or not isinstance(workflow_path, str):
            raise GlyphRegistryError(f"invalid glyph registry at {path}: names and paths must be strings")
        registry[name] = workflow_path
    return registry


def _save_registry(registry: dict[str, str]) -> None:
    path = REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_glyph(name: str, workflow_path: Path) -> Path:
    """Register ``name`` for ``workflow_path`` and return the stored path."""

    resolved_path = workflow_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise GlyphRegistryError(f"workflow file does not exist: {resolved_path}")

    if resolved_path.suffix.lower() != ".md":
        raise GlyphRegistryError(f"workflow file must be a markdown file: {resolved_path}")

    registry = _load_registry()
    if name in registry:
        raise GlyphRegistryError(f"glyph name is already taken: {name}")

    registry[name] = str(resolved_path)
    _save_registry(registry)
    return resolved_path


def list_registered_glyphs() -> list[tuple[str, str]]:
    """Return registered glyph names and workflow paths, sorted by name."""

    registry = _load_registry()
    return sorted(registry.items(), key=lambda item: item[0])


def resolve_glyph(name: str) -> Path:
    """Resolve a registered glyph name to its markdown workflow path."""

    registry = _load_registry()
    workflow_path = registry.get(name)
    if workflow_path is None:
        raise GlyphRegistryError(f"unknown glyph: {name}")

    return Path(workflow_path).expanduser()


def remove_glyph(name: str) -> None:
    """Remove a registered glyph name from the registry."""

    registry = _load_registry()
    if name not in registry:
        raise GlyphRegistryError(f"unknown glyph: {name}")

    del registry[name]
    _save_registry(registry)
