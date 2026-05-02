"""Glyph artifacts."""

from .core import Artifact
from .core import ArtifactCapability
from .core import ArtifactContext
from .core import get_artifact
from .core import list_artifacts
from .core import register_artifact as artifact
from .core import register_artifact

# Import built-ins for registration side effects.
from . import change_request as _change_request
from . import ci as _ci
from . import command as _command
from . import misc as _misc
from . import project as _project
from . import repo as _repo
from . import worktree as _worktree


__all__ = [
    "Artifact",
    "ArtifactCapability",
    "ArtifactContext",
    "artifact",
    "get_artifact",
    "list_artifacts",
    "register_artifact",
]
