"""Configuration and backend resolution."""

from dataclasses import dataclass
from typing import Literal


ProviderName = Literal["auto", "openai", "claude"]
BackendName = Literal["openai", "claude"]


@dataclass
class AgentOptions:
    """Options shared across vendors; ``model`` and optional ``provider`` pick the backend.

    - ``instructions`` maps to OpenAI ``Agent.instructions`` and Claude ``system_prompt``.
    - ``name`` is used for OpenAI ``Agent.name`` only (Claude CLI ignores it).
    """

    model: str
    instructions: str | None = None
    name: str = "Assistant"
    provider: ProviderName = "auto"

    def __post_init__(self) -> None:
        self.model = self.model.strip()
        if not self.model:
            raise ValueError("model must be a non-empty string.")
        if self.provider not in ("auto", "openai", "claude"):
            raise ValueError("provider must be one of: 'auto', 'openai', 'claude'.")


def resolve_backend(options: AgentOptions) -> BackendName:
    """Return the concrete backend for ``options``.

    With ``provider="auto"``, uses ``model`` (case-insensitive):

    - **Claude** if the string contains ``"claude"`` or ``"anthropic"``.
    - **OpenAI** if it starts with ``gpt-``, ``o1``, ``o3``, ``o4``, or ``chatgpt``.

    Raises ``ValueError`` if auto-detection cannot decide; set ``provider`` explicitly.
    """
    if options.provider == "openai":
        return "openai"
    if options.provider == "claude":
        return "claude"

    m = options.model.lower()

    is_claude = "claude" in m or "anthropic" in m
    is_openai = (
        m.startswith("gpt-")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
        or m.startswith("chatgpt")
    )

    if is_claude and is_openai:
        raise ValueError(
            f"Ambiguous model {options.model!r} for provider 'auto'; set provider to 'openai' or 'claude' explicitly."
        )
    if is_claude:
        return "claude"
    if is_openai:
        return "openai"

    raise ValueError(
        f"Cannot infer backend from model {options.model!r}. "
        "Use provider='openai' or provider='claude', or a recognized model id."
    )
