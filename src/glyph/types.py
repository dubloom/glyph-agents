from dataclasses import dataclass
import enum
from typing import Optional


@dataclass
class ModelOptions:
    # not supported by claude_agent SDK
    temperature: Optional[float] = None

    reasoning: Optional[str] = None


class AllowedTools(enum.Enum):
    BASH = 0
    EDIT = 1
    GLOB = 2
    GREP = 3
    READ = 4
    WRITE = 5
