"""Small explicit extension contract; importing it loads no optional extension."""
from dataclasses import dataclass
from typing import Callable
from ai_client import DeepSeekSettings

CORE_API_VERSION = 1


@dataclass(frozen=True)
class ExtensionContext:
    api_version: int
    get_settings: Callable[[bool], DeepSeekSettings]
    get_resume_text: Callable[[], str]
    get_transcript: Callable[[], str]
    post: Callable[[Callable[[], None]], None]
