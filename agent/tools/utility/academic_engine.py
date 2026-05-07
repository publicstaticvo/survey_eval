from .openalex import get_openalex_client
from .s2 import get_semantic_scholar_client
from .tool_config import ToolConfig


def get_academic_engine(config: ToolConfig | None = None):
    config = config or ToolConfig()
    engine_name = (config.default_academic_search_engine or "openalex").strip().lower()
    if engine_name == "openalex":
        return get_openalex_client(config)
    if engine_name in {"semantic_scholar", "semanticscholar", "s2"}:
        return get_semantic_scholar_client(config)
    raise ValueError(f"Unsupported academic search engine: {config.default_academic_search_engine}")
