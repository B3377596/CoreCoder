"""CoreCoder - Minimal AI coding agent inspired by Claude Code's architecture."""
__version__ = "0.3.0"
# Lazy imports so subpackages can be imported without
# requiring all runtime dependencies (openai, etc.) to be installed.
# Access Agent/LLM/etc. through the module as normal ?*they are loaded
# on first access.
def __getattr__(name: str):
    _imports = {
        "Agent": "corecoder.agent",
        "LLM": "corecoder.llm",
        "LiteLLM": "corecoder.llm",
        "Config": "corecoder.config",
        "ALL_TOOLS": "corecoder.tools",
    }

    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name])
        attr = getattr(mod, name)
        # Cache in globals so __getattr__ is only called once per name
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Agent", "LLM", "LiteLLM", "Config", "ALL_TOOLS", "__version__"]

