from glyph import AgentOptions
from glyph import resolve_backend


def main() -> None:
    models = [
        "gpt-4.1-mini",
        "o4-mini",
        "claude-haiku-4-5",
    ]
    for model in models:
        options = AgentOptions(model=model)
        backend = resolve_backend(options)
        print(f"{model} -> {backend}")


if __name__ == "__main__":
    main()
