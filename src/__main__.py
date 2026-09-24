"""Entry point: uv run python -m src <command> [options]."""

import sys
import fire
from src.cli import Cli, CliError


def main() -> int:
    """Dispatch to Fire and convert expected errors into messages."""
    try:
        fire.Fire(Cli, name="python -m src")
    except CliError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
