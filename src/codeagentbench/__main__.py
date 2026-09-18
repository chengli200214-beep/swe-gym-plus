"""Allow ``python -m codeagentbench``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
