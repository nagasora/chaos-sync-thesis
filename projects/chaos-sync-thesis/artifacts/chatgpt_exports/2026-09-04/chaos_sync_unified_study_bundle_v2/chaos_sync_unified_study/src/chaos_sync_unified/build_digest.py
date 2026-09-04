from pathlib import Path

from .digest import build_digest


if __name__ == "__main__":
    build_digest(Path.cwd())
