from pathlib import Path

from .report import build_report


if __name__ == "__main__":
    build_report(Path.cwd())
