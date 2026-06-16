try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("selfix")
except PackageNotFoundError:
    __version__ = "unknown (no version information available)"

from pathlib import Path

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent
