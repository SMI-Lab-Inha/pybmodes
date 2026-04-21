"""pybmodes — Python finite-element library for wind turbine modal analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pybmodes")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
