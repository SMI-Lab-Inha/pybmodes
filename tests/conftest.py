"""Shared pytest fixtures for pyBModes.

This module no longer references any bundled reference data — all
integration testing uses analytical / textbook benchmarks defined inline
in :mod:`tests.fem.test_cantilever` and
:mod:`tests.fem.test_uniform_tower_analytical`.  Parser tests that need
sample input files build them on the fly via ``tmp_path``.
"""

from __future__ import annotations
