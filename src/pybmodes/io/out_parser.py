"""Parser for modal analysis output files (.out).

Output format:
  - Header / title
  - "rotating blade frequencies & mode shapes"   OR
    "tower frequencies & mode shapes"
  - For each mode:
      -------- Mode No.  N  (freq = X.XXXE+XX Hz)
      <column-header line>
      <blank line>
      <data rows: span_loc + 5 columns>
      <blank line>

Column order:
  Blade : span_loc  flap_disp  flap_slope  lag_disp  lag_slope  twist
  Tower : span_loc  ss_disp    ss_slope    fa_disp   fa_slope   twist
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModeShape:
    """Mode shape data for a single mode."""
    mode_number: int
    frequency: float             # Hz

    span_loc:   np.ndarray       # normalised span (0–1)
    col1:       np.ndarray       # flap_disp  (blade) or ss_disp  (tower)
    col2:       np.ndarray       # flap_slope (blade) or ss_slope (tower)
    col3:       np.ndarray       # lag_disp   (blade) or fa_disp  (tower)
    col4:       np.ndarray       # lag_slope  (blade) or fa_slope (tower)
    twist:      np.ndarray

    col_names: list[str] = field(default_factory=list)

    # Convenience properties — blade
    @property
    def flap_disp(self) -> np.ndarray:
        return self.col1

    @property
    def flap_slope(self) -> np.ndarray:
        return self.col2

    @property
    def lag_disp(self) -> np.ndarray:
        return self.col3

    @property
    def lag_slope(self) -> np.ndarray:
        return self.col4

    # Convenience properties — tower (fore-aft / side-side)
    @property
    def ss_disp(self) -> np.ndarray:
        return self.col1

    @property
    def ss_slope(self) -> np.ndarray:
        return self.col2

    @property
    def fa_disp(self) -> np.ndarray:
        return self.col3

    @property
    def fa_slope(self) -> np.ndarray:
        return self.col4


@dataclass
class BModeOutput:
    """All mode shapes parsed from a .out file."""
    title: str
    beam_type: str               # 'blade' or 'tower'
    modes: list[ModeShape]
    source_file: Optional[pathlib.Path] = None

    def __len__(self) -> int:
        return len(self.modes)

    def __getitem__(self, index: int) -> ModeShape:
        return self.modes[index]

    def frequencies(self) -> np.ndarray:
        """Return array of natural frequencies (Hz) in mode order."""
        return np.array([m.frequency for m in self.modes])


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_MODE_HEADER = re.compile(
    r'-+\s*Mode\s+No\.\s*(\d+)\s*\(freq\s*=\s*([0-9Ee.+\-]+)\s*Hz\)',
    re.IGNORECASE,
)
_RE_BEAM_TYPE_BLADE = re.compile(r'rotating\s+blade\s+frequencies', re.IGNORECASE)
_RE_BEAM_TYPE_TOWER = re.compile(r'tower\s+frequencies', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_out(path: str | pathlib.Path) -> BModeOutput:
    """Parse a .out file and return a :class:`BModeOutput`."""
    path = pathlib.Path(path)
    lines = path.read_text(encoding='latin-1').splitlines()
    return _parse(lines, source_file=path)


def _parse(lines: list[str], source_file: pathlib.Path | None = None) -> BModeOutput:
    title = ''
    beam_type = 'blade'
    modes: list[ModeShape] = []

    i = 0
    n = len(lines)

    # --- scan header for title and beam type ---
    while i < n:
        ln = lines[i]
        if _RE_BEAM_TYPE_BLADE.search(ln):
            beam_type = 'blade'
        elif _RE_BEAM_TYPE_TOWER.search(ln):
            beam_type = 'tower'

        m = _RE_MODE_HEADER.search(ln)
        if m:
            break

        # The second non-empty, non-separator line is the title
        stripped = ln.strip()
        if stripped and not stripped.startswith('=') and not title:
            # Generated-by line comes first; the second such line is title
            title = stripped

        i += 1

    # --- parse mode blocks ---
    while i < n:
        ln = lines[i]
        m = _RE_MODE_HEADER.search(ln)
        if not m:
            i += 1
            continue

        mode_no = int(m.group(1))
        freq    = float(m.group(2))

        # consume column-header line (skip blank lines before it)
        i += 1
        while i < n and not lines[i].strip():
            i += 1
        col_names = lines[i].split() if i < n else []
        i += 1

        # collect data rows
        rows: list[list[float]] = []
        while i < n:
            row_line = lines[i].strip()
            i += 1
            if not row_line:
                continue
            # stop at next mode header
            if _RE_MODE_HEADER.search(row_line) or row_line.startswith('='):
                i -= 1   # put the header line back
                break
            tokens = row_line.split()
            if len(tokens) < 6:
                continue
            try:
                rows.append([float(t) for t in tokens[:6]])
            except ValueError:
                continue

        if rows:
            arr = np.array(rows, dtype=float)
            modes.append(ModeShape(
                mode_number=mode_no,
                frequency=freq,
                span_loc=arr[:, 0],
                col1=arr[:, 1],
                col2=arr[:, 2],
                col3=arr[:, 3],
                col4=arr[:, 4],
                twist=arr[:, 5],
                col_names=col_names,
            ))

    return BModeOutput(
        title=title,
        beam_type=beam_type,
        modes=modes,
        source_file=source_file,
    )
