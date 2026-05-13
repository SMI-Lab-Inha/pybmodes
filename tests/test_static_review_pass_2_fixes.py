"""Regression tests for the pass-2 static-review findings.

Three classes of bug caught after the v1.0.0 release:

1. **MoorDyn parser — silent malformed-row drops.** ``from_moordyn``
   used to ``continue`` on rows that failed the ``len(parts) < N``
   check or the per-column ``float`` / ``int`` parse, which turned a
   typo into an incomplete mooring model with no diagnostic. The
   parser now raises ``ValueError`` on rows that look like data but
   fail strict parsing; rows that look like column-name / units
   headers are still skipped (via ``_looks_like_header_row``) so
   MoorDyn-deck variants with 0 / 1 / 2 header rows all parse cleanly.

2. **MoorDyn section-splitter — hardcoded 2-row header assumption.**
   ``_split_sections`` used to skip exactly 2 rows after every section
   divider. A valid-ish deck with 1 header row (no units line) had
   its first data row eaten. Now the splitter inspects each
   post-divider row and only skips it if it matches
   ``_looks_like_header_row``.

3. **SubDyn adapter — bare ``StopIteration`` on missing reaction /
   interface joint IDs.** A deck whose ``BASE REACTION`` / ``INTERFACE
   JOINTS`` block referenced a joint ID absent from the
   ``STRUCTURE JOINTS`` table raised an uninformative
   ``StopIteration``; now it raises a clear ``ValueError`` naming the
   missing ID and the source file.
"""

from __future__ import annotations

import pathlib

import pytest

# ---------------------------------------------------------------------------
# Shared MoorDyn synthetic-deck helpers
# ---------------------------------------------------------------------------

# Minimum valid v2 deck — 1 line type, 2 points, 1 line. We assemble
# the file as four named sections so each test can vary the section it
# wants to break while keeping the others intact.

_HEADER = "----------- MoorDyn v2 Input File -------------------\n"

_LINE_TYPES_2HEADER = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   0.10    50.0            1.0e9   0.0
"""

_LINE_TYPES_1HEADER = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
chain   0.10    50.0            1.0e9   0.0
"""

_POINTS_2HEADER = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1       Fixed           100.0   0.0     -50.0
2       Vessel          5.0     0.0     -10.0
"""

_LINES_2HEADER = """\
---------------------- LINES ---------------------------------------------
ID      LineType        AttachA AttachB UnstrLen        NumSegs Outputs
(-)     (-)             (-)     (-)     (m)             (-)     (-)
1       chain           1       2       102.0           20      -
"""

_OPTIONS = """\
---------------------- OPTIONS -------------------------------------------
200.0   WtrDpth
1025.0  WtrDens
"""


def _write_moordyn(path: pathlib.Path, *sections: str) -> pathlib.Path:
    """Concatenate a header + arbitrary section blocks into a
    syntactically valid MoorDyn file at ``path``."""
    path.write_text(_HEADER + "".join(sections), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Group #1 — strict parsing on malformed data rows
# ---------------------------------------------------------------------------

class TestMoorDynStrictParsing:
    """Malformed-but-data-looking rows raise rather than silently
    skipping. Pre-1.0 review pass 2 surfaced that the previous
    ``continue`` branches converted typos into incomplete mooring
    models.
    """

    def test_line_types_too_few_columns_raises(self, tmp_path: pathlib.Path) -> None:
        bad_section = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   0.10    50.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", bad_section, _POINTS_2HEADER, _LINES_2HEADER,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(ValueError, match="Malformed LINE TYPES"):
            MooringSystem.from_moordyn(path)

    def test_line_types_non_numeric_value_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        bad_section = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   not_a_number    50.0    1.0e9   0.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", bad_section, _POINTS_2HEADER, _LINES_2HEADER,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(
            ValueError,
            match="Malformed LINE TYPES row.*chain.*Diam / MassPerLength / EA",
        ):
            MooringSystem.from_moordyn(path)

    def test_points_too_few_columns_raises(self, tmp_path: pathlib.Path) -> None:
        bad_section = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1       Fixed           100.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", _LINE_TYPES_2HEADER, bad_section, _LINES_2HEADER,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(ValueError, match="Malformed POINTS"):
            MooringSystem.from_moordyn(path)

    def test_points_non_integer_id_raises(self, tmp_path: pathlib.Path) -> None:
        bad_section = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1.5     Fixed           100.0   0.0     -50.0
2       Vessel          5.0     0.0     -10.0
"""
        # The first POINTS row's first column is "1.5" which DOES match
        # ``_looks_like_number`` (=> it's a data row, not a header).
        # The strict ``int(parts[0])`` raises ValueError on the
        # non-integer, so the parser converts it to the friendly
        # diagnostic.
        path = _write_moordyn(
            tmp_path / "bad.dat", _LINE_TYPES_2HEADER, bad_section, _LINES_2HEADER,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(
            ValueError, match="Malformed POINTS.*expected integer ID",
        ):
            MooringSystem.from_moordyn(path)

    def test_lines_too_few_columns_raises(self, tmp_path: pathlib.Path) -> None:
        bad_section = """\
---------------------- LINES ---------------------------------------------
ID      LineType        AttachA AttachB UnstrLen        NumSegs Outputs
(-)     (-)             (-)     (-)     (m)             (-)     (-)
1       chain
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", _LINE_TYPES_2HEADER, _POINTS_2HEADER, bad_section,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(ValueError, match="Malformed LINES"):
            MooringSystem.from_moordyn(path)


# ---------------------------------------------------------------------------
# Group #2 — content-aware header detection
# ---------------------------------------------------------------------------

class TestMoorDynHeaderVariants:
    """``_looks_like_header_row`` lets the parser tolerate decks
    shipped with a 1-row header (no units line). Previously the
    hard-coded ``pending_skip = 2`` ate the first data row.
    """

    def test_one_row_header_still_parses(self, tmp_path: pathlib.Path) -> None:
        """LINE TYPES section with ONLY a column-name header (no units
        line). The previous ``pending_skip = 2`` would have eaten the
        ``chain ...`` data row; the new content-aware skip stops the
        moment it sees a data-looking row.
        """
        path = _write_moordyn(
            tmp_path / "single_header.dat",
            _LINE_TYPES_1HEADER, _POINTS_2HEADER, _LINES_2HEADER, _OPTIONS,
        )
        from pybmodes.mooring import MooringSystem
        ms = MooringSystem.from_moordyn(path)
        assert "chain" in ms.line_types, (
            "single-header LINE TYPES should still produce a parsed "
            "LineType — the content-aware skip must stop before the "
            "data row."
        )
        # Sanity: the parsed properties match the source row.
        lt = ms.line_types["chain"]
        assert lt.diam == pytest.approx(0.10)
        assert lt.mass_per_length_air == pytest.approx(50.0)
        assert lt.EA == pytest.approx(1.0e9)

    def test_two_row_header_still_parses(self, tmp_path: pathlib.Path) -> None:
        """The standard MoorDyn convention (column names + units) is
        still accepted unchanged."""
        path = _write_moordyn(
            tmp_path / "two_header.dat",
            _LINE_TYPES_2HEADER, _POINTS_2HEADER, _LINES_2HEADER, _OPTIONS,
        )
        from pybmodes.mooring import MooringSystem
        ms = MooringSystem.from_moordyn(path)
        assert "chain" in ms.line_types
        assert ms.depth == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Group #3 — SubDyn adapter friendly errors
# ---------------------------------------------------------------------------

_SUBDYN_VALID = """\
----------- SubDyn MultiMember Support Structure Input File ---------------------------
Synthetic 3-joint monopile.
-------------------------- SIMULATION CONTROL -----------------------------------------
False            Echo        - Echo input data
---- STRUCTURE JOINTS: joints connect structure members ------------------
             3   NJoints     - Number of joints
JointID  JointXss  JointYss  JointZss  JointType  JointDirX  JointDirY  JointDirZ  JointStiff
  (-)      (m)       (m)       (m)       (-)        (-)        (-)        (-)      (Nm/rad)
   1     0.000     0.000   -20.000        1        0.0        0.0        0.0        0.0
   2     0.000     0.000   -10.000        1        0.0        0.0        0.0        0.0
   3     0.000     0.000     0.000        1        0.0        0.0        0.0        0.0
------------------- BASE REACTION JOINTS ----------------------------------------------
             1   NReact      - Number of reaction joints
RJointID  RctTDXss  RctTDYss  RctTDZss  RctRDXss  RctRDYss  RctRDZss  SSIfile
  (-)      (flag)    (flag)    (flag)    (flag)    (flag)    (flag)   (string)
   {react_id}     1     1     1     1     1     1     ""
------- INTERFACE JOINTS --------------------------------------------------------------
             1   NInterf     - Number of interface joints
IJointID  TPID  ItfTDXss  ItfTDYss  ItfTDZss  ItfRDXss  ItfRDYss  ItfRDZss
  (-)     (-)    (flag)    (flag)    (flag)    (flag)    (flag)    (flag)
   {iface_id}   1   1   1   1   1   1   1
----------------------------------- MEMBERS -------------------------------------------
             2   NMembers    - Number of members
MemberID  MJointID1  MJointID2  MPropSetID1  MPropSetID2  MType  MSpin/COSMID
  (-)        (-)        (-)         (-)          (-)       (-)    (deg/-)
   1          1          2           1            1         1c       0
   2          2          3           1            1         1c       0
------------------ CIRCULAR BEAM CROSS-SECTION PROPERTIES -----------------------------
             1   NPropSetsCyl - Number of circular cross-sections
PropSetID  YoungE       ShearG       MatDens   XsecD    XsecT
  (-)      (N/m2)       (N/m2)       (kg/m3)    (m)      (m)
   1     2.10000e+11  8.08000e+10   8500.00    6.000    0.060
"""


def test_subdyn_adapter_missing_reaction_joint_raises_valueerror(
    tmp_path: pathlib.Path,
) -> None:
    """A SubDyn deck whose ``BASE REACTION`` references a joint ID
    absent from ``STRUCTURE JOINTS`` raises a clear ``ValueError``
    naming the missing ID — not a bare ``StopIteration``."""
    deck = tmp_path / "bad_react.dat"
    deck.write_text(
        _SUBDYN_VALID.format(react_id=99, iface_id=3),  # 99 not in {1,2,3}
        encoding="latin-1",
    )

    # Build a minimal ElastoDyn main / tower stub so the adapter can
    # reach the joint lookup. The adapter inspects ``main.tower_ht`` /
    # ``main.tower_bs_ht`` AFTER the joint lookup, so we don't need
    # populated values — just attribute presence.
    from dataclasses import dataclass

    @dataclass
    class _StubMain:
        tower_ht: float = 90.0
        tower_bs_ht: float = 0.0

    from pybmodes.io.subdyn_reader import read_subdyn, to_pybmodes_pile_tower

    subdyn = read_subdyn(deck)
    with pytest.raises(ValueError, match="no joint with id=99.*reaction"):
        to_pybmodes_pile_tower(main=_StubMain(), tower=None, subdyn=subdyn)


def test_subdyn_adapter_missing_interface_joint_raises_valueerror(
    tmp_path: pathlib.Path,
) -> None:
    """Same diagnostic for a missing interface-joint ID."""
    deck = tmp_path / "bad_iface.dat"
    deck.write_text(
        _SUBDYN_VALID.format(react_id=1, iface_id=77),  # 77 not in {1,2,3}
        encoding="latin-1",
    )

    from dataclasses import dataclass

    @dataclass
    class _StubMain:
        tower_ht: float = 90.0
        tower_bs_ht: float = 0.0

    from pybmodes.io.subdyn_reader import read_subdyn, to_pybmodes_pile_tower

    subdyn = read_subdyn(deck)
    with pytest.raises(ValueError, match="no joint with id=77.*interface"):
        to_pybmodes_pile_tower(main=_StubMain(), tower=None, subdyn=subdyn)
