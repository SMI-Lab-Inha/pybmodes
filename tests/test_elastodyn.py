"""Tests for elastodyn/params.py and elastodyn/writer.py."""

from __future__ import annotations

import pathlib

import pytest

from pybmodes.models import RotatingBlade, Tower
from pybmodes.elastodyn import (
    BladeElastoDynParams,
    TowerElastoDynParams,
    compute_blade_params,
    compute_tower_params,
    patch_dat,
)


CERT_DIR = pathlib.Path(__file__).parent / "data" / "certtest"
ROOT_DIR = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Blade params
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBladeParams:

    @pytest.fixture(autouse=True)
    def compute(self):
        modal = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi").run(n_modes=10)
        self.params = compute_blade_params(modal)

    def test_returns_blade_params(self):
        assert isinstance(self.params, BladeElastoDynParams)

    def test_fl1sh_constraint(self):
        c = self.params.BldFl1Sh
        assert c.c2 + c.c3 + c.c4 + c.c5 + c.c6 == pytest.approx(1.0, abs=1e-12)

    def test_fl2sh_constraint(self):
        c = self.params.BldFl2Sh
        assert c.c2 + c.c3 + c.c4 + c.c5 + c.c6 == pytest.approx(1.0, abs=1e-12)

    def test_edgsh_constraint(self):
        c = self.params.BldEdgSh
        assert c.c2 + c.c3 + c.c4 + c.c5 + c.c6 == pytest.approx(1.0, abs=1e-12)

    def test_as_dict_keys(self):
        d = self.params.as_dict()
        expected = {
            "BldFl1Sh(2)", "BldFl1Sh(3)", "BldFl1Sh(4)", "BldFl1Sh(5)", "BldFl1Sh(6)",
            "BldFl2Sh(2)", "BldFl2Sh(3)", "BldFl2Sh(4)", "BldFl2Sh(5)", "BldFl2Sh(6)",
            "BldEdgSh(2)", "BldEdgSh(3)", "BldEdgSh(4)", "BldEdgSh(5)", "BldEdgSh(6)",
        }
        assert set(d.keys()) == expected

    def test_as_dict_fl1sh_sums_to_one(self):
        d = self.params.as_dict()
        total = sum(d[f"BldFl1Sh({k})"] for k in range(2, 7))
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_fl1sh_rms_low(self):
        # First flap is smooth; poly fit should be good
        assert self.params.BldFl1Sh.rms_residual < 0.05

    def test_edgsh_rms_low(self):
        assert self.params.BldEdgSh.rms_residual < 0.05

    def test_insufficient_modes_raises(self):
        # Only 1 mode → can't form BldFl2Sh
        modal = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi").run(n_modes=1)
        with pytest.raises(ValueError, match="flap modes"):
            compute_blade_params(modal)


# ---------------------------------------------------------------------------
# Tower params
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTowerParams:

    @pytest.fixture(autouse=True)
    def compute(self):
        modal = Tower(CERT_DIR / "Test03_tower.bmi").run(n_modes=10)
        self.params = compute_tower_params(modal)

    def test_returns_tower_params(self):
        assert isinstance(self.params, TowerElastoDynParams)

    def test_fa1_constraint(self):
        c = self.params.TwFAM1Sh
        assert c.c2 + c.c3 + c.c4 + c.c5 + c.c6 == pytest.approx(1.0, abs=1e-12)

    def test_fa2_constraint(self):
        c = self.params.TwFAM2Sh
        assert c.c2 + c.c3 + c.c4 + c.c5 + c.c6 == pytest.approx(1.0, abs=1e-12)

    def test_ss1_constraint(self):
        c = self.params.TwSSM1Sh
        assert c.c2 + c.c3 + c.c4 + c.c5 + c.c6 == pytest.approx(1.0, abs=1e-12)

    def test_ss2_constraint(self):
        c = self.params.TwSSM2Sh
        assert c.c2 + c.c3 + c.c4 + c.c5 + c.c6 == pytest.approx(1.0, abs=1e-12)

    def test_as_dict_keys(self):
        d = self.params.as_dict()
        expected = {
            "TwFAM1Sh(2)", "TwFAM1Sh(3)", "TwFAM1Sh(4)", "TwFAM1Sh(5)", "TwFAM1Sh(6)",
            "TwFAM2Sh(2)", "TwFAM2Sh(3)", "TwFAM2Sh(4)", "TwFAM2Sh(5)", "TwFAM2Sh(6)",
            "TwSSM1Sh(2)", "TwSSM1Sh(3)", "TwSSM1Sh(4)", "TwSSM1Sh(5)", "TwSSM1Sh(6)",
            "TwSSM2Sh(2)", "TwSSM2Sh(3)", "TwSSM2Sh(4)", "TwSSM2Sh(5)", "TwSSM2Sh(6)",
        }
        assert set(d.keys()) == expected

    def test_fa1_rms_low(self):
        assert self.params.TwFAM1Sh.rms_residual < 0.05

    def test_insufficient_modes_raises(self):
        modal = Tower(CERT_DIR / "Test03_tower.bmi").run(n_modes=2)
        with pytest.raises(ValueError, match="FA modes"):
            compute_tower_params(modal)


@pytest.mark.integration
class TestIEA15TowerParams:

    @pytest.fixture(autouse=True)
    def compute(self):
        modal = Tower(ROOT_DIR / "IEA-15-240-RWT_BModes_tower.bmi").run(n_modes=12)
        self.params = compute_tower_params(modal)

    def test_first_family_fits_are_reasonable(self):
        assert self.params.TwFAM1Sh.rms_residual < 0.10
        assert self.params.TwSSM1Sh.rms_residual < 0.10

    def test_second_family_fits_skip_poor_support_modes(self):
        assert self.params.TwFAM2Sh.rms_residual < 0.10
        assert self.params.TwSSM2Sh.rms_residual < 0.10


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

BLADE_DAT_TEMPLATE = """\
------- BLADE MODE SHAPES ------------------------------------------
          1   BldFl1Sh(2)   - Blade 1 flap mode 1, coeff of x^2
          0   BldFl1Sh(3)   - Blade 1 flap mode 1, coeff of x^3
          0   BldFl1Sh(4)   - Blade 1 flap mode 1, coeff of x^4
          0   BldFl1Sh(5)   - Blade 1 flap mode 1, coeff of x^5
          0   BldFl1Sh(6)   - Blade 1 flap mode 1, coeff of x^6
          1   BldFl2Sh(2)   - Blade 1 flap mode 2, coeff of x^2
          0   BldFl2Sh(3)   - Blade 1 flap mode 2, coeff of x^3
          0   BldFl2Sh(4)   - Blade 1 flap mode 2, coeff of x^4
          0   BldFl2Sh(5)   - Blade 1 flap mode 2, coeff of x^5
          0   BldFl2Sh(6)   - Blade 1 flap mode 2, coeff of x^6
          1   BldEdgSh(2)   - Blade 1 edge mode 1, coeff of x^2
          0   BldEdgSh(3)   - Blade 1 edge mode 1, coeff of x^3
          0   BldEdgSh(4)   - Blade 1 edge mode 1, coeff of x^4
          0   BldEdgSh(5)   - Blade 1 edge mode 1, coeff of x^5
          0   BldEdgSh(6)   - Blade 1 edge mode 1, coeff of x^6
"""

TOWER_DAT_TEMPLATE = """\
---------------------- TOWER FORE-AFT MODE SHAPES ------------------------------
     1.0000   TwFAM1Sh(2) - Mode 1, coefficient of x^2 term
     0.0000   TwFAM1Sh(3) -       , coefficient of x^3 term
     0.0000   TwFAM1Sh(4) -       , coefficient of x^4 term
     0.0000   TwFAM1Sh(5) -       , coefficient of x^5 term
     0.0000   TwFAM1Sh(6) -       , coefficient of x^6 term
     1.0000   TwFAM2Sh(2) - Mode 2, coefficient of x^2 term
     0.0000   TwFAM2Sh(3) -       , coefficient of x^3 term
     0.0000   TwFAM2Sh(4) -       , coefficient of x^4 term
     0.0000   TwFAM2Sh(5) -       , coefficient of x^5 term
     0.0000   TwFAM2Sh(6) -       , coefficient of x^6 term
---------------------- TOWER SIDE-TO-SIDE MODE SHAPES --------------------------
     1.0000   TwSSM1Sh(2) - Mode 1, coefficient of x^2 term
     0.0000   TwSSM1Sh(3) -       , coefficient of x^3 term
     0.0000   TwSSM1Sh(4) -       , coefficient of x^4 term
     0.0000   TwSSM1Sh(5) -       , coefficient of x^5 term
     0.0000   TwSSM1Sh(6) -       , coefficient of x^6 term
     1.0000   TwSSM2Sh(2) - Mode 2, coefficient of x^2 term
     0.0000   TwSSM2Sh(3) -       , coefficient of x^3 term
     0.0000   TwSSM2Sh(4) -       , coefficient of x^4 term
     0.0000   TwSSM2Sh(5) -       , coefficient of x^5 term
     0.0000   TwSSM2Sh(6) -       , coefficient of x^6 term
"""


class TestWriter:

    @pytest.mark.integration
    def test_blade_patch_roundtrip(self, tmp_path):
        dat = tmp_path / "ElastoDyn.dat"
        dat.write_text(BLADE_DAT_TEMPLATE, encoding='utf-8')

        modal = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi").run(n_modes=10)
        params = compute_blade_params(modal)
        patch_dat(dat, params)

        text = dat.read_text(encoding='utf-8')
        d = params.as_dict()
        # Each patched line should contain the new value as its first token
        for name, expected in d.items():
            for line in text.splitlines():
                tokens = line.split()
                if len(tokens) >= 2 and tokens[1] == name:
                    assert float(tokens[0]) == pytest.approx(expected, rel=1e-6)
                    break
            else:
                pytest.fail(f"Parameter {name} not found in patched file")

    @pytest.mark.integration
    def test_tower_patch_roundtrip(self, tmp_path):
        dat = tmp_path / "ElastoDyn_tower.dat"
        dat.write_text(TOWER_DAT_TEMPLATE, encoding='utf-8')

        modal = Tower(CERT_DIR / "Test03_tower.bmi").run(n_modes=10)
        params = compute_tower_params(modal)
        patch_dat(dat, params)

        text = dat.read_text(encoding='utf-8')
        d = params.as_dict()
        for name, expected in d.items():
            for line in text.splitlines():
                tokens = line.split()
                if len(tokens) >= 2 and tokens[1] == name:
                    assert float(tokens[0]) == pytest.approx(expected, rel=1e-6)
                    break
            else:
                pytest.fail(f"Parameter {name} not found in patched file")

    def test_missing_param_raises(self, tmp_path):
        dat = tmp_path / "incomplete.dat"
        dat.write_text("nothing useful here\n", encoding='utf-8')

        modal = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi").run(n_modes=10)
        params = compute_blade_params(modal)
        with pytest.raises(KeyError, match="BldFl1Sh"):
            patch_dat(dat, params)

    def test_preserves_indentation(self, tmp_path):
        dat = tmp_path / "ElastoDyn.dat"
        dat.write_text(BLADE_DAT_TEMPLATE, encoding='utf-8')

        modal = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi").run(n_modes=10)
        params = compute_blade_params(modal)
        patch_dat(dat, params)

        for line in dat.read_text().splitlines():
            if "BldFl1Sh(2)" in line:
                # Leading spaces should be preserved
                assert line.startswith(" ")
                break

    def test_preserves_comment(self, tmp_path):
        dat = tmp_path / "ElastoDyn.dat"
        dat.write_text(BLADE_DAT_TEMPLATE, encoding='utf-8')

        modal = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi").run(n_modes=10)
        params = compute_blade_params(modal)
        patch_dat(dat, params)

        for line in dat.read_text().splitlines():
            if "BldFl1Sh(2)" in line:
                assert "coeff of x^2" in line
                break
