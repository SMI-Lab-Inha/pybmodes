"""Shared pytest fixtures — paths to CertTest reference data."""

import pathlib
import pytest

CERT_DIR = pathlib.Path(__file__).parent / "data" / "certtest"
REF_DIR = CERT_DIR / "expected"


@pytest.fixture
def cert_dir() -> pathlib.Path:
    return CERT_DIR


@pytest.fixture
def ref_dir() -> pathlib.Path:
    return REF_DIR


@pytest.fixture
def blade_bmi(cert_dir) -> pathlib.Path:
    return cert_dir / "Test01_nonunif_blade.bmi"


@pytest.fixture
def blade_tip_bmi(cert_dir) -> pathlib.Path:
    return cert_dir / "Test02_blade_with_tip_mass.bmi"


@pytest.fixture
def tower_bmi(cert_dir) -> pathlib.Path:
    return cert_dir / "Test03_tower.bmi"


@pytest.fixture
def wire_tower_bmi(cert_dir) -> pathlib.Path:
    return cert_dir / "Test04_wires_supported_tower.bmi"


@pytest.fixture
def blade_sec_props(cert_dir) -> pathlib.Path:
    return cert_dir / "blade_sec_props.dat"


@pytest.fixture
def tower_sec_props(cert_dir) -> pathlib.Path:
    return cert_dir / "tower_sec_props.dat"


@pytest.fixture
def blade_ref_out(ref_dir) -> pathlib.Path:
    return ref_dir / "Test01_nonunif_blade.out"


@pytest.fixture
def tower_ref_out(ref_dir) -> pathlib.Path:
    return ref_dir / "Test03_tower.out"
