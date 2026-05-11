"""ModalResult dataclass returned by RotatingBlade and Tower."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pybmodes.fem.normalize import NodeModeShape


@dataclass
class ModalResult:
    """Frequencies and mode shapes from a single FEM solve.

    Attributes
    ----------
    frequencies : (n_modes,) array of natural frequencies in Hz.
    shapes : list of :class:`NodeModeShape`, one per mode, ordered root
        to tip.
    participation : (n_modes, 3) array of energy fractions in the
        per-mode (flap or FA, edge or SS, torsion) axes — populated by
        downstream classification code such as
        :func:`pybmodes.campbell._participation`. Each row sums to 1.
        ``None`` when not yet computed; included in the saved archive
        only when set.
    fit_residuals : optional ``{block_name: rms_value}`` dict of
        polynomial-fit RMS residuals — populated by
        :func:`pybmodes.elastodyn.compute_tower_params` /
        ``compute_blade_params`` callers that want to embed the fit
        quality in the serialised result. ``None`` when not set.
    metadata : optional metadata dict (pyBmodes version, source file,
        save timestamp, git hash) attached automatically by
        :meth:`save` / :meth:`to_json` if not already populated.
    """

    frequencies: np.ndarray
    shapes: list[NodeModeShape]
    participation: np.ndarray | None = None
    fit_residuals: dict[str, float] | None = None
    metadata: dict[str, Any] | None = field(default=None)

    # ------------------------------------------------------------------
    # NPZ round-trip
    # ------------------------------------------------------------------

    def save(
        self, path: str | pathlib.Path, *,
        source_file: str | pathlib.Path | None = None,
    ) -> None:
        """Write the result to a ``.npz`` archive.

        ``source_file`` is recorded in the metadata when supplied
        (typically the BMI / ElastoDyn deck the solve came from).
        """
        from pybmodes.io._serialize import _capture_metadata, _metadata_to_npz_value

        path = pathlib.Path(path)
        if self.metadata is None:
            meta = _capture_metadata(source_file=source_file)
        else:
            meta = dict(self.metadata)

        n_modes = self.frequencies.size
        if not self.shapes:
            # Allow an empty result (e.g. failed solve) to round-trip.
            shared_span = np.empty(0, dtype=float)
            stacked: dict[str, np.ndarray] = {
                name: np.empty((0, 0), dtype=float)
                for name in ("flap_disp", "flap_slope", "lag_disp",
                             "lag_slope", "twist")
            }
            mode_numbers = np.empty(0, dtype=int)
        else:
            shared_span = np.asarray(self.shapes[0].span_loc, dtype=float)
            stacked = {
                name: np.stack([np.asarray(getattr(s, name), dtype=float)
                                for s in self.shapes])
                for name in ("flap_disp", "flap_slope", "lag_disp",
                             "lag_slope", "twist")
            }
            mode_numbers = np.array([int(s.mode_number) for s in self.shapes])

        kwargs: dict[str, np.ndarray] = {
            "frequencies": np.asarray(self.frequencies, dtype=float),
            "mode_numbers": mode_numbers,
            "span_loc": shared_span,
            **stacked,
            "__meta__": _metadata_to_npz_value(meta),
        }
        if self.participation is not None:
            kwargs["participation"] = np.asarray(self.participation, dtype=float)
        if self.fit_residuals is not None:
            keys = list(self.fit_residuals.keys())
            kwargs["fit_residual_keys"] = np.array(keys, dtype=object)
            kwargs["fit_residual_values"] = np.array(
                [float(self.fit_residuals[k]) for k in keys], dtype=float,
            )
        # Verify n_modes consistency before write.
        if mode_numbers.size and n_modes != mode_numbers.size:
            raise ValueError(
                f"len(frequencies)={n_modes} != len(shapes)={mode_numbers.size}"
            )
        # The numpy stub for savez_compressed mis-types **kwargs as a
        # positional ``bool`` first arg; the call is correct at runtime.
        np.savez_compressed(path, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ModalResult":
        """Read a result back from a ``.npz`` archive saved by
        :meth:`save`. The reconstructed instance is value-equal to the
        original modulo numpy dtype promotion."""
        from pybmodes.io._serialize import _metadata_from_npz_value

        path = pathlib.Path(path)
        with np.load(path, allow_pickle=True) as npz:
            frequencies = np.asarray(npz["frequencies"], dtype=float)
            mode_numbers = np.asarray(npz["mode_numbers"], dtype=int)
            span_loc = np.asarray(npz["span_loc"], dtype=float)
            arrays = {
                name: np.asarray(npz[name], dtype=float)
                for name in ("flap_disp", "flap_slope", "lag_disp",
                             "lag_slope", "twist")
            }
            metadata = _metadata_from_npz_value(npz["__meta__"])
            participation: np.ndarray | None = None
            if "participation" in npz.files:
                participation = np.asarray(npz["participation"], dtype=float)
            fit_residuals: dict[str, float] | None = None
            if "fit_residual_keys" in npz.files:
                keys = [str(k) for k in npz["fit_residual_keys"]]
                vals = [float(v) for v in npz["fit_residual_values"]]
                fit_residuals = dict(zip(keys, vals))

        shapes = [
            NodeModeShape(
                mode_number=int(mode_numbers[i]),
                freq_hz=float(frequencies[i]),
                span_loc=span_loc,
                flap_disp=arrays["flap_disp"][i],
                flap_slope=arrays["flap_slope"][i],
                lag_disp=arrays["lag_disp"][i],
                lag_slope=arrays["lag_slope"][i],
                twist=arrays["twist"][i],
            )
            for i in range(int(mode_numbers.size))
        ]
        return cls(
            frequencies=frequencies,
            shapes=shapes,
            participation=participation,
            fit_residuals=fit_residuals,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # JSON round-trip
    # ------------------------------------------------------------------

    def to_json(
        self, path: str | pathlib.Path, *,
        source_file: str | pathlib.Path | None = None,
    ) -> None:
        """Write the result to a JSON file. Arrays are emitted as
        nested lists; metadata is embedded under ``"metadata"``."""
        from pybmodes.io._serialize import _capture_metadata

        path = pathlib.Path(path)
        meta = self.metadata if self.metadata is not None else _capture_metadata(
            source_file=source_file,
        )
        payload: dict[str, Any] = {
            "schema_version": "1",
            "metadata": meta,
            "frequencies": [float(f) for f in self.frequencies],
            "shapes": [
                {
                    "mode_number": int(s.mode_number),
                    "freq_hz": float(s.freq_hz),
                    "span_loc": [float(x) for x in s.span_loc],
                    "flap_disp": [float(x) for x in s.flap_disp],
                    "flap_slope": [float(x) for x in s.flap_slope],
                    "lag_disp": [float(x) for x in s.lag_disp],
                    "lag_slope": [float(x) for x in s.lag_slope],
                    "twist": [float(x) for x in s.twist],
                }
                for s in self.shapes
            ],
            "participation": (
                [[float(c) for c in row] for row in self.participation]
                if self.participation is not None else None
            ),
            "fit_residuals": (
                {k: float(v) for k, v in self.fit_residuals.items()}
                if self.fit_residuals is not None else None
            ),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "ModalResult":
        """Read a result back from a JSON file saved by
        :meth:`to_json`."""
        path = pathlib.Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        shapes = [
            NodeModeShape(
                mode_number=int(s["mode_number"]),
                freq_hz=float(s["freq_hz"]),
                span_loc=np.asarray(s["span_loc"], dtype=float),
                flap_disp=np.asarray(s["flap_disp"], dtype=float),
                flap_slope=np.asarray(s["flap_slope"], dtype=float),
                lag_disp=np.asarray(s["lag_disp"], dtype=float),
                lag_slope=np.asarray(s["lag_slope"], dtype=float),
                twist=np.asarray(s["twist"], dtype=float),
            )
            for s in payload["shapes"]
        ]
        participation = (
            np.asarray(payload["participation"], dtype=float)
            if payload.get("participation") is not None else None
        )
        fit_residuals = (
            {k: float(v) for k, v in payload["fit_residuals"].items()}
            if payload.get("fit_residuals") is not None else None
        )
        return cls(
            frequencies=np.asarray(payload["frequencies"], dtype=float),
            shapes=shapes,
            participation=participation,
            fit_residuals=fit_residuals,
            metadata=payload.get("metadata"),
        )
