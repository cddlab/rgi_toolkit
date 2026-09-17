"""Approximate uncertainties for geometry measured from reference coordinates.

Bond/angle defaults follow Gemmi's coordinate-derived restraint fallback
(`make_chemcomp_with_restraints`, v0.7.5, src/topo.cpp). These are not measured
uncertainties of the predictor coordinates or replacements for dictionary ESDs.
"""

from __future__ import annotations

import math

import numpy as np

from rgi_toolkit._monlib_records import NamedRestraint, chiral_volume_esd

BOND_ESD = 0.02  # Angstrom; Gemmi's coordinate-derived fallback.
ANGLE_ESD = math.radians(3.0)  # Gemmi's 3-degree fallback, in internal units.
PLANE_ESD = 0.02  # Angstrom; approximate CCP4 planar-group uncertainty.
CISTRANS_ESD = math.radians(5.0)  # Same approximation as other sp2 torsions.


def reference_chiral_esd(lengths, angles):
    """Propagate three reference bond/angle ESDs using the dictionary formula.

    Lengths are center-to-neighbor distances; angles are between neighbor pairs
    (1, 2), (2, 3), (3, 1), in radians. Return NaN for degenerate geometry so only
    an enabled, retained chiral restraint needs to reject the invalid reference.
    """
    if not np.isfinite([*lengths, *angles]).all():
        return float("nan")
    bonds = [
        NamedRestraint((0, i), float(length), BOND_ESD)
        for i, length in enumerate(lengths, 1)
    ]
    bends = [
        NamedRestraint((i, 0, j), math.degrees(angle), math.degrees(ANGLE_ESD))
        for (i, j), angle in zip(((1, 2), (2, 3), (3, 1)), angles, strict=True)
    ]
    result = chiral_volume_esd(NamedRestraint((0, 1, 2, 3)), bonds, bends)
    return result[1] if result is not None else float("nan")


def inverse_variance_weights(esds, weight, kind, *, use_esd=True):
    """Validate reference ESDs and optionally apply inverse-variance weights."""
    sigma = np.asarray(esds, dtype=np.float64)
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError(f"reference conformer {kind}: ESD must be finite and > 0")
    if not use_esd:
        return np.full_like(sigma, weight)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        weights = weight / np.square(sigma)
    if not np.isfinite(weights).all():
        raise ValueError(f"reference conformer {kind}: ESD produces a nonfinite weight")
    return weights
