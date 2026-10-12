"""Assess graph-defined E/Z against every heavy-atom substituent pair."""

from itertools import product

import numpy as np
from rdkit import Chem

EZ_METHOD = "heavy-substituent-consensus-v1"
_EPS = 1e-7


def assess_ez(mol, coords, bonds):
    """Return match, mismatch, or None (undefined) for each requested atom pair.

    Coordinates must follow the reference graph's atom order. The graph's stereo
    atoms define one cis/trans pair; choosing the other substituent on either end
    reverses its expected sign. Every heavy substituent pair must agree. Conflicting
    signs, perpendicular projections, and collinear/zero-length bonds are undefined.
    Undefined sites fail the preservation metric and remain in its denominator.
    """
    reference = Chem.Mol(mol)
    Chem.AssignStereochemistry(reference, cleanIt=False, force=True)
    xyz = np.asarray(coords, dtype=float)
    if xyz.shape != (reference.GetNumAtoms(), 3):
        raise ValueError("E/Z coordinates must match the reference atom order")
    result = {}
    for pair in bonds:
        result[pair] = None
        bond = reference.GetBondBetweenAtoms(*pair)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError("E/Z assessment requires a reference double bond")
        stereo = bond.GetStereo()
        if stereo in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ):
            expected = 1
        elif stereo in (Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE):
            expected = -1
        else:
            continue
        anchors = list(bond.GetStereoAtoms())
        if len(anchors) != 2:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        axis = xyz[j] - xyz[i]
        length = np.linalg.norm(axis)
        if not np.isfinite(length) or length <= _EPS:
            continue
        axis /= length
        ends = []
        for center, other, anchor in ((i, j, anchors[0]), (j, i, anchors[1])):
            neighbors = [
                n.GetIdx()
                for n in reference.GetAtomWithIdx(center).GetNeighbors()
                if n.GetIdx() != other and n.GetAtomicNum() > 1
            ]
            projections = []
            if 1 <= len(neighbors) <= 2:
                for neighbor in neighbors:
                    vector = xyz[neighbor] - xyz[center]
                    norm = np.linalg.norm(vector)
                    if not np.isfinite(norm) or norm <= _EPS:
                        break
                    vector /= norm
                    projection = vector - np.dot(vector, axis) * axis
                    norm = np.linalg.norm(projection)
                    if norm <= _EPS:
                        break
                    sign = 1 if neighbor == anchor else -1
                    projections.append(sign * projection / norm)
            if not projections or len(projections) != len(neighbors):
                break
            ends.append(projections)
        if len(ends) != 2:
            continue
        cosines = [expected * np.dot(a, b) for a, b in product(*ends)]
        if all(value > _EPS for value in cosines):
            result[pair] = True
        elif all(value < -_EPS for value in cosines):
            result[pair] = False
    return result
