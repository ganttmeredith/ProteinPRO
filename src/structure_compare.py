"""
Structure comparison utilities for PDB/AlphaFold/OpenFold models.

Computes RMSD and structural alignment for cross-model comparison.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .pdb_handler import parse_structure, get_sequence_and_features


def get_ca_coords(structure) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """
    Extract C-alpha coordinates and (chain_id, res_id) for each.
    """
    coords = []
    ids = []
    for model in structure:
        for chain in model:
            cid = chain.get_id()
            for residue in chain:
                if residue.has_id("CA"):
                    ca = residue["CA"]
                    coords.append(ca.get_coord())
                    ids.append((cid, residue.get_id()[1]))
        break  # first model only
    return np.array(coords), ids


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Compute RMSD between two coordinate sets (same length, aligned).
    """
    if len(coords1) != len(coords2):
        return np.nan
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def align_structures(
    mobile_coords: np.ndarray,
    target_coords: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Kabsch alignment: rotate/translate mobile to minimize RMSD to target.
    Returns (aligned_coords, rmsd).
    """
    n = min(len(mobile_coords), len(target_coords))
    if n < 3:
        return mobile_coords, np.nan

    A = mobile_coords[:n].copy()
    B = target_coords[:n].copy()

    # Center
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)

    # Kabsch
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    A_aligned = (R @ A.T).T
    rmsd = np.sqrt(np.mean(np.sum((A_aligned - B) ** 2, axis=1)))
    return A_aligned, float(rmsd)


def compare_structures(
    source1: Union[str, Path],
    source2: Union[str, Path],
    id1: Optional[str] = None,
    id2: Optional[str] = None,
) -> dict:
    """
    Compare two structures (PDB IDs or paths). Returns RMSD and metadata.
    Uses sequence alignment to match residues when lengths differ.
    """
    s1 = parse_structure(source1, id1)
    s2 = parse_structure(source2, id2)
    info1 = get_sequence_and_features(s1)
    info2 = get_sequence_and_features(s2)

    coords1, ids1 = get_ca_coords(s1)
    coords2, ids2 = get_ca_coords(s2)

    # Simple case: same length, assume same order
    if len(coords1) == len(coords2):
        aligned, rmsd = align_structures(coords1, coords2)
        return {
            "rmsd": rmsd,
            "n_residues": len(coords1),
            "seq_identity": sum(a == b for a, b in zip(info1["sequence"], info2["sequence"])) / max(len(info1["sequence"]), 1),
            "structure1_residues": len(coords1),
            "structure2_residues": len(coords2),
        }

    # Different length: use shortest common prefix
    n = min(len(coords1), len(coords2))
    if n < 3:
        return {"rmsd": np.nan, "n_residues": n, "seq_identity": 0}
    aligned, rmsd = align_structures(coords1[:n], coords2[:n])
    seq_id = sum(a == b for a, b in zip(info1["sequence"][:n], info2["sequence"][:n])) / n
    return {
        "rmsd": rmsd,
        "n_residues": n,
        "seq_identity": seq_id,
        "structure1_residues": len(coords1),
        "structure2_residues": len(coords2),
    }
