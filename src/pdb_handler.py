"""
PDB structure fetching, parsing, and protein featurization.

Supports PDB IDs, local PDB/CIF files, and is extensible for
AlphaFold/OpenFold structures via the same parsing pipeline.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import requests
import yaml
from Bio.PDB import PDBParser, MMCIFParser, is_aa
from Bio.SeqUtils import seq1

# Kyte-Doolittle hydrophobicity scale (standard)
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2, "X": 0.0,
}

# Surface accessibility classes (from DSSP-like simplified)
POSITIVE_RESIDUES = {"K", "R", "H"}
NEGATIVE_RESIDUES = {"D", "E"}
HYDROPHOBIC_RESIDUES = {"A", "V", "L", "I", "M", "F", "W", "P", "G"}
POLAR_RESIDUES = {"S", "T", "N", "Q", "Y", "C"}


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_pdb(pdb_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Fetch PDB structure from RCSB. Returns path to local file.
    """
    pdb_id = pdb_id.upper().strip()
    config = load_config()
    cache_dir = cache_dir or config.get("pdb", {}).get("cache_dir", ".pdb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{pdb_id}.pdb")

    if os.path.exists(local_path):
        return local_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    timeout = config.get("pdb", {}).get("fetch_timeout", 10)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(local_path, "w") as f:
        f.write(r.text)
    return local_path


def parse_structure(
    source: Union[str, Path],
    pdb_id: Optional[str] = None,
) -> "Bio.PDB.Structure.Structure":
    """
    Parse PDB or CIF file. source can be file path or PDB ID.
    """
    source = str(source).strip()
    if len(source) == 4 and source.isalnum():
        path = fetch_pdb(source)
        pdb_id = pdb_id or source
    else:
        path = source
        pdb_id = pdb_id or Path(path).stem

    if path.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, path)


def get_sequence_and_features(structure) -> dict:
    """
    Extract sequence and residue-level features from structure.
    Returns dict with sequence, per-residue hydrophobicity, charge, etc.
    """
    residues = []
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue):
                    continue
                try:
                    resname = residue.get_resname()
                    letter = seq1(resname)
                except (KeyError, ValueError):
                    continue
                seq.append(letter)
                residues.append(residue)

    seq_str = "".join(seq)
    n = len(seq_str)

    # Per-residue features
    hydrophobicity = np.array([HYDROPHOBICITY.get(a, 0) for a in seq_str])
    is_positive = np.array([1 if a in POSITIVE_RESIDUES else 0 for a in seq_str])
    is_negative = np.array([1 if a in NEGATIVE_RESIDUES else 0 for a in seq_str])
    is_hydrophobic = np.array([1 if a in HYDROPHOBIC_RESIDUES else 0 for a in seq_str])
    is_polar = np.array([1 if a in POLAR_RESIDUES else 0 for a in seq_str])

    # Global protein features (for ML)
    return {
        "sequence": seq_str,
        "n_residues": n,
        "mean_hydrophobicity": float(np.mean(hydrophobicity)),
        "std_hydrophobicity": float(np.std(hydrophobicity)),
        "fraction_positive": float(np.mean(is_positive)),
        "fraction_negative": float(np.mean(is_negative)),
        "fraction_hydrophobic": float(np.mean(is_hydrophobic)),
        "fraction_polar": float(np.mean(is_polar)),
        "net_charge_density": float(np.mean(is_positive) - np.mean(is_negative)),
        "hydrophobicity_profile": hydrophobicity.tolist(),
        "charge_profile": (is_positive - is_negative).tolist(),
    }


def featurize_protein(source: Union[str, Path], pdb_id: Optional[str] = None) -> dict:
    """
    Full pipeline: parse structure and return protein feature vector
    suitable for stability prediction model.
    """
    structure = parse_structure(source, pdb_id)
    features = get_sequence_and_features(structure)
    features["structure"] = structure
    return features


def get_residue_roles_for_visualization(structure) -> dict:
    """
    Extract residue (chain, resi) for each chemical type for 3D highlighting.
    Returns dict: {"polar": [(chain, resi), ...], "positive": [...], "negative": [...], "hydrophobic": [...]}
    Charge takes precedence over polar (K,R,H = positive; D,E = negative).
    """
    out = {"polar": [], "positive": [], "negative": [], "hydrophobic": []}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                if not is_aa(residue):
                    continue
                try:
                    resname = residue.get_resname()
                    letter = seq1(resname)
                    resi = residue.get_id()[1]
                except (KeyError, ValueError):
                    continue
                if letter in POSITIVE_RESIDUES:
                    out["positive"].append((chain_id, resi))
                elif letter in NEGATIVE_RESIDUES:
                    out["negative"].append((chain_id, resi))
                elif letter in POLAR_RESIDUES:
                    out["polar"].append((chain_id, resi))
                elif letter in HYDROPHOBIC_RESIDUES:
                    out["hydrophobic"].append((chain_id, resi))
    return out


def get_coordinates_for_visualization(structure) -> str:
    """Return PDB string for 3D viewer (py3Dmol)."""
    import io
    from Bio.PDB import PDBIO
    buf = io.StringIO()
    io_obj = PDBIO()
    io_obj.set_structure(structure)
    io_obj.save(buf)
    return buf.getvalue()
