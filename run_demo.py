#!/usr/bin/env python3
"""
ProteinPRO demo pipeline.

Run from project root:
  python run_demo.py

Demonstrates: PDB fetch → protein featurization → monomer featurization
→ design space sampling → stability prediction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pdb_handler import fetch_pdb, featurize_protein
from src.monomer_featurizer import featurize_all_monomers, composition_to_polymer_features
from src.stability_model import StabilityPredictor, sample_design_space


def main():
    print("ProteinPRO Demo")
    print("=" * 50)

    # 1. Fetch and featurize protein (lipase example)
    pdb_id = "1LBS"
    print(f"\n1. Fetching PDB {pdb_id}...")
    path = fetch_pdb(pdb_id)
    print(f"   Cached at {path}")

    print("\n2. Featurizing protein...")
    pf = featurize_protein(path, pdb_id)
    print(f"   Residues: {pf['n_residues']}")
    print(f"   Mean hydrophobicity: {pf['mean_hydrophobicity']:.3f}")
    print(f"   Net charge density: {pf['net_charge_density']:.3f}")

    # 2. Featurize monomers
    print("\n3. Featurizing monomers...")
    mf_df = featurize_all_monomers()
    print(mf_df[["monomer_name", "MolWt", "LogP", "TPSA"]].to_string(index=False))

    # 3. Sample design space and predict
    print("\n4. Sampling design space and predicting stability...")
    compositions = sample_design_space(n_samples=5, seed=42)
    predictor = StabilityPredictor(use_surrogate=True)

    for i, comp in enumerate(compositions):
        comp_str = ", ".join(f"{k}:{v:.2f}" for k, v in comp.items())
        score, _ = predictor.predict(path, comp, pdb_id)
        print(f"   Formulation {i+1}: {comp_str} → score = {score:.4f}")

    # 4. Rank formulations
    print("\n5. Ranking top formulations...")
    df = predictor.rank_formulations(path, n_candidates=20, pdb_id=pdb_id)
    print(df[["composition", "stability_score"]].head(5).to_string(index=False))

    print("\n" + "=" * 50)
    print("Demo complete. Run 'streamlit run app.py' for the web interface.")


if __name__ == "__main__":
    main()
