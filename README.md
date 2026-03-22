# ProteinPRO: Protein Polymer Reactivity Optimization

Predict polymer-protein hybrid formulation stability from PDB structures and monomer compositions using chemical descriptor featurization and machine learning.

---

## Overview

ProteinPRO combines:

1. **PDB-derived protein featurization** — hydrophobicity, charge distribution, polarity (Kyte-Doolittle, residue composition)
2. **Chemical descriptor featurization** — RDKit molecular descriptors for PET-RAFT monomers (SPMA, TMAEMA, DEAEMA, DMAPMA, HPMA, PEGMA, BMA, EHMA)
3. **Design space exploration** — restricted monomer composition sampling
4. **Stability prediction** — Multiple models trained on previous stabilization data*)

*note* previous stabilization data is not for public domain; rather, sample data is use. Perform your own stabilization measurements and improve model prediction.

---

## Quick Start

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install
pip install -r requirements.txt

# Run demo
python run_demo.py

# Launch web app
streamlit run app.py
```

---

## Project Structure

```
ProteinPRO/
├── app.py              # Streamlit app housing
├── run_demo.py         # CLI demo one-shot
├── config.yaml         # Monomer SMILES, chemical design space
├── requirements.txt    # Necessary installs
├── src/
│   ├── __init__.py
│   ├── gpr_predictor.py       # Gaussian Process Regressor model loading
│   ├── monomer_featurizer.py # RDKit chemical descriptors
│   ├── pdb_handler.py       # Protein Data Bank file retrieval, parsing, and protein featurization
│   ├── stability_data_analysis.py  # Perform ML model featurization and normalization
│   ├── stability_model.py   # ML model selection, design space sampling
│   ├── structure_compare.py  # Root Mean Square Deviation
│   └── user_pdb_cache.py   # Save previous protein structures
|   ├── /integrations/
```

---

## Web App Features

- **PDB ID or file upload** — retrieve from RCSB or upload local PDB/CIF
- **3D structure visualization** — py3Dmol cartoon view
- **Monomer composition** — set molar fractions for each PET-RAFT monomer
- **Stability prediction** — real-time score for current formulation
- **Design space exploration** — sample and rank formulations by predicted stability
- **Download results** — export rankings as CSV
- **Ask AI (Gemini and ElevenLabs)** — natural language formulation optimization triage
- **Auth0 login** — Sign in to save PDB files to your personal cache
- **Your saved structures** — Load previously fetched/uploaded structures when logged in

---

## Deployment (Hack Duke Demo)

### Local
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### DigitalOcean App Platform
1. Push repo to GitHub
2. Create new App → Source: GitHub repo
3. Build: `pip install -r requirements.txt`
4. Run: `streamlit run app.py --server.port 8080 --server.address 0.0.0.0`
5. Set `PORT=8080` in environment

### Streamlit Cloud
1. Connect GitHub repo
2. Select `app.py` as main file
3. Add `packages.txt` if needed for system deps

---

## Configuration

Edit `config.yaml` to:
- Add or modify monomer SMILES
- Adjust design space bounds (`min_monomer_fraction`, `max_monomer_fraction`)
- Change PDB cache location

---

## Hack Duke 2026 Tech Stack

| Technology | Integration | Award Category |
|------------|-------------|----------------|
| **GitHub** | Repo, version control | Grand Prize - Health Track|
| **Streamlit** | Web app framework | Best Use of AI |
| **Gemini API** | "Ask AI" tab — natural language formulation advice | Best Use of Gemini API |
| **DigitalOcean** | App Platform deployment | Best Use of DigitalOcean |
| **Auth0** | User authentication (add keys to enable) | Best Use of Auth0 |

### API Keys

Copy `.env.example` to `.env` and add keys. Or use Streamlit secrets (`.streamlit/secrets.toml`):

```toml
GEMINI_API_KEY = "your-key"
ELEVENLABS_API_KEY = "your-key"
```

- **Gemini**: https://aistudio.google.com/apikey
- **ElevenLabs**: https://elevenlabs.io/app/settings/api-keys
- **Auth0**: https://auth0.com/dashboard → Create "Regular Web Application" → Add `http://localhost:8501/oauth2callback` to Allowed Callback URLs → Add `http://localhost:8501` to Allowed Logout URLs. Add AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET to `.env`. The app auto-generates `.streamlit/secrets.toml` for Streamlit auth.

The app runs without API keys; integrations degrade gracefully.

---

## License
Created by Gantt Meredith, Orator
This work was created for demonstration purposes for Hack Duke 2026. Images and workflows may be used for future demonstration purposes of my greater work with protein stability and the Rutgers Artificial Intelligence and Data Science Collaboratory. 
