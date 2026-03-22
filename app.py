"""
ProteinPRO Streamlit App
Developed 21-22 March 2026 by Gantt Meredith

Deployable web interface for polymer-protein hybrid (PPH) formulation prediction and data analysis.
Automated with Streamlit and deployed to DigitalOcean App Platform.
Run command: streamlit run app.py
"""

import io
import re
import sys
import tempfile
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent))

# Load API keys from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure Auth0 secrets exist (from .env) before Streamlit loads
try:
    import os
    from pathlib import Path
    _root = Path(__file__).parent
    _domain = os.environ.get("AUTH0_DOMAIN", "").strip()
    _cid = os.environ.get("AUTH0_CLIENT_ID", "").strip()
    _csec = os.environ.get("AUTH0_CLIENT_SECRET", "").strip()
    if _domain and _cid and _csec:
        _streamlit_dir = _root / ".streamlit"
        _streamlit_dir.mkdir(exist_ok=True)
        _secrets = _streamlit_dir / "secrets.toml"
        _redirect = os.environ.get("AUTH0_REDIRECT_URI", "http://localhost:8501/oauth2callback")
        _cookie = os.environ.get("AUTH0_COOKIE_SECRET", "proteinpro-auth-cookie-secret-change-in-production")
        _meta = f"https://{_domain}/.well-known/openid-configuration"
        _block = f'[auth]\nredirect_uri = "{_redirect}"\ncookie_secret = "{_cookie}"\n\n[auth.auth0]\nclient_id = "{_cid}"\nclient_secret = "{_csec}"\nserver_metadata_url = "{_meta}"\n'
        if not _secrets.exists() or _secrets.read_text() != _block:
            _secrets.write_text(_block)
except Exception:
    pass

import numpy as np
import streamlit as st
import pandas as pd
import yaml

# 3D viewer
try:
    import py3Dmol
    HAS_3D = True
except ImportError:
    HAS_3D = False

from src.pdb_handler import (
    fetch_pdb,
    parse_structure,
    featurize_protein,
    get_sequence_and_features,
    get_coordinates_for_visualization,
    get_residue_roles_for_visualization,
    load_config,
)
from src.monomer_featurizer import featurize_all_monomers, composition_to_polymer_features, load_monomers
from src.stability_model import StabilityPredictor, sample_design_space, MODEL_TYPES
try:
    from src.gpr_predictor import GPRStabilityPredictor
    GPR_AVAILABLE = True
except ImportError:
    GPRStabilityPredictor = None
    GPR_AVAILABLE = False
try:
    from src.integrations.gemini_api import ask_formulation_advice
except ImportError:
    def ask_formulation_advice(*a, **k):
        return "Add GEMINI_API_KEY and install: pip install google-generativeai"
try:
    from src.integrations.elevenlabs_tts import text_to_speech_audio, is_available as elevenlabs_available
except ImportError:
    text_to_speech_audio = lambda *a, **k: None
    elevenlabs_available = lambda: False
try:
    from src.integrations.solana_verify import formulation_hash
except ImportError:
    formulation_hash = lambda p, c, s: "N/A"
try:
    from src.integrations.auth0_config import is_available as auth0_available, is_logged_in as auth_is_logged_in, get_user_id as auth_get_user_id
except ImportError:
    auth0_available = lambda: False
    auth_is_logged_in = lambda: False
    auth_get_user_id = lambda: None
try:
    from src.stability_data_analysis import (
        read_round_file,
        run_analysis,
    )
except ImportError:
    read_round_file = None
    run_analysis = None
try:
    from src.user_pdb_cache import (
        save_fetched_to_user_cache,
        save_upload_to_user_cache,
        list_user_cached,
        load_from_user_cache,
    )
except ImportError:
    save_fetched_to_user_cache = lambda u, p, s: s
    save_upload_to_user_cache = lambda u, f, b: None
    list_user_cached = lambda u: []
    load_from_user_cache = lambda u, n: None

st.set_page_config(page_title="ProteinPRO", page_icon="assets/logo.png", layout="wide")

# Global styling for dark theme
st.markdown("""
<style>
    /* Softer block container edges */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    /* Headers with accent color (light purple on dark) */
    h1, h2, h3 { color: #a78bfa !important; }
    /* Divider styling */
    hr { border-color: rgba(139, 122, 184, 0.3) !important; }
    /* Info boxes */
    .stAlert { border-radius: 8px; }
    /* Metric cards */
    [data-testid="stMetricValue"] { color: #8b7ab8 !important; }
</style>
""", unsafe_allow_html=True)

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

# Sidebar config
if LOGO_PATH.exists():
    _sb_b64 = __import__("base64").b64encode(LOGO_PATH.read_bytes()).decode()
    st.sidebar.markdown(f'<img src="data:image/png;base64,{_sb_b64}" style="width:50px;opacity:0.95;margin-bottom:4px;"/>', unsafe_allow_html=True)
else:
    st.sidebar.image("assets/logo.png", width=50)
st.sidebar.divider()

# Input mode
input_options = ["PDB ID", "Upload file"]
if auth_is_logged_in():
    input_options.insert(0, "From saved")
input_mode = st.sidebar.radio(
    "Protein input",
    input_options,
    help="Retrieve from RCSB, upload, or load from your saved structures",
)

protein_source = None
pdb_id = None

if input_mode == "From saved" and auth_is_logged_in():
    user_id = auth_get_user_id()
    saved = list_user_cached(user_id)
    if saved:
        chosen = st.sidebar.selectbox("Your saved structures", options=[n for n, _ in saved], format_func=lambda x: x)
        if chosen:
            protein_source = load_from_user_cache(user_id, chosen)
            pdb_id = chosen
    else:
        st.sidebar.info("Hey, there. No saved structures yet. Request or upload a PDB or CIF to save it.")
elif input_mode == "PDB ID":
    pdb_id = st.sidebar.text_input("PDB ID", value="1LYZ", max_chars=10)
    if pdb_id:
        try:
            path = fetch_pdb(pdb_id)
            protein_source = path
            if auth_is_logged_in():
                save_fetched_to_user_cache(auth_get_user_id(), pdb_id, path)
        except Exception as e:
            st.sidebar.error(f"Request failed: {e}")
else:
    uploaded = st.sidebar.file_uploader("Upload PDB or CIF", type=["pdb", "cif"])
    if uploaded:
        data = uploaded.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as f:
            f.write(data)
            protein_source = f.name
        pdb_id = Path(uploaded.name).stem
        if auth_is_logged_in():
            save_upload_to_user_cache(auth_get_user_id(), uploaded.name, data)

# Config
config = load_config()
monomers = load_monomers()
monomer_names = list(monomers.keys())
MAX_MONOMERS = 4

# Model selector
_model_labels = [f"{name} ({k})" for k, (name, _, _) in MODEL_TYPES.items()]
_model_keys = list(MODEL_TYPES.keys())
if GPR_AVAILABLE:
    _model_labels.append("GPR (with uncertainty)")
    _model_keys.append("gpr")
model_choice_idx = st.sidebar.selectbox(
    "Prediction model",
    range(len(_model_labels)),
    format_func=lambda i: _model_labels[i],
    help="RF | SVM | Ridge | Logistic | Gradient Boosting | KNN | GPR",
)
model_key = _model_keys[model_choice_idx]
use_gpr = model_key == "gpr"

def get_predictor():
    if use_gpr:
        return GPRStabilityPredictor()
    return StabilityPredictor(use_surrogate=True, model_type=model_key)

# Auth0: trigger login via query param (for gradient link-button)
if auth0_available() and st.query_params.get("login") == "auth0":
    st.login("auth0")
    st.stop()

# Auth0: status in sidebar when logged in
if auth0_available() and hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
    st.sidebar.caption(f"Signed in as {getattr(st.user, 'name', getattr(st.user, 'email', ''))}")

# Persist composition in session state (updated in Structure tab)
if "composition" not in st.session_state:
    st.session_state.composition = {name: 0.25 if i < 4 else 0.0 for i, name in enumerate(monomer_names)}
    total = 1.0
    st.session_state.composition = {k: v for k, v in st.session_state.composition.items() if v > 0}
    if st.session_state.composition:
        t = sum(st.session_state.composition.values())
        st.session_state.composition = {k: v/t for k, v in st.session_state.composition.items()}

# Main content - header row (compact: no tall widget to avoid blank space)
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    logo_col, title_col = st.columns([1, 4])
    with logo_col:
        if LOGO_PATH.exists():
            _logo_b64 = __import__("base64").b64encode(LOGO_PATH.read_bytes()).decode()
            st.markdown(f"""
            <style>
            @keyframes logo-rotate {{
                0% {{ transform: rotate(0deg); filter: drop-shadow(0 0 20px rgba(255, 80, 120, 0.5)) drop-shadow(0 0 6px rgba(255, 200, 220, 0.8)); }}
                12.5% {{ transform: rotate(45deg); filter: drop-shadow(0 0 20px rgba(255, 150, 80, 0.5)) drop-shadow(0 0 6px rgba(255, 220, 180, 0.8)); }}
                25% {{ transform: rotate(90deg); filter: drop-shadow(0 0 20px rgba(255, 220, 80, 0.5)) drop-shadow(0 0 6px rgba(255, 248, 200, 0.8)); }}
                37.5% {{ transform: rotate(135deg); filter: drop-shadow(0 0 20px rgba(120, 255, 100, 0.5)) drop-shadow(0 0 6px rgba(180, 255, 200, 0.8)); }}
                50% {{ transform: rotate(180deg); filter: drop-shadow(0 0 20px rgba(80, 220, 255, 0.5)) drop-shadow(0 0 6px rgba(180, 240, 255, 0.8)); }}
                62.5% {{ transform: rotate(225deg); filter: drop-shadow(0 0 20px rgba(80, 120, 255, 0.5)) drop-shadow(0 0 6px rgba(180, 200, 255, 0.8)); }}
                75% {{ transform: rotate(270deg); filter: drop-shadow(0 0 20px rgba(160, 80, 255, 0.5)) drop-shadow(0 0 6px rgba(220, 180, 255, 0.8)); }}
                87.5% {{ transform: rotate(315deg); filter: drop-shadow(0 0 20px rgba(255, 80, 180, 0.5)) drop-shadow(0 0 6px rgba(255, 180, 220, 0.8)); }}
                100% {{ transform: rotate(360deg); filter: drop-shadow(0 0 20px rgba(255, 80, 120, 0.5)) drop-shadow(0 0 6px rgba(255, 200, 220, 0.8)); }}
            }}
            .proteinpro-logo {{
                width: 80px; display: block;
                animation: logo-rotate 10s linear infinite;
            }}
            .proteinpro-logo:hover {{
                animation: none;
                transform: scale(1.08); filter: drop-shadow(0 0 24px rgba(255, 200, 255, 0.6)) drop-shadow(0 0 10px rgba(255, 255, 255, 0.7));
            }}
            </style>
            <img src="data:image/png;base64,{_logo_b64}" class="proteinpro-logo" alt="ProteinPRO"/>
            """, unsafe_allow_html=True)
        else:
            st.image("assets/logo.png", width=80)
    with title_col:
        st.title("ProteinPRO")
        st.markdown("**Mapping Protein Chemistry to Polymer Chemistry**")
with header_col2:
    st.markdown('<div style="height: 36px;"></div>', unsafe_allow_html=True)
    if auth0_available():
        try:
            logged_in = getattr(st.user, "is_logged_in", False) if hasattr(st, "user") else False
            if logged_in:
                st.caption(f"Signed in as {getattr(st.user, 'name', getattr(st.user, 'email', 'User'))}")
                if st.button("Log out", key="auth_logout", type="primary"):
                    st.logout()
            else:
                st.markdown("""
                <style>
                @keyframes auth0-breathe {
                    0%, 100% { transform: scale(1); box-shadow: 0 2px 12px rgba(110, 84, 148, 0.35); }
                    50% { transform: scale(1.04); box-shadow: 0 4px 24px rgba(110, 84, 148, 0.55), 0 0 20px rgba(139, 122, 184, 0.25); }
                }
                .auth0-btn {
                    display: inline-block; padding: 10px 20px; margin-bottom: 8px;
                    background: linear-gradient(135deg, #6e5494 0%, #8b7ab8 100%);
                    color: #fafafa !important; text-decoration: none;
                    border-radius: 50px; font-weight: 600; font-size: 14px;
                    text-align: center; border: 1px solid rgba(139, 122, 184, 0.5);
                    animation: auth0-breathe 2.5s ease-in-out infinite;
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                .auth0-btn:hover {
                    animation: none;
                    transform: scale(1.06); box-shadow: 0 6px 28px rgba(110, 84, 148, 0.6);
                    color: #fafafa !important;
                }
                </style>
                <div style="text-align: right;"><a href="?login=auth0" class="auth0-btn">Auth0 Login</a></div>
                """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <style>
            @keyframes auth0-breathe{0%,100%{transform:scale(1);box-shadow:0 2px 12px rgba(110,84,148,0.35);}50%{transform:scale(1.04);box-shadow:0 4px 24px rgba(110,84,148,0.55);}}
            .auth0-btn{display:inline-block;padding:10px 20px;margin-bottom:8px;background:linear-gradient(135deg,#6e5494,#8b7ab8);color:#fafafa!important;text-decoration:none;border-radius:50px;font-weight:600;font-size:14px;border:1px solid rgba(139,122,184,0.5);animation:auth0-breathe 2.5s ease-in-out infinite;}
            .auth0-btn:hover{animation:none;transform:scale(1.06);}
            </style>
            <div style="text-align: right;"><a href="?login=auth0" class="auth0-btn">Auth0 Login</a></div>
            """, unsafe_allow_html=True)

# Hero content block: immediately below header, centered (no nested container)
st.markdown("""
    <div class="hero-block">
        <p class="hero-mission">ProteinPRO predicts protein–polymer hybrid (PPH) formulation stability by matching chemical features from your protein structure to PET-RAFT monomer composition chemical features.</p>
        <div class="hero-how">
            <span class="hero-how-label">How it works</span>
            <div class="hero-steps">
                <div class="hero-step" data-step="1"><span class="hero-icon">🧬</span><span>Load protein</span></div>
                <div class="hero-step" data-step="2"><span class="hero-icon">🔍</span><span>Explore protein features</span></div>
                <div class="hero-step" data-step="3"><span class="hero-icon">⚗️</span><span>Select monomer compositions</span></div>
                <div class="hero-step" data-step="4"><span class="hero-icon">📊</span><span>Predict stability</span></div>
                <div class="hero-step" data-step="5"><span class="hero-icon">🔄</span><span>Optimize and iterate</span></div>
            </div>
        </div>
        <div class="hero-powered">
            <span class="hero-powered-label">Powered by</span>
            <div class="hero-tech-marquee"><div class="hero-tech-inner"><span class="hero-tech-track">PDB · Biopython · RDKit · scikit-learn · py3Dmol · Streamlit · Gemini · ElevenLabs · Auth0</span><span class="hero-tech-track">PDB · Biopython · RDKit · scikit-learn · py3Dmol · Streamlit · Gemini · ElevenLabs · Auth0</span></div></div>
        </div>
    </div>
    <style>
    .hero-block { text-align: center; padding: 12px 16px 8px; max-width: 720px; margin: 0 auto; }
    .hero-mission { font-size: 0.95rem; color: #94a3b8; line-height: 1.6; margin: 0 0 16px 0; }
    .hero-how, .hero-powered { margin-top: 12px; }
    .hero-how-label, .hero-powered-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; display: block; margin-bottom: 10px; }
    .hero-steps { display: flex; flex-wrap: wrap; justify-content: center; gap: 12px 20px; }
    .hero-step { display: flex; align-items: center; gap: 6px; padding: 8px 14px; border-radius: 8px; font-size: 0.8rem; color: #94a3b8; background: rgba(139, 122, 184, 0.08); border: 1px solid rgba(139, 122, 184, 0.15); transition: all 0.3s ease; }
    .hero-step .hero-icon { font-size: 1rem; }
    .hero-steps .hero-step:nth-child(1) { animation: hero-pulse 5s ease-in-out infinite; }
    .hero-steps .hero-step:nth-child(2) { animation: hero-pulse 5s ease-in-out infinite 1s; }
    .hero-steps .hero-step:nth-child(3) { animation: hero-pulse 5s ease-in-out infinite 2s; }
    .hero-steps .hero-step:nth-child(4) { animation: hero-pulse 5s ease-in-out infinite 3s; }
    .hero-steps .hero-step:nth-child(5) { animation: hero-pulse 5s ease-in-out infinite 4s; }
    @keyframes hero-pulse { 0%, 18%, 100% { opacity: 0.7; border-color: rgba(139, 122, 184, 0.15); background: rgba(139, 122, 184, 0.08); } 8% { opacity: 1; border-color: rgba(167, 139, 250, 0.4); background: rgba(139, 122, 184, 0.18); box-shadow: 0 0 12px rgba(110, 84, 148, 0.2); } }
    .hero-tech-marquee { overflow: hidden; user-select: none; padding: 8px 0; }
    .hero-tech-inner { display: flex; animation: hero-scroll 20s linear infinite; width: max-content; }
    .hero-tech-track { flex-shrink: 0; padding: 0 24px; font-size: 0.8rem; color: #64748b; }
    @keyframes hero-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    @media (max-width: 640px) { .hero-steps { flex-direction: column; align-items: center; } .hero-block { padding: 16px 12px; } }
    </style>
    """, unsafe_allow_html=True)


# Main mode: Protein Analysis vs Custom Data Analysis
main_tab_protein, main_tab_data = st.tabs(["Protein Analysis", "Custom Data Analysis"])

with main_tab_protein:
    if protein_source:
        # Parse once for all tabs
        try:
            structure = parse_structure(protein_source, pdb_id)
            info = get_sequence_and_features(structure)
        except Exception as e:
            st.error(f"Failed to parse structure: {e}")
            structure = None
            info = {}

        tab_features, tab_structure, tab3, tab5 = st.tabs(
            ["Features", "Structure", "Prediction", "Ask Gemini"]
        )

        with tab_features:
            st.subheader("Protein features & chemical landscape")
            pf = featurize_protein(protein_source, pdb_id)
            col_viewer, col_features = st.columns([1.2, 1])

            with col_viewer:
                if HAS_3D and structure:
                    try:
                        pdb_str = get_coordinates_for_visualization(structure)
                        roles = get_residue_roles_for_visualization(structure)
                        view = py3Dmol.view(width=500, height=380)
                        view.addModel(pdb_str, "pdb")
                        view.setStyle({"cartoon": {"color": "#b0b0b0", "opacity": 0.85}})
                        _COLORS = {"polar": "#6e5494", "positive": "#3b82f6", "negative": "#ef4444", "hydrophobic": "#f97316"}
                        for rtype, color in _COLORS.items():
                            pairs = roles.get(rtype, [])
                            if not pairs:
                                continue
                            by_chain = {}
                            for ch, resi in pairs:
                                by_chain.setdefault(ch, []).append(resi)
                            for ch, resis in by_chain.items():
                                view.setStyle({"chain": ch, "resi": resis}, {"cartoon": {"color": color, "thickness": 1.2}})
                        view.zoomTo()
                        view.spin(True)
                        st.components.v1.html(view.write_html(), height=400)
                        st.caption("🟣 Polar · 🔵 Positive · 🔴 Negative · 🟠 Hydrophobic")
                    except Exception as e:
                        st.warning(f"3D view error: {e}")
                elif not HAS_3D:
                    st.info("Install py3Dmol for 3D visualization: pip install py3Dmol")
                st.markdown('<div style="min-height: 200px;"></div>', unsafe_allow_html=True)
                _convai_html = """<!DOCTYPE html><html><head><script src="https://unpkg.com/@elevenlabs/convai-widget-embed" async type="text/javascript"></script></head><body style="margin:0;padding:0;background:transparent;min-height:500px;"><elevenlabs-convai agent-id="agent_6001km9er5c3em99vsjb4x5fgw33" variant="compact" action-text="Chat with Gemini"></elevenlabs-convai></body></html>"""
                st.components.v1.html(_convai_html, height=500, scrolling=True)

            with col_features:
                n = pf.get("n_residues", 0)
                fracs = {
                    "Polar": pf.get("fraction_polar", 0),
                    "Positive": pf.get("fraction_positive", 0),
                    "Negative": pf.get("fraction_negative", 0),
                    "Hydrophobic": pf.get("fraction_hydrophobic", 0),
                }
                _bar_colors = {
                    "Polar": ("#8b7ab8", "#6e5494"),
                    "Positive": ("#5b9bd5", "#3b82f6"),
                    "Negative": ("#e57373", "#c62828"),
                    "Hydrophobic": ("#ffb74d", "#f57c00"),
                }
                _bars_html = "".join(
                    f'<div class="res-bar"><span class="res-label">{label}</span>'
                    f'<div class="res-track"><div class="res-fill" style="width:{int(round(val*100))}%;background:linear-gradient(90deg,{_bar_colors[label][0]},{_bar_colors[label][1]});"></div></div>'
                    f'<span class="res-pct">{int(round(val*100))}%</span></div>'
                    for label, val in fracs.items()
                )
                st.markdown(f"""
                <style>
                .res-composition {{ margin:0; padding:0; font-family:system-ui,sans-serif; }}
                .res-composition-title {{ font-weight:600; font-size:14px; color:#e0e0e0; margin-bottom:10px; }}
                .res-bars {{ display:flex; flex-direction:column; gap:8px; }}
                .res-bar {{ display:flex; align-items:center; gap:10px; min-height:28px; }}
                .res-label {{ flex:0 0 85px; font-size:12px; color:#b0b0b0; font-weight:500; }}
                .res-track {{ flex:1; min-width:0; height:12px; background:rgba(255,255,255,0.06); border-radius:6px; overflow:hidden; box-shadow:inset 0 1px 2px rgba(0,0,0,0.2); border:1px solid rgba(255,255,255,0.04); }}
                .res-fill {{ height:100%; border-radius:5px; transition:width 0.4s ease; box-shadow:0 0 8px rgba(0,0,0,0.15); }}
                .res-pct {{ flex:0 0 38px; font-size:12px; color:#9ca3af; text-align:right; font-variant-numeric:tabular-nums; }}
                @media (max-width: 640px) {{ .res-label {{ flex:0 0 70px; font-size:11px; }} .res-pct {{ flex:0 0 32px; }} }}
                </style>
                <div class="res-composition">
                <div class="res-composition-title">Residue composition</div>
                <div class="res-bars">{_bars_html}</div>
                </div>
                """, unsafe_allow_html=True)
                st.divider()
                st.markdown("**Key Descriptors**")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Residues", n)
                    h = pf.get("mean_hydrophobicity", 0)
                    st.metric("Hydrophobicity", f"{h:.2f}")
                with m2:
                    q = pf.get("net_charge_density", 0)
                    st.metric("Net charge density", f"{q:.3f}")
                    st.metric("Std hydrophobicity", f"{pf.get('std_hydrophobicity', 0):.2f}")

                st.divider()
                st.markdown("**Suggested Monomers**")
                suggestions = []
                if pf.get("fraction_positive", 0) > 0.15:
                    suggestions.append(("Anionic SPMA", "Balances positive charge for favorable interactions", "anionic", "#06b6d4"))
                if pf.get("fraction_negative", 0) > 0.15:
                    suggestions.append(("Cationic TMAEMA / DEAEMA", "Balances negative charge", "cationic", "#3b82f6"))
                if pf.get("fraction_hydrophobic", 0) > 0.35:
                    suggestions.append(("BMA or EHMA", "Match hydrophobic patches", "hydrophobic", "#f97316"))
                if pf.get("fraction_polar", 0) > 0.2:
                    suggestions.append(("HPMA or PEGMA", "Complement polar regions", "neutral_hydrophilic", "#22c55e"))
                if pf.get("mean_hydrophobicity", 0) > 0.5:
                    suggestions.append(("Hydrophobic blend", "Add BMA/EHMA for compatibility", "hydrophobic", "#f97316"))
                if not suggestions:
                    suggestions.append(("HPMA + DEAEMA", "Versatile starting point for most proteins", "neutral_hydrophilic", "#22c55e"))
                for name, reason, cat, col in suggestions[:4]:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{col}22,{col}08);border-left:4px solid {col};padding:10px 12px;margin-bottom:8px;border-radius:6px;font-size:13px;">
                    <strong>{name}</strong><br><span style="color:#6b7280;">{reason}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_structure:
            col_struct, col_monomer = st.columns([1.1, 0.9])

            with col_struct:
                st.subheader("3D structure")
                if HAS_3D and structure:
                    try:
                        pdb_str = get_coordinates_for_visualization(structure)
                        view = py3Dmol.view(width=600, height=420)
                        view.addModel(pdb_str, "pdb")
                        view.setStyle({"cartoon": {"color": "spectrum"}})
                        view.zoomTo()
                        view.spin(True)
                        st.components.v1.html(view.write_html(), height=440)
                    except Exception as e:
                        st.warning(f"3D view error: {e}")
                elif not HAS_3D:
                    st.info("Install py3Dmol for 3D visualization: pip install py3Dmol")

                if info:
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Residues", info["n_residues"])
                    with m2:
                        st.metric("Hydrophobicity", f"{info['mean_hydrophobicity']:.2f}")
                    with m3:
                        st.metric("Charge Density", f"{info['net_charge_density']:.3f}")

            with col_monomer:
                st.subheader("Monomer Composition")
                st.caption(f"Select up to {MAX_MONOMERS} monomers and set molar fractions. The sum of the molar fractions should be 1.")

                selected = st.multiselect(
                    "Monomers",
                    options=monomer_names,
                    default=list(st.session_state.composition.keys())[:MAX_MONOMERS] if st.session_state.composition else monomer_names[:2],
                    max_selections=MAX_MONOMERS,
                    key="monomer_multiselect",
                )

                if len(selected) > MAX_MONOMERS:
                    selected = selected[:MAX_MONOMERS]
                    st.warning(f"Limited to {MAX_MONOMERS} monomers.")

                composition = {}
                if selected:
                    n = len(selected)
                    default_frac = 1.0 / n
                    fracs = {}
                    for i, name in enumerate(selected):
                        fracs[name] = st.slider(
                            name,
                            min_value=0.05,
                            max_value=1.0,
                            value=float(st.session_state.composition.get(name, default_frac)),
                            step=0.05,
                            key=f"frac_{name}",
                        )
                    total = sum(fracs.values())
                    if total > 0:
                        composition = {k: v / total for k, v in fracs.items()}
                    st.session_state.composition = composition
                    st.metric("Total", f"{total:.2f}" + (" (normalized)" if abs(total - 1.0) > 0.01 else ""))

                    # Polymer descriptors (weighted)
                    st.divider()
                    st.subheader("Polymer Descriptors (weighted)")
                    st.caption("Chemical properties of your monomer blend, averaged by molar fraction. These chemical properties are featurized and used as training data for a stability model to predict protein–polymer compatibility.")
                    with st.expander("What do these descriptors mean?"):
                        st.markdown("""
                        **Core Descriptors (from RDKit):**
                        - **MolWt** — Molecular weight (g/mol). Larger polymers can affect diffusion and binding.
                        - **LogP** — Partition coefficient (lipophilicity). High = hydrophobic; low = hydrophilic. Affects how well the polymer matches protein surface chemistry and potential future binding.
                        - **TPSA** — Topological polar surface area (Å²). Measures polarity and hydrogen-bonding potential.
                        - **NumHDonors** / **NumHAcceptors** — Ratio of H-bond donors and acceptors. Important for polar interactions with relevant protein side chains.
                        - **FractionCSP3** — Fraction of carbons that are sp³ (saturated). Higher value = more flexible.

                        **Why "weighted"?** Your polymer is a mixture of 1-4 monomers. Each descriptor is defined by a molar-fraction-weighted average. For example, 50% HPMA with 50% BMA gives a LogP comprised of a weighted .50 for HPMA and .50 for BMA.
                        """)
                    mf_df = featurize_all_monomers()
                    poly_f = composition_to_polymer_features(composition, mf_df)
                    _DESC_LABELS = {
                        "MolWt": "Molecular Weight (g/mol)",
                        "LogP": "Log Partition Coefficient (lipophilicity)",
                        "TPSA": "Total Polar Surface Area (Å²)",
                        "NumHDonors": "H-bond Donors",
                        "NumHAcceptors": "H-bond Acceptors",
                        "FractionCSP3": "Carbon Saturation (CSP3)",
                    }
                    main_desc = {k: v for k, v in poly_f.items() if k in _DESC_LABELS and isinstance(v, (int, float))}
                    if main_desc:
                        for k, v in main_desc.items():
                            label = _DESC_LABELS.get(k, k)
                            st.metric(label, f"{v:.2f}" if isinstance(v, float) else v)
                    with st.expander("All descriptors (raw)"):
                        st.json(poly_f)
                else:
                    st.info("Select at least one monomer.")

        with tab3:
            st.subheader("Stability prediction")
            active_comp = {k: v for k, v in st.session_state.composition.items() if v > 0}
            if active_comp:
                predictor = get_predictor()
                score, details = predictor.predict(protein_source, active_comp, pdb_id)
                if use_gpr and "uncertainty_scaled" in details:
                    unc = details["uncertainty_scaled"]
                    st.metric("Stability score", f"{score:.4f} ± {unc:.3f}")
                else:
                    st.metric("Stability score", f"{score:.4f}")
                st.caption("Higher = more favorable (surrogate model)")
                with st.expander("Prediction details"):
                    st.markdown("**Score equations**")
                    if use_gpr:
                        st.latex(r"\text{mean}, \sigma = \text{GPR}(\mathbf{x}_{\text{scaled}}) \quad \text{(Matern kernel)}")
                    else:
                        _m = {"rf": "RF", "svr": "SVR", "ridge": "Ridge", "logistic": "Logistic", "gradient_boosting": "GB", "knn": "KNN"}.get(model_key, "Model")
                        st.latex(r"\text{raw\_score} = \text{" + _m + r"}(\mathbf{x}_{\text{scaled}})")
                    st.latex(r"\text{score} = \frac{\tanh(\text{raw\_score}/50) + 1}{2} \quad \in [0, 1]")
                    st.markdown("*Surrogate objectives:*")
                    st.latex(r"0.3\,(1 - |H_{\text{prot}} - H_{\text{poly}}|) + 0.3\,(1 - |q_{\text{net}}|) + 0.2\,\text{polarity}")
                    st.markdown("---")
                    st.markdown("**Computed values**")
                    st.json({k: v for k, v in details.items() if k != "hydrophobicity_profile" and k != "charge_profile"})
            else:
                st.warning("Select at least one monomer")

            st.divider()
            st.subheader("Monomer combinations to explore")
            st.caption("Sample and rank formulation compositions for optimal protein–polymer stability.")
            n_samples = st.slider("Number of formulations to sample", 10, 200, 50, key="prediction_design_n")
            if st.button("Rank formulations", key="prediction_rank_btn"):
                predictor = get_predictor()
                df = predictor.rank_formulations(protein_source, n_candidates=n_samples, pdb_id=pdb_id)
                display_cols = ["composition", "stability_score"]
                if "uncertainty" in df.columns:
                    display_cols.append("uncertainty")
                st.dataframe(
                    df[display_cols].head(20),
                    use_container_width=True,
                )
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button("Download full results (CSV)", buf.getvalue(), file_name="formulation_rankings.csv", mime="text/csv", key="dl_prediction_rankings")

        with tab5:
            st.subheader("Ask RAG-enabled AI (Gemini)")
            st.caption("Best Use of Generative AI - Hack Duke 2026")
            active_comp = {k: v for k, v in st.session_state.composition.items() if v > 0}
            if active_comp:
                pf = featurize_protein(protein_source, pdb_id)
                predictor = get_predictor()
                score, _ = predictor.predict(protein_source, active_comp, pdb_id)
                summary = f"{pf['n_residues']} residues, hydrophobicity {pf['mean_hydrophobicity']:.2f}, charge {pf['net_charge_density']:.2f}"
                question = st.text_input("Ask me about formulation optimization", placeholder="e.g. If I were to try to optimize stability of lipase upon thermal heating, which monomers should I turn to first??")
                if st.button("Get AI insight via an informed RAG pipeline"):
                    if question:
                        with st.spinner("Querying Gemini..."):
                            answer = ask_formulation_advice(summary, active_comp, score, question)
                        st.write(answer)
                    else:
                        st.warning("Enter a question")
            else:
                st.info("Select monomers in the Structure tab first.")
    else:
        st.info("Enter a PDB ID or upload a structure (PDB/CIF) to begin.")

with main_tab_data:
    st.subheader("Custom Data Analysis")
    if read_round_file is None or run_analysis is None:
        st.warning("Analysis module not available. Install openpyxl: pip install openpyxl")
    else:
        st.markdown("**Upload Stability Data (Excel, please!)**")
        st.caption(
            "Upload one or more Excel files from previous polymerization design rounds. "
            "Expects columns: performance (Average_REA_across_days or similar), optional monomers (DEAEMA, HPMA, etc.), Degree of Polymerization. "
            "Multiple rounds of data can be uploaded and analyzed concurrently; simply label the columns by round number."
        )
        uploaded_files = st.file_uploader(
            "Choose Excel file(s)",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key="stability_data_upload",
        )
        if uploaded_files:
            all_dfs = []
            errors = []
            for uf in uploaded_files:
                try:
                    df = read_round_file(uf.read(), uf.name)
                    all_dfs.append(df)
                    st.success(f"Loaded: {uf.name} ({len(df)} rows)")
                except Exception as e:
                    errors.append(f"{uf.name}: {e}")
            if errors:
                for err in errors:
                    st.error(err)
            if all_dfs:
                data = pd.concat(all_dfs, ignore_index=True).sort_values("Round").reset_index(drop=True)
                st.caption("Combined data (first 50 rows)")
                st.dataframe(data.head(50), use_container_width=True)
                if st.button("Run analysis", key="run_stability_analysis"):
                    with st.spinner("Running pipeline... vite vite!"):
                        try:
                            summary, figures = run_analysis(data)
                            st.session_state["custom_analysis_summary"] = summary
                            st.session_state["custom_analysis_figures"] = figures
                            st.session_state["custom_analysis_data"] = data
                            st.success("Hey, your analysis iscomplete! View your results below under Design Space Exploration.")
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")

    st.divider()
    st.subheader("Design Space Exploration")
    st.caption("Analysis results from your uploaded stability data.")
    if "custom_analysis_summary" in st.session_state and "custom_analysis_figures" in st.session_state:
        summary = st.session_state["custom_analysis_summary"]
        figures = st.session_state["custom_analysis_figures"]
        st.markdown("**Summary Metrics**")
        st.dataframe(summary, use_container_width=True)
        st.markdown("**Figures**")
        n_figs = len(figures)
        if n_figs > 0:
            for i in range(0, n_figs, 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < n_figs:
                        with col:
                            title, png_bytes = figures[idx]
                            st.markdown(f"**{title}**")
                            st.image(png_bytes, use_container_width=True)
            if st.session_state.get("custom_analysis_data") is not None:
                import zipfile
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("round_summary.csv", summary.to_csv(index=False))
                    for title, png_bytes in figures:
                        safe_name = re.sub(r"[^\w\-]", "_", title) + ".png"
                        zf.writestr(safe_name, png_bytes)
                buf.seek(0)
                st.download_button("Download results (ZIP)", buf.getvalue(), file_name="stability_analysis.zip", mime="application/zip", key="dl_stability_zip")
    else:
        st.info("Upload Excel file(s) above and click **Run analysis** to see results here.")

# Footer - pushed down so it's not always in view
st.markdown("""
<style>
.footer-container {
    margin-top: 320px;
    padding: 32px 24px;
    background: linear-gradient(180deg, rgba(110, 84, 148, 0.12) 0%, rgba(110, 84, 148, 0.04) 100%);
    border-top: 1px solid rgba(139, 122, 184, 0.25);
    border-radius: 12px 12px 0 0;
}
.footer-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 4px;
}
.footer-creator {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 20px;
}
.footer-creator a {
    color: #8b7ab8;
    text-decoration: none;
    font-weight: 500;
}
.footer-creator a:hover {
    text-decoration: underline;
    color: #a78bfa;
}
.footer-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-top: 16px;
}
.footer-badges a {
    display: inline-block;
    transition: transform 0.2s;
}
.footer-badges a:hover {
    transform: translateY(-2px);
}
.footer-event {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 20px;
}
</style>
<div class="footer-container">
    <div class="footer-title">ProteinPRO</div>
    <div class="footer-creator">Created by <a href="https://linkedin.com/in/ganttmeredith" target="_blank">@ganttmeredith</a></div>
    <div class="footer-badges">
        <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub" style="height:24px;" title="GitHub"/>
        <a href="https://streamlit.io" target="_blank" title="Streamlit">
            <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" style="height:24px;"/>
        </a>
        <a href="https://ai.google.dev/gemini-api" target="_blank" title="Gemini AI">
            <img src="https://img.shields.io/badge/Gemini%20AI-4285F4?style=flat-square&logo=google&logoColor=white" alt="Gemini AI" style="height:24px;"/>
        </a>
        <a href="https://elevenlabs.io" target="_blank" title="ElevenLabs">
            <img src="https://img.shields.io/badge/ElevenLabs-000000?style=flat-square&logo=elevenlabs&logoColor=white" alt="ElevenLabs" style="height:24px;"/>
        </a>
        <a href="https://www.digitalocean.com" target="_blank" title="DigitalOcean">
            <img src="https://img.shields.io/badge/DigitalOcean-0080FF?style=flat-square&logo=digitalocean&logoColor=white" alt="DigitalOcean" style="height:24px;"/>
        </a>
        <a href="https://auth0.com" target="_blank" title="Auth0">
            <img src="https://img.shields.io/badge/Auth0-EB5424?style=flat-square&logo=auth0&logoColor=white" alt="Auth0" style="height:24px;"/>
        </a>
        <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" style="height:24px;" title="Python"/>
        <img src="https://img.shields.io/badge/RDKit-Chemical%20Featurization-4B5563?style=flat-square" alt="RDKit" style="height:24px;" title="RDKit"/>
        <img src="https://img.shields.io/badge/Biopython-2D2A2E?style=flat-square" alt="Biopython" style="height:24px;" title="Biopython"/>
    </div>
    <div class="footer-event">Hack Duke 2026 — Code for Good — Gantt Meredith, Orator</div>
</div>
""", unsafe_allow_html=True)
