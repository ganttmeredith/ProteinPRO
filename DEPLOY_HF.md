# Deploy ProteinPRO on Hugging Face Spaces

## Why local works but Space failed

- **Entry file:** Spaces created from the Streamlit template use `src/streamlit_app.py`. This repo’s app is **`app.py` at the project root**. The Space must run **`app.py`** (set in README frontmatter `app_file` and in Space settings).
- **Dependencies:** Only packages in the Space’s root **`requirements.txt`** are installed. `import yaml` needs **`PyYAML`** / **`pyyaml`** in that file (included in this repo).

## One-time setup

1. **Create or update the Space**
   - [New Space](https://huggingface.co/new-space) → SDK: **Streamlit**, or  
   - Link your GitHub repo: Space **Settings → Repository** → connect `ganttmeredith/ProteinPRO` (or your fork).

2. **Files at repo root (must match GitHub)**
   - `app.py` — main Streamlit app  
   - `requirements.txt`  
   - `README.md` (with YAML frontmatter at top for HF)  
   - `config.yaml`, `assets/`, `src/`, `.streamlit/config.toml` (optional but recommended)

3. **Remove template clutter**  
   If you started from the HF template, delete **`src/streamlit_app.py`** so nothing points to it. Only **`app.py`** should be the app entry.

4. **Variables and secrets** (Space → **Settings → Variables and secrets**)

   | Name | Notes |
   |------|--------|
   | `GEMINI_API_KEY` | Optional; enables Ask AI |
   | `ELEVENLABS_API_KEY` | Optional; TTS |
   | `AUTH0_DOMAIN` | Optional |
   | `AUTH0_CLIENT_ID` | Optional |
   | `AUTH0_CLIENT_SECRET` | Optional |
   | `AUTH0_REDIRECT_URI` | **Required for Auth0 on HF:** e.g. `https://<your-subdomain>.hf.space/oauth2callback` (use your real Space URL) |
   | `AUTH0_COOKIE_SECRET` | Random string for session cookies |

   Hugging Face injects these as environment variables; `python-dotenv` is not required for them on the Space.

5. **Rebuild**  
   After changing `requirements.txt`, push to the repo (or **Factory reboot** the Space) so dependencies reinstall.

## Build issues

- **RDKit / MDAnalysis** are heavy; first build may take 10+ minutes or hit timeouts on free hardware.
- If install fails, check **Build logs** in the Space. You may need a **Docker** Space with a custom `Dockerfile` for system libraries.

## Auth0 callback URLs

In the Auth0 dashboard, add your live Space URL to **Allowed Callback URLs** and **Allowed Logout URLs**, for example:

- `https://YOURNAME-proteinpro.hf.space/oauth2callback`
- `https://YOURNAME-proteinpro.hf.space`

(Replace with the exact URL shown when you open your Space.)
