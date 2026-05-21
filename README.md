# Wealth Quintile Predictor — Streamlit

Ordinal logistic regression (`mord.LogisticAT`) on Cameroon DHS 2018.

## Files in this folder
- `streamlit_app.py` — the app (UI + model loading/training)
- `requirements.txt` — Python dependencies
- `wealth-bg.jpg` — background image (gold currency symbols + chart)
- `CMHR71FL.SAV` — **you must add this** (the DHS 2018 household recode)
- `ordinal_logit.pkl` — created automatically on first run

## Run in VS Code (Windows / macOS / Linux)

1. **Open the folder** in VS Code → `File ▸ Open Folder…` → select this folder.
2. **Add the dataset**: copy your `CMHR71FL.SAV` next to `streamlit_app.py`.
3. **Open a terminal**: `` Ctrl+` `` (or `Terminal ▸ New Terminal`).
4. **Create & activate a virtual environment**:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
6. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```
   Your browser opens at `http://localhost:8501`.

The first launch trains the LogisticAT model from `CMHR71FL.SAV`
(~10–30 s) and caches it as `ordinal_logit.pkl`. Subsequent launches
load instantly.

## Notes
- Python 3.10 – 3.12 recommended.
- If `mord` complains on install, upgrade pip: `pip install -U pip setuptools wheel`.
- To force retraining, delete `ordinal_logit.pkl` and restart.
