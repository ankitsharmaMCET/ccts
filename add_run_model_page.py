# add_run_model_page.py — creates ccts_dashboard_full/pages/0_Run_Model.py
# and inserts a sidebar link in app.py

from pathlib import Path
import re

ROOT = Path("ccts_dashboard_full")
PAGES = ROOT / "pages"
PAGES.mkdir(parents=True, exist_ok=True)

PAGE_CODE = r'''
import os, sys, subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Run Model • CCTS", page_icon="🚀", layout="wide")
st.title("🚀 Run Model")
st.markdown("Use this page to run your **ccts** simulation and load the results into the dashboard.")

with st.expander("Environment", expanded=True):
    model_root = st.text_input("Model project root", value="E:\\model_2", help="Folder that contains the ccts package (e.g., E:\\model_2\\ccts).")
    python_exec = st.text_input("Python executable", value=sys.executable, help="Interpreter to use for the run.")
    run_mode = st.radio("Run method", ["module (-m ccts.run_simulation)", "script (ccts\\run_simulation.py)"], index=0)

with st.expander("Simulation inputs", expanded=True):
    firms_csv = st.text_input("Firms CSV path", value="E:\\model_2\\ccts_inputs\\firms\\firms_india_2025.csv")
    scenario = st.selectbox("Scenario", [
        "baseline","high_ambition","low_liquidity","no_banking",
        "volatile_market","cooperative","technology_push","stress_test"
    ], index=0)
    steps = st.number_input("Steps", min_value=1, value=24, step=1)
    seed  = st.number_input("Seed (optional, 0 = unset)", min_value=0, value=42, step=1)
    log_level = st.selectbox("Log level", ["INFO","DEBUG","WARNING","ERROR"], index=0)

with st.expander("Output options", expanded=True):
    out_base = st.text_input("Base output folder", value=str(Path(model_root) / "output_runs"))
    excel_name = st.text_input("Excel filename (optional)", value="enhanced_ccts_simulation_results.xlsx")

def _validate_firms(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return "Firms CSV does not exist."
    try:
        df = pd.read_csv(p, nrows=5)
    except Exception as e:
        return f"Failed to read CSV: {e}"
    required = {"firm_id","sector","baseline_production","baseline_emissions_intensity","cash_on_hand","revenue_per_unit"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        return f"Firms CSV missing required columns: {missing}"
    return ""

def _build_cmd():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = str(Path(out_base) / f"run_{ts}_{scenario}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if "module" in run_mode:
        cmd = [
            python_exec, "-m", "ccts.run_simulation",
            "--firms-csv", firms_csv,
            "--scenario", scenario,
            "--steps", str(steps),
            "--out-dir", out_dir,
            "--log-level", log_level,
            "--excel-filename", excel_name
        ]
    else:
        cmd = [
            python_exec, str(Path(model_root) / "ccts" / "run_simulation.py"),
            "--firms-csv", firms_csv,
            "--scenario", scenario,
            "--steps", str(steps),
            "--out-dir", out_dir,
            "--log-level", log_level,
            "--excel-filename", excel_name
        ]
    if seed > 0:
        cmd += ["--seed", str(seed)]
    return cmd, out_dir

def _run_and_stream(cmd, cwd):
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    log_lines = []
    log_box = st.empty()
    with st.status("Running simulation…", expanded=True) as status:
        for line in proc.stdout:
            log_lines.append(line.rstrip("\n"))
            log_box.code("\n".join(log_lines[-200:]), language="bash")
        ret = proc.wait()
        if ret == 0:
            status.update(label="Run finished ✅", state="complete")
        else:
            status.update(label=f"Run failed (exit {ret}) ❌", state="error")
    return ret, "\n".join(log_lines)

st.markdown("---")
colA, colB = st.columns([1,1])

with colA:
    if st.button("Run Simulation", use_container_width=True, type="primary"):
        if not Path(model_root).exists():
            st.error("Model root does not exist."); st.stop()
        if not Path(python_exec).exists() and "python" not in str(python_exec).lower():
            st.error("Python executable not found — set a valid path like E:\\model_2\\.venv\\Scripts\\python.exe or just 'python'."); st.stop()
        msg = _validate_firms(firms_csv)
        if msg: st.error(msg); st.stop()

        cmd, out_dir = _build_cmd()
        st.write("**Working directory:** ", model_root)
        st.write("**Command:** ", " ".join(cmd))
        ret, logs = _run_and_stream(cmd, cwd=str(model_root))
        if ret == 0:
            st.success(f"Outputs saved to: {out_dir}")
            st.session_state["__ccts_output_path__"] = out_dir
            st.page_link("pages/1_Overview.py", label="Open Overview for this run →", icon="🏠")

with colB:
    st.markdown("""
**Tips**
- For imports to work, run from the project **root** (`model_root`).
- Prefer **module** mode if your `ccts` folder has `__init__.py`.
- If you hit `ModuleNotFoundError: ccts`, switch to **script** mode or adjust the working directory.
- If your firms CSV has a `macc` column with JSON strings, parse it to a list of dicts in your loader.
""")
'''

# write the page
(PAGES / "0_Run_Model.py").write_text(PAGE_CODE, encoding="utf-8")

# patch sidebar to include the page link, if missing
app_path = ROOT / "app.py"
if app_path.exists():
    txt = app_path.read_text(encoding="utf-8")
    if "pages/0_Run_Model.py" not in txt:
        txt = re.sub(
            r'(st\.page_link\("pages/1_Overview\.py".*\n)',
            'st.page_link("pages/0_Run_Model.py", label="Run Model", icon="🚀")\n\\1',
            txt,
            count=1
        )
        app_path.write_text(txt, encoding="utf-8")

print("Added pages/0_Run_Model.py and updated sidebar.")
