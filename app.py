# app.py — AstroGuard (Neon UI) with sidebar checkboxes + expanders
# Save & run: streamlit run app.py

import os, io, zipfile, time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# optional imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except Exception:
    SGP4_AVAILABLE = False

# ---------- config ----------
st.set_page_config(page_title="AstroGuard — Panels", layout="wide", initial_sidebar_state="expanded")
ROOT = Path.cwd()
OUT_DIR = ROOT / "astroguard_outputs"
MODELS_DIR = OUT_DIR / "models"
VIS_DIR = OUT_DIR / "visualizations"
REPORTS_DIR = OUT_DIR / "reports"
for d in [OUT_DIR, MODELS_DIR, VIS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_KM = 6378.137
np.random.seed(0)

# Developer-provided PPTX path (local to this session) — used as download link
PPTX_LOCAL_PATH = "/mnt/data/ASTROGUARD.pptx.pptx"
PPTX_URL = f"file://{PPTX_LOCAL_PATH}"

# ---------- CSS / Neon styles ----------
NEON_CSS = """
<style>
body { background: radial-gradient(circle at 10% 10%, #050017 0%, #03031a 40%, #020210 100%); color: #dbe9ff; }
.neon-card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding: 14px; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.03); margin-bottom:12px; }
.header-title { font-size:26px; font-weight:800; color: #00E5FF; letter-spacing:0.6px; }
.header-sub { color:#9aa6b2; font-size:13px; margin-top:-6px; }
.metric-card { background: rgba(255,255,255,0.02); padding: 10px; border-radius: 10px; border-left: 4px solid #00E5FF; margin-bottom:8px; }
.streamlit-expanderHeader { font-weight:700; }
.sidebar .stButton button { background: linear-gradient(90deg,#00e5ff,#9d4edd); color: #021; border: none; box-shadow: 0 8px 24px rgba(157,78,221,0.12); }
.footer { color:#5b6b7a; font-size:12px; margin-top:8px; }
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------- Helper functions (same robust logic as earlier) ----------
def safe_get_tles(n=6, timeout=6):
    urls = [
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"
    ]
    lines = []
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            txt = r.text.splitlines()
            if len(txt) > 10:
                lines = txt; break
        except Exception:
            continue
    if not lines:
        lines = [
            "ISS (ZARYA)",
            "1 25544U 98067A   20348.54791667  .00001264  00000-0  29634-4 0  9991",
            "2 25544  51.6448  20.6781 0001691  57.3457 302.8509 15.49409478"
        ]
    tles=[]
    for i in range(0, len(lines), 3):
        if i+2 < len(lines):
            name = lines[i].strip(); l1 = lines[i+1].strip(); l2 = lines[i+2].strip()
            if l1.startswith("1 ") and l2.startswith("2 "):
                tles.append({'name': name, 'line1': l1, 'line2': l2})
        if len(tles) >= n:
            break
    return tles

def build_times(start=None, duration_s=1200, step_s=10):
    if start is None: start = datetime.utcnow()
    steps = max(1, int(duration_s/step_s)+1)
    return [start + timedelta(seconds=i*step_s) for i in range(steps)]

def propagate_tle(line1, line2, times):
    if not SGP4_AVAILABLE:
        return np.full((len(times),3), np.nan)
    sat = Satrec.twoline2rv(line1, line2)
    coords=[]
    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond*1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e!=0:
            coords.append([np.nan, np.nan, np.nan])
        else:
            coords.append(r)
    return np.array(coords)

def circular_orbit_coords(r_km, incl_deg=0, rasc_deg=0, n_steps=200, phase=0):
    theta = np.linspace(0, 2*np.pi, n_steps)
    X = r_km * np.cos(theta + phase)
    Y = r_km * np.sin(theta + phase)
    Z = (np.sin(theta*3) * (r_km*0.03))
    inc = np.deg2rad(incl_deg); rasc = np.deg2rad(rasc_deg)
    x2 = X * np.cos(inc) - Z * np.sin(inc)
    z2 = X * np.sin(inc) + Z * np.cos(inc)
    x3 = x2 * np.cos(rasc) - Y * np.sin(rasc)
    y3 = x2 * np.sin(rasc) + Y * np.cos(rasc)
    return np.vstack([x3, y3, z2]).T

def generate_debris_field(times, n_debris=200, alt_choices=(500,700,900), spread_km=80):
    debris=[]
    mu=398600.4418
    for i in range(n_debris):
        shell = int(np.random.choice(alt_choices))
        alt = np.random.normal(loc=shell, scale=spread_km)
        a = EARTH_RADIUS_KM + max(160, alt)
        T = 2*np.pi * (a**1.5) / (mu**0.5)
        inc = np.random.uniform(0,98)
        rasc = np.random.uniform(0,360)
        phase0 = np.random.uniform(0, 2*np.pi)
        traj = circular_orbit_coords(a, incl_deg=inc, rasc_deg=rasc, n_steps=len(times), phase=phase0)
        traj += np.random.normal(0, 0.5, traj.shape)
        debris.append({"name":f"DEB_{i:04d}", "pos":traj})
    return debris

def detect_close_approaches(objects, times, threshold_km=30.0):
    events=[]
    N=len(times); dt=(times[1]-times[0]).total_seconds() if N>1 else 1.0
    M=len(objects)
    for ti in range(N):
        positions = [o["pos"][ti] for o in objects]
        for i in range(M):
            for j in range(i+1, M):
                p1 = positions[i]; p2 = positions[j]
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)): continue
                dist = np.linalg.norm(p1-p2)
                if dist <= threshold_km:
                    if 0<ti<N-1:
                        v1=(objects[i]["pos"][ti+1]-objects[i]["pos"][ti-1])/(2*dt)
                        v2=(objects[j]["pos"][ti+1]-objects[j]["pos"][ti-1])/(2*dt)
                    else:
                        v1=v2=np.zeros(3)
                    rel_vel = v1 - v2; rel_speed = np.linalg.norm(rel_vel)
                    rel_pos = p1 - p2
                    denom = np.dot(rel_vel, rel_vel)
                    tca = 0.0 if denom < 1e-9 else -np.dot(rel_pos, rel_vel)/denom
                    dv = min(300.0, 1200.0/(dist*1000.0 + 1e-6))
                    events.append({"t_idx":ti,"t_utc":times[ti],"A":objects[i]["name"],"B":objects[j]["name"],
                                   "dist_km":float(dist),"rel_speed_km_s":float(rel_speed),"tca_s":float(tca),"dv_m_s":float(dv)})
    return events

def ensure_dataset(X,y,min_samples=200):
    if len(X) < min_samples:
        add = max(0, min_samples - len(X))
        syn_X=[]; syn_y=[]
        for _ in range(add):
            d=np.random.uniform(0.2,40.0); s=np.random.uniform(0,5.0); tca=np.random.uniform(0,900)
            syn_X.append([d,s,tca]); syn_y.append(1 if d < 5.0 else 0)
        if len(X)==0:
            X=np.array(syn_X); y=np.array(syn_y)
        else:
            X=np.vstack([X,syn_X]); y=np.hstack([y,syn_y])
    return X,y

def train_models_safe(X,y,out_dir=MODELS_DIR):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, out_dir / "scaler.pkl")
    if Xs.shape[0] < 10:
        X_train,X_test,y_train,y_test = Xs,Xs,y,y
    else:
        strat = y if len(np.unique(y))>1 else None
        X_train,X_test,y_train,y_test = train_test_split(Xs,y,test_size=0.25,random_state=42,stratify=strat)

    models = {
        "LogReg": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradBoost": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=1200, random_state=42)
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, use_label_encoder=False, eval_metric="logloss", random_state=42)
    else:
        models["XGBoost"] = RandomForestClassifier(n_estimators=150, random_state=42)

    results=[]; trained={}
    for name, m in models.items():
        try:
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            y_proba = m.predict_proba(X_test)[:,1] if hasattr(m,"predict_proba") else None
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
            results.append([name, acc, f1, prec, rec, auc])
            trained[name] = m
            joblib.dump(m, out_dir / f"model_{name}.pkl")
        except Exception as e:
            print("train fail", name, e)

    estimators = [(n, trained[n]) for n in trained if hasattr(trained[n], "predict_proba")]
    if len(estimators) > 1:
        vc = VotingClassifier(estimators=estimators, voting="soft")
        vc.fit(X_train, y_train)
        joblib.dump(vc, out_dir / "model_Voting.pkl")
        y_pred = vc.predict(X_test); y_proba = vc.predict_proba(X_test)[:,1]
        results.append(["Voting", accuracy_score(y_test,y_pred), f1_score(y_test,y_pred), precision_score(y_test,y_pred), recall_score(y_test,y_pred), roc_auc_score(y_test,y_proba)])
        trained["Voting"] = vc

    df_results = pd.DataFrame(results, columns=["Model","Accuracy","F1","Precision","Recall","AUC"]).sort_values(by="F1", ascending=False)
    df_results.to_csv(out_dir / "model_results.csv", index=False)
    return df_results, trained, (X_train,X_test,y_train,y_test)

# plotting helpers
def plot_3d_scene(objects, times, frame_idx=None, title="3D Orbit"):
    if frame_idx is None: frame_idx = min(len(times)-1, int(len(times)/2))
    u=np.linspace(0,2*np.pi,60); v=np.linspace(0,np.pi,30)
    xs = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    ys = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    zs = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, colorscale=[[0,'#020018'],[1,'#071236']], opacity=0.93, showscale=False))
    colors = px.colors.qualitative.Dark24 + px.colors.sequential.Aggrnyl
    for i,o in enumerate(objects):
        traj = o["pos"][:frame_idx+1]
        if np.isnan(traj).all(): continue
        fig.add_trace(go.Scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode="lines", line=dict(color=colors[i%len(colors)], width=2), name=o["name"]))
        p = traj[-1]; fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", marker=dict(size=4, color=colors[i%len(colors)]), showlegend=False))
    fig.update_layout(template="plotly_dark", title=title, scene=dict(aspectmode="data"), height=640)
    return fig

def plot_metric_bars(df_results):
    fig_acc = px.bar(df_results, x="Model", y="Accuracy", color="Model", template="plotly_dark", title="Accuracy")
    fig_f1 = px.bar(df_results, x="Model", y="F1", color="Model", template="plotly_dark", title="F1 Score")
    return fig_acc, fig_f1

# ---------- Sidebar: checkboxes to enable panels ----------
st.sidebar.markdown("<div class='neon-card'><b>AstroGuard Panels</b></div>", unsafe_allow_html=True)
show_sim = st.sidebar.checkbox("🔭 Simulation Panel", value=True)
show_train = st.sidebar.checkbox("⚙ Training Panel", value=False)
show_viz = st.sidebar.checkbox("📊 Visualizations Panel", value=True)
show_download = st.sidebar.checkbox("📁 Downloads Panel", value=False)

# Panel-specific settings inside sidebar (only visible when corresponding checkbox is True)
st.sidebar.markdown("---")
if show_sim:
    st.sidebar.markdown("**Simulation settings**")
    duration_s = st.sidebar.slider("Duration (s)", 600, 3600, 1200, step=200, key="dur")
    step_s = st.sidebar.slider("Timestep (s)", 4, 20, 8, step=2, key="step")
    n_debris = st.sidebar.slider("Debris count", 50, 800, 200, step=50, key="debris")
    tles_count = st.sidebar.slider("TLEs to load", 0, 12, 6, step=1, key="tles")
    threshold_km = st.sidebar.slider("Threshold km", 5.0, 200.0, 30.0, step=5.0, key="thres")
else:
    # default values (when panel not shown)
    duration_s = st.sidebar.number_input("Duration (s)", value=1200, key="dur_hidden", disabled=True)
    step_s = 8; n_debris = 200; tles_count = 6; threshold_km = 30.0

if show_train:
    st.sidebar.markdown("**Training settings**")
    min_train = st.sidebar.number_input("Min dataset size", min_value=50, max_value=2000, value=300, step=50, key="mintrain")
else:
    min_train = 300

st.sidebar.markdown("---")
if Path(PPTX_LOCAL_PATH).exists():
    st.sidebar.markdown(f"Uploaded PPTX: [Open PPTX]({PPTX_URL})")
else:
    st.sidebar.markdown("_PPTX not found in session path_")
st.sidebar.markdown("<div class='footer'>AstroGuard • Select panels, then open sections</div>", unsafe_allow_html=True)

# session state
if "sim" not in st.session_state: st.session_state["sim"] = None
if "df_results" not in st.session_state: st.session_state["df_results"] = None
if "trained" not in st.session_state: st.session_state["trained"] = None

# header
st.markdown("<div style='display:flex;align-items:center;gap:12px'>"
            "<img src='https://img.icons8.com/color/96/satellite.png' width=48/>"
            "<div><div class='header-title'>AstroGuard</div><div class='header-sub'>Neon UI — select panels to view</div></div>"
            "</div>", unsafe_allow_html=True)

# MAIN: show only selected panels; each panel is an expander (arrow)
# Simulation Panel
if show_sim:
    with st.expander("🔭 Simulation Panel — run sim & preview", expanded=True):
        col1, col2 = st.columns([3,1])
        with col2:
            st.markdown("<div class='metric-card'><b>Sim Controls</b><br>"
                        f"Duration: {duration_s}s<br>Step: {step_s}s<br>Debris: {n_debris}<br>Threshold: {threshold_km} km</div>", unsafe_allow_html=True)
            if st.button("▶ Run Simulation", key="run_sim_main"):
                with st.spinner("Running simulation..."):
                    times = build_times(duration_s=duration_s, step_s=step_s)
                    objects = []
                    if tles_count > 0:
                        tles = safe_get_tles(n=tles_count)
                        for idx,t in enumerate(tles):
                            pos = propagate_tle(t["line1"], t["line2"], times)
                            if np.isnan(pos).all():
                                traj = circular_orbit_coords(7000 + idx*200, incl_deg=idx*10, n_steps=len(times), phase=np.random.uniform(0,2*np.pi))
                                objects.append({"name":t["name"], "pos":traj})
                            else:
                                objects.append({"name":t["name"], "pos":pos})
                    debris = generate_debris_field(times, n_debris=n_debris)
                    objects = objects + debris
                    events = detect_close_approaches(objects, times, threshold_km=threshold_km)
                    st.session_state["sim"] = {"times":times, "objects":objects, "events":events}
                    st.success(f"Simulation complete — {len(events)} events detected.")
        with col1:
            if st.session_state["sim"] is not None:
                sim = st.session_state["sim"]
                st.markdown("**Simulation summary**")
                st.write(f"Objects: {len(sim['objects'])} • Timesteps: {len(sim['times'])} • Events: {len(sim['events'])}")
                if len(sim["events"])>0:
                    st.dataframe(pd.DataFrame(sim["events"][:20]))
                st.markdown("**3D preview**")
                frame_idx = st.slider("Preview frame index", 0, max(0,len(sim["times"])-1), min(10, max(0,len(sim["times"])-1)), key="frame_sim")
                fig3d = plot_3d_scene(sim["objects"], sim["times"], frame_idx=frame_idx, title="Orbit Preview")
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("No simulation ran yet. Press ▶ Run Simulation in the panel.")

# Training Panel
if show_train:
    with st.expander("⚙ Training Panel — build dataset & train models", expanded=False):
        if st.button("⚙ Train Models (7)", key="train_main"):
            if st.session_state["sim"] is None:
                st.error("Run simulation first (enable Simulation panel).")
            else:
                with st.spinner("Preparing dataset & training..."):
                    events = st.session_state["sim"]["events"]
                    X=[]; y=[]
                    for e in events:
                        X.append([e["dist_km"], e["rel_speed_km_s"], abs(e["tca_s"])])
                        y.append(1 if e["dist_km"] < 5.0 else 0)
                    X=np.array(X); y=np.array(y)
                    X,y = ensure_dataset(X,y,min_samples=min_train)
                    df_results, trained, splits = train_models_safe(X,y, out_dir=MODELS_DIR)
                    st.session_state["df_results"] = df_results
                    st.session_state["trained"] = trained
                    st.success("Training finished. Models saved.")
        # display training results if present
        if st.session_state.get("df_results") is not None:
            st.markdown("**Latest training results**")
            st.write(st.session_state["df_results"])

# Visualizations Panel
if show_viz:
    with st.expander("📊 Visualizations Panel — charts & model metrics", expanded=False):
        if st.session_state.get("sim") is None:
            st.info("Run Simulation first, then come here to explore visuals.")
        else:
            sim = st.session_state["sim"]
            st.markdown("**Orbit 3D snapshot**")
            frame_idx = st.slider("Viz frame index", 0, max(0,len(sim["times"])-1), min(10, max(0,len(sim["times"])-1)), key="frame_viz")
            fig3d = plot_3d_scene(sim["objects"], sim["times"], frame_idx=frame_idx, title="Orbit Visualization")
            st.plotly_chart(fig3d, use_container_width=True)
            if st.button("Save orbit HTML", key="save_html_viz"):
                fn = VIS_DIR / f"orbit_{int(time.time())}.html"
                fig3d.write_html(str(fn), include_plotlyjs="cdn")
                st.success(f"Saved {fn.name}")
                with open(fn,"rb") as fh: st.download_button("⬇ Download orbit HTML", fh, file_name=fn.name, mime="text/html")

        if st.session_state.get("df_results") is not None:
            df = st.session_state["df_results"]
            st.markdown("**Model performance**")
            acc_fig, f1_fig = plot_metric_bars(df)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(acc_fig, use_container_width=True)
            with c2:
                st.plotly_chart(f1_fig, use_container_width=True)

# Downloads Panel
if show_download:
    with st.expander("📁 Downloads Panel — package & export", expanded=False):
        st.markdown("**Exports**")
        if st.button("📦 Package outputs & download ZIP", key="zip_main"):
            zipname = ROOT / "AstroGuard_Submission.zip"
            with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as zf:
                for folder in [MODELS_DIR, VIS_DIR, REPORTS_DIR]:
                    for root,_,files in os.walk(folder):
                        for fname in files:
                            fp = os.path.join(root, fname)
                            arc = os.path.relpath(fp, OUT_DIR)
                            zf.write(fp, arc)
                if Path(PPTX_LOCAL_PATH).exists():
                    zf.write(PPTX_LOCAL_PATH, os.path.join("ppt", os.path.basename(PPTX_LOCAL_PATH)))
            st.success(f"Packaged to {zipname.name}")
            with open(zipname,"rb") as fh:
                st.download_button("⬇ Download ZIP", fh, file_name=zipname.name, mime="application/zip")
        st.markdown("---")
        st.markdown("**Direct files**")
        # list models
        if MODELS_DIR.exists():
            models_list = list(MODELS_DIR.glob("*.pkl"))
            if models_list:
                for m in models_list:
                    with open(m,"rb") as fh:
                        st.download_button(f"⬇ {m.name}", fh, file_name=m.name, mime="application/octet-stream")
        # PPTX
        if Path(PPTX_LOCAL_PATH).exists():
            st.markdown("Download uploaded PPTX")
            with open(PPTX_LOCAL_PATH, "rb") as fh:
                st.download_button("⬇ Download PPTX", fh, file_name=os.path.basename(PPTX_LOCAL_PATH), mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
        else:
            st.info("No PPTX available in session path.")

# footer
st.markdown("---")
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>"
            "<div style='color:#9aa6b2'>AstroGuard • Neon Panels UI</div>"
            "<div style='color:#5b6b7a;font-size:12px'>Tip: enable panels on the left to reveal sections here</div>"
            "</div>", unsafe_allow_html=True)

if not XGBOOST_AVAILABLE:
    st.warning("XGBoost not available in this environment. App uses fallback RandomForest for that slot.")
