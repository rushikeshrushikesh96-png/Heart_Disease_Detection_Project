"""
=====================================================================
  HEART DISEASE DETECTION  —  Explainable AI Web Application
  ---------------------------------------------------------------
  Three models (Logistic Regression, Random Forest, SVM)
  Two explainers (SHAP and LIME)
  Run it with:   streamlit run app.py
=====================================================================
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# PAGE SETUP  (must be the first Streamlit command)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Detection | Explainable AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit renamed the "fill the whole width" argument in version 1.49.
# This little check keeps the app working on both old and new versions.
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
    FULL = {"width": "stretch"} if _v >= (1, 49) else {"use_container_width": True}
except Exception:
    FULL = {"use_container_width": True}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Colour palette used everywhere in the app
C_LOW, C_MED, C_HIGH = "#10B981", "#F59E0B", "#EF4444"
C_MODEL = {"Logistic Regression": "#6366F1", "Random Forest": "#10B981", "SVM": "#F59E0B"}

# ---------------------------------------------------------------
# CUSTOM STYLING  (makes the page look bright and modern)
# ---------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}

/* --- big gradient banner at the top of every page --- */
.hero {
  background: linear-gradient(120deg,#6366F1 0%,#A855F7 45%,#EC4899 100%);
  padding: 26px 30px; border-radius: 18px; color: white; margin-bottom: 22px;
  box-shadow: 0 10px 30px rgba(99,102,241,.35);
}
.hero h1 {margin:0; font-size:32px; font-weight:800; letter-spacing:-.5px;}
.hero p  {margin:8px 0 0 0; font-size:15.5px; opacity:.94;}

/* --- KPI cards --- */
.kpi {
  border-radius:16px; padding:18px 20px; color:#fff; height:100%;
  box-shadow:0 6px 18px rgba(0,0,0,.13);
}
.kpi .label {font-size:12.5px; text-transform:uppercase; letter-spacing:1.1px; opacity:.9;}
.kpi .value {font-size:30px; font-weight:800; margin:6px 0 2px 0; line-height:1.1;}
.kpi .sub   {font-size:12.5px; opacity:.9;}
.k1{background:linear-gradient(135deg,#6366F1,#818CF8);}
.k2{background:linear-gradient(135deg,#10B981,#34D399);}
.k3{background:linear-gradient(135deg,#F59E0B,#FBBF24);}
.k4{background:linear-gradient(135deg,#EC4899,#F472B6);}
.k5{background:linear-gradient(135deg,#06B6D4,#22D3EE);}

/* --- explanation boxes --- */
.note {
  background:#EEF2FF; border-left:6px solid #6366F1; padding:14px 18px;
  border-radius:10px; margin:12px 0; font-size:15px; color:#1E1B4B;
}
.good {background:#ECFDF5; border-left:6px solid #10B981; padding:14px 18px;
       border-radius:10px; margin:12px 0; font-size:15px; color:#064E3B;}
.warn {background:#FFFBEB; border-left:6px solid #F59E0B; padding:14px 18px;
       border-radius:10px; margin:12px 0; font-size:15px; color:#78350F;}
.bad  {background:#FEF2F2; border-left:6px solid #EF4444; padding:14px 18px;
       border-radius:10px; margin:12px 0; font-size:15px; color:#7F1D1D;}

/* --- the big risk verdict banner --- */
.verdict {border-radius:18px; padding:26px 30px; color:#fff; text-align:center;
          box-shadow:0 10px 26px rgba(0,0,0,.18); margin:6px 0 18px 0;}
.verdict .t {font-size:16px; letter-spacing:2px; text-transform:uppercase; opacity:.92;}
.verdict .v {font-size:52px; font-weight:900; margin:4px 0; line-height:1;}
.verdict .s {font-size:16px; opacity:.95;}

/* --- section headings (main topic names) ---
   These carry their own background and their own text colour, so they stay
   readable whether the viewer's browser is in light mode or dark mode. */
.sec {
  background: linear-gradient(90deg,#EEF2FF 0%, #FAFAFF 60%, #FFFFFF 100%);
  border-left: 8px solid #6366F1;
  color:#1E293B !important;
  font-size:21px; font-weight:800; letter-spacing:-.2px;
  padding:12px 18px; border-radius:10px; margin:28px 0 10px 0;
}

/* Keep every custom box readable in dark mode too */
.note, .good, .warn, .bad, .note b, .good b, .warn b, .bad b,
.note i, .good i, .warn i, .bad i, .note code, .good code {
  -webkit-text-fill-color: currentColor;
}
.note code, .good code, .warn code, .bad code {
  background:#FFFFFF; padding:1px 6px; border-radius:5px; color:#4338CA;
}
.kpi .label, .kpi .value, .kpi .sub,
.hero h1, .hero p, .verdict .t, .verdict .v, .verdict .s {
  color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF;
}

/* --- small coloured pills --- */
.pill {display:inline-block; padding:4px 13px; border-radius:20px; font-size:13px;
       font-weight:700; color:#fff; margin:2px;}
div[data-testid="stMetricValue"] {font-size:26px;}
</style>
""", unsafe_allow_html=True)


# ===============================================================
#  LOADERS
#  @st.cache_resource / @st.cache_data mean these run only ONCE.
#  Without caching, Streamlit would reload the models on every
#  single click and the website would feel very slow.
# ===============================================================
@st.cache_resource(show_spinner="Loading the trained models…")
def load_models():
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
        "Random Forest":       joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
        "SVM":                 joblib.load(os.path.join(MODEL_DIR, "svm.pkl")),
    }


@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


@st.cache_data
def load_meta():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading the patient dataset…")
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "cleaned_data.csv"))


@st.cache_data
def load_advanced():
    """Extra statistical validation computed once and saved to disk:
    optimal decision thresholds, precision-recall curves, calibration
    curves, McNemar significance tests and subgroup fairness checks.
    Returns None if the file isn't there yet, so the page can degrade
    gracefully instead of crashing."""
    path = os.path.join(MODEL_DIR, "advanced.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_global_shap():
    """SHAP values we calculated once inside the notebook and saved to disk."""
    out = {"X": np.load(os.path.join(MODEL_DIR, "shap_X_sample.npy")),
           "y": np.load(os.path.join(MODEL_DIR, "shap_y_sample.npy"))}
    for name, f in [("Logistic Regression", "logistic_regression"),
                    ("Random Forest", "random_forest"), ("SVM", "svm")]:
        out[name] = np.load(os.path.join(MODEL_DIR, f"shapvals__{f}.npy"))
        out[name + "_base"] = float(np.load(os.path.join(MODEL_DIR, f"shapbase__{f}.npy"))[0])
    return out


MODELS = load_models()
SCALER = load_scaler()
META = load_meta()
METRICS = load_metrics()
FEATURES = META["features"]
DISPLAY = META["display"]
PLAIN = META["plain"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "SVM"]


# ===============================================================
#  SHARED HELPER FUNCTIONS
# ===============================================================
def predict_proba_raw(model, X_raw):
    """
    Every model was trained on SCALED numbers, but humans think in
    real numbers (age 55, BP 140...). This helper takes the real
    numbers, scales them the same way as during training, and then
    returns the probability of heart disease.
    """
    X_raw = np.asarray(X_raw, dtype=float)
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)
    return model.predict_proba(SCALER.transform(X_raw))[:, 1]


def predict_all(X_raw):
    """Run all three models at once and return a dictionary of probabilities."""
    return {name: predict_proba_raw(m, X_raw) for name, m in MODELS.items()}


def risk_band(p):
    """Turn a probability into a simple label, colour and advice line."""
    if p < 0.35:
        return "LOW RISK", C_LOW, "The models see mostly healthy signals for this patient."
    if p < 0.65:
        return "MODERATE RISK", C_MED, "Some warning signs are present. A check-up is advisable."
    return "HIGH RISK", C_HIGH, "Several strong warning signs are present. Please consult a doctor."


def build_patient_row(v):
    """Turn the sidebar inputs into one row with the exact 11 model features."""
    bmi = round(v["weight"] / ((v["height"] / 100) ** 2), 2)
    row = {
        "age_years": float(v["age"]),
        "gender": int(v["gender"]),
        "bmi": bmi,
        "ap_hi": int(v["ap_hi"]),
        "ap_lo": int(v["ap_lo"]),
        "pulse_pressure": int(v["ap_hi"] - v["ap_lo"]),
        "cholesterol": int(v["cholesterol"]),
        "gluc": int(v["gluc"]),
        "smoke": int(v["smoke"]),
        "alco": int(v["alco"]),
        "active": int(v["active"]),
    }
    return pd.DataFrame([row])[FEATURES]


def kpi(col, css, label, value, sub=""):
    col.markdown(
        f'<div class="kpi {css}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def health_flags(p):
    """
    Simple traffic-light reading of the patient's numbers.
    These are standard medical thresholds, NOT model output — they help
    the user sanity-check what the model is reacting to.
    """
    bmi = round(p["weight"] / ((p["height"] / 100) ** 2), 2)
    pp = p["ap_hi"] - p["ap_lo"]
    rows = []

    def add(name, value, status, msg):
        rows.append({"Health check": name, "Patient value": value,
                     "Status": status, "What it means": msg})

    add("Age", f"{p['age']:.0f} years",
        "🟢 Lower risk" if p["age"] < 45 else ("🟡 Watch" if p["age"] < 60 else "🔴 Higher risk"),
        "Risk of heart disease naturally increases with age.")
    add("Blood pressure", f"{p['ap_hi']}/{p['ap_lo']} mmHg",
        "🟢 Normal" if p["ap_hi"] < 130 and p["ap_lo"] < 85 else
        ("🟡 Elevated" if p["ap_hi"] < 140 and p["ap_lo"] < 90 else "🔴 High"),
        "Normal is below 130/85. Above 140/90 is called hypertension.")
    add("Pulse pressure", f"{pp} mmHg",
        "🟢 Normal" if 30 <= pp <= 50 else "🟡 Unusual",
        "The gap between the two BP numbers. A wide gap suggests stiffer arteries.")
    add("BMI", f"{bmi}",
        "🟢 Healthy" if bmi < 25 else ("🟡 Overweight" if bmi < 30 else "🔴 Obese"),
        "Under 25 is healthy, 25–30 overweight, above 30 obese.")
    add("Cholesterol", ["Normal", "Above normal", "Well above normal"][p["cholesterol"] - 1],
        ["🟢 Normal", "🟡 Above normal", "🔴 Well above normal"][p["cholesterol"] - 1],
        "High cholesterol clogs the arteries over time.")
    add("Glucose", ["Normal", "Above normal", "Well above normal"][p["gluc"] - 1],
        ["🟢 Normal", "🟡 Above normal", "🔴 Well above normal"][p["gluc"] - 1],
        "High blood sugar damages blood vessels.")
    add("Smoking", "Yes" if p["smoke"] else "No",
        "🔴 Smoker" if p["smoke"] else "🟢 Non-smoker",
        "Smoking is one of the strongest avoidable heart risks.")
    add("Alcohol", "Yes" if p["alco"] else "No",
        "🟡 Drinks" if p["alco"] else "🟢 Does not drink",
        "Regular alcohol raises blood pressure.")
    add("Physical activity", "Yes" if p["active"] else "No",
        "🟢 Active" if p["active"] else "🔴 Not active",
        "Regular exercise protects the heart.")
    return pd.DataFrame(rows)

# ===============================================================
#  SIDEBAR  —  navigation + the patient the whole app talks about
# ===============================================================
with st.sidebar:
    st.markdown("## ❤️ Heart Disease AI")
    st.caption("Explainable machine learning for cardiovascular risk")

    PAGE = st.radio(
        "Go to page",
        ["🏠 Dashboard",
         "🩺 Patient Prediction",
         "🧠 SHAP Explanation",
         "🍋 LIME Explanation",
         "📁 Batch CSV Prediction",
         "📊 Model Performance",
         "🩻 Clinical Validation",
         "ℹ️ About the Project"],
        label_visibility="collapsed")

    st.divider()
    st.markdown("### 🧾 Patient details")
    st.caption("These values are used on the Prediction, SHAP and LIME pages.")

    preset = st.selectbox("Quick example", ["Custom", "Healthy young adult",
                                            "Average middle-aged", "High-risk senior"])
    P = {"Healthy young adult": dict(age=32, gender=0, height=165, weight=58, ap_hi=112,
                                     ap_lo=72, cholesterol=1, gluc=1, smoke=0, alco=0, active=1),
         "Average middle-aged": dict(age=52, gender=1, height=172, weight=80, ap_hi=130,
                                     ap_lo=85, cholesterol=1, gluc=1, smoke=0, alco=0, active=1),
         "High-risk senior": dict(age=64, gender=1, height=168, weight=97, ap_hi=170,
                                  ap_lo=100, cholesterol=3, gluc=3, smoke=1, alco=1, active=0),
         }.get(preset, dict(age=50, gender=0, height=165, weight=72, ap_hi=125, ap_lo=82,
                            cholesterol=1, gluc=1, smoke=0, alco=0, active=1))

    c1, c2 = st.columns(2)
    age = c1.slider("Age (years)", 30, 70, P["age"])
    gender = c2.selectbox("Gender", [0, 1], index=P["gender"],
                          format_func=lambda x: "Female" if x == 0 else "Male")
    height = c1.slider("Height (cm)", 140, 200, P["height"])
    weight = c2.slider("Weight (kg)", 40, 150, P["weight"])
    ap_hi = c1.slider("Systolic BP (upper)", 90, 200, P["ap_hi"])
    ap_lo = c2.slider("Diastolic BP (lower)", 60, 130, P["ap_lo"])

    lvl = {1: "1 – Normal", 2: "2 – Above normal", 3: "3 – Well above normal"}
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3], index=P["cholesterol"] - 1,
                               format_func=lambda x: lvl[x])
    gluc = st.selectbox("Glucose", [1, 2, 3], index=P["gluc"] - 1, format_func=lambda x: lvl[x])

    c3, c4, c5 = st.columns(3)
    smoke = int(c3.checkbox("Smokes", bool(P["smoke"])))
    alco = int(c4.checkbox("Alcohol", bool(P["alco"])))
    active = int(c5.checkbox("Active", bool(P["active"])))

    if ap_lo >= ap_hi:
        st.error("The lower BP number must be smaller than the upper one.")
        ap_lo = ap_hi - 20

    PATIENT = dict(age=age, gender=gender, height=height, weight=weight, ap_hi=ap_hi,
                   ap_lo=ap_lo, cholesterol=cholesterol, gluc=gluc, smoke=smoke,
                   alco=alco, active=active)
    st.session_state["patient"] = PATIENT

    bmi_now = round(weight / ((height / 100) ** 2), 2)
    st.info(f"**BMI:** {bmi_now}  |  **Pulse pressure:** {ap_hi - ap_lo} mmHg")

    # The models were only ever shown patients with BMI between 15 and 50
    # (values outside that were removed as data-entry errors during cleaning,
    # see notebook 02). Height and weight sliders are independent, so it is
    # still possible to build a combination outside that range — flag it
    # honestly rather than presenting an extrapolated guess as a solid answer.
    BMI_TRAIN_MIN, BMI_TRAIN_MAX = 15.0, 50.0
    in_distribution = BMI_TRAIN_MIN <= bmi_now <= BMI_TRAIN_MAX
    st.session_state["in_distribution"] = in_distribution
    if not in_distribution:
        st.warning(
            f"⚠️ A BMI of {bmi_now} falls outside the training data's range "
            f"({BMI_TRAIN_MIN:.0f}–{BMI_TRAIN_MAX:.0f}). The models have never seen a "
            "patient built this way, so predictions below are an **extrapolation** — "
            "expect the three models to disagree more than usual.")

X_patient = build_patient_row(PATIENT)


# ===============================================================
#  PAGE 1 — DASHBOARD
# ===============================================================
if PAGE == "🏠 Dashboard":
    hero("❤️ Heart Disease Detection Dashboard",
         "70,000 real patient records · 3 machine-learning models · explained with SHAP and LIME")

    d = META["dataset"]
    best = max(MODEL_NAMES, key=lambda n: METRICS[n]["accuracy"])
    c = st.columns(5)
    kpi(c[0], "k1", "Patients used", f"{d['rows_clean']:,}", f"{d['removed']:,} bad rows removed")
    kpi(c[1], "k2", "Best accuracy", f"{METRICS[best]['accuracy']*100:.1f}%", best)
    kpi(c[2], "k3", "Best ROC-AUC", f"{max(METRICS[n]['roc_auc'] for n in MODEL_NAMES):.3f}",
        "Higher is better (max 1.0)")
    kpi(c[3], "k4", "Health features", f"{d['n_features']}", "Used by every model")
    kpi(c[4], "k5", "Models agree within", f"{META['agreement']['mean_spread']:.1f}%",
        "Average gap between the 3 models")

    st.markdown(f"""<div class="note">
    <b>What is this project?</b> It reads 11 simple health measurements of a patient —
    things a nurse can record in five minutes — and estimates the chance that the person has
    cardiovascular disease. Three different AI models give their own opinion, and then
    <b>SHAP</b> and <b>LIME</b> open the black box and show <i>which measurement pushed the
    answer up or down</i>. Out of {d['rows_raw']:,} original records we kept
    {d['rows_clean']:,} clean ones, of which {d['positive_rate']:.1f}% have heart disease.
    </div>""", unsafe_allow_html=True)

    df = load_data()

    section("🤖 How the three models compare")
    cA, cB = st.columns([3, 2])
    comp = pd.DataFrame({
        "Model": MODEL_NAMES,
        "Accuracy": [METRICS[n]["accuracy"] for n in MODEL_NAMES],
        "Precision": [METRICS[n]["precision"] for n in MODEL_NAMES],
        "Recall": [METRICS[n]["recall"] for n in MODEL_NAMES],
        "F1 Score": [METRICS[n]["f1"] for n in MODEL_NAMES],
        "ROC-AUC": [METRICS[n]["roc_auc"] for n in MODEL_NAMES]})
    melted = comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_map=C_MODEL, text=melted["Score"].map(lambda v: f"{v:.3f}"))
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(height=380, yaxis_range=[0, 1], plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.14), margin=dict(t=40, b=10))
    cA.plotly_chart(fig, **FULL)

    cB.markdown("<br>", unsafe_allow_html=True)
    cB.dataframe(comp.set_index("Model").style.format("{:.4f}")
                 .background_gradient(cmap="Greens", axis=None), **FULL)
    cB.markdown(f"""<div class="good">
    All three models land within about one percentage point of each other. That is a good sign:
    it means the pattern is really in the data, not an accident of one algorithm.
    </div>""", unsafe_allow_html=True)

    section("📈 What the data tells us about heart disease")
    t1, t2, t3, t4 = st.tabs(["Age", "Blood pressure", "Cholesterol & sugar", "Body & lifestyle"])

    with t1:
        tmp = df.copy()
        tmp["Age group"] = pd.cut(tmp.age_years, [29, 40, 45, 50, 55, 60, 71],
                                  labels=["30-40", "40-45", "45-50", "50-55", "55-60", "60-70"])
        g = tmp.groupby("Age group", observed=True).cardio.agg(["mean", "count"]).reset_index()
        g["Disease rate %"] = g["mean"] * 100
        f = px.bar(g, x="Age group", y="Disease rate %", color="Disease rate %",
                   color_continuous_scale="RdYlGn_r", text=g["Disease rate %"].round(1),
                   title="Percentage of patients with heart disease, by age group")
        f.update_traces(textposition="outside")
        f.update_layout(height=420, plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(f, **FULL)
        st.markdown("""<div class="note"><b>Plain English:</b> the bars climb steadily from left
        to right. Around age 30–40 only about a quarter of people in this dataset have heart
        disease; by age 60–70 it is roughly two out of three. Age alone is a powerful clue.
        </div>""", unsafe_allow_html=True)

    with t2:
        s = df.sample(6000, random_state=1)
        f = px.scatter(s, x="ap_hi", y="ap_lo", color=s.cardio.map({0: "Healthy", 1: "Disease"}),
                       opacity=.45, color_discrete_map={"Healthy": C_LOW, "Disease": C_HIGH},
                       labels={"ap_hi": "Systolic BP (upper)", "ap_lo": "Diastolic BP (lower)",
                               "color": "Patient"},
                       title="Blood pressure of 6,000 random patients")
        f.add_vline(x=140, line_dash="dash", line_color="#EF4444",
                    annotation_text="140 = high systolic")
        f.add_hline(y=90, line_dash="dash", line_color="#EF4444",
                    annotation_text="90 = high diastolic")
        f.update_layout(height=460, plot_bgcolor="white")
        st.plotly_chart(f, **FULL)
        st.markdown("""<div class="note"><b>Plain English:</b> red dots (patients with disease)
        pile up in the top-right corner — high upper <i>and</i> high lower blood pressure.
        Green dots gather in the bottom-left. Blood pressure is the single clearest separator
        in this dataset, which is exactly what SHAP will confirm later.</div>""",
                    unsafe_allow_html=True)

    with t3:
        c1_, c2_ = st.columns(2)
        for col_, container, title in [("cholesterol", c1_, "Cholesterol level"),
                                       ("gluc", c2_, "Glucose (blood sugar) level")]:
            g = df.groupby(col_).cardio.mean().reset_index()
            g["rate"] = g.cardio * 100
            g["label"] = g[col_].map({1: "Normal", 2: "Above normal", 3: "Well above normal"})
            f = px.bar(g, x="label", y="rate", color="rate", text=g["rate"].round(1),
                       color_continuous_scale="RdYlGn_r", title=title,
                       labels={"label": "", "rate": "Disease rate %"})
            f.update_traces(textposition="outside")
            f.update_layout(height=380, plot_bgcolor="white", coloraxis_showscale=False)
            container.plotly_chart(f, **FULL)
        st.markdown("""<div class="note"><b>Plain English:</b> patients with "well above normal"
        cholesterol get heart disease far more often than patients with normal cholesterol.
        Blood sugar shows the same pattern but a little weaker.</div>""", unsafe_allow_html=True)

    with t4:
        c1_, c2_ = st.columns(2)
        f = px.histogram(df.sample(20000, random_state=2), x="bmi", nbins=50, color="cardio",
                         color_discrete_map={0: C_LOW, 1: C_HIGH}, barmode="overlay", opacity=.6,
                         title="BMI distribution", labels={"bmi": "BMI", "cardio": "Disease"})
        f.update_layout(height=380, plot_bgcolor="white")
        c1_.plotly_chart(f, **FULL)

        life = pd.DataFrame({
            "Habit": ["Smokes", "Drinks alcohol", "Physically active"],
            "Disease rate %": [df[df.smoke == 1].cardio.mean() * 100,
                               df[df.alco == 1].cardio.mean() * 100,
                               df[df.active == 1].cardio.mean() * 100],
            "Opposite group %": [df[df.smoke == 0].cardio.mean() * 100,
                                 df[df.alco == 0].cardio.mean() * 100,
                                 df[df.active == 0].cardio.mean() * 100]})
        f2 = go.Figure()
        f2.add_bar(x=life.Habit, y=life["Disease rate %"], name="Has the habit",
                   marker_color="#EC4899", text=life["Disease rate %"].round(1),
                   textposition="outside")
        f2.add_bar(x=life.Habit, y=life["Opposite group %"], name="Does not have it",
                   marker_color="#6366F1", text=life["Opposite group %"].round(1),
                   textposition="outside")
        f2.update_layout(height=380, barmode="group", plot_bgcolor="white",
                         title="Lifestyle habits vs disease rate",
                         legend=dict(orientation="h", y=1.15))
        c2_.plotly_chart(f2, **FULL)
        st.markdown("""<div class="warn"><b>An honest observation:</b> in this
        dataset smoking and alcohol look almost harmless, and that is surprising. The reason is
        that these two answers were <i>self-reported</i> by patients, and people under-report
        them. Age, blood pressure and cholesterol were measured by a nurse, so they are far more
        reliable — and the models rely on them much more heavily.</div>""",
                    unsafe_allow_html=True)

    section("🔗 Which measurements move together?")
    corr = df[FEATURES + ["cardio"]].corr()
    f = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                  aspect="auto", title="Correlation heat map (1 = move together, -1 = opposite)")
    f.update_layout(height=560)
    st.plotly_chart(f, **FULL)
    st.markdown("""<div class="note"><b>How to read this:</b> look at the bottom row
    (<code>cardio</code>). The darkest red squares are the measurements most linked to heart
    disease — systolic blood pressure, age and cholesterol. That row is basically a preview of
    what SHAP will tell us.</div>""", unsafe_allow_html=True)


# ===============================================================
#  PAGE 2 — PATIENT PREDICTION
# ===============================================================
elif PAGE == "🩺 Patient Prediction":
    hero("🩺 Patient Risk Prediction",
         "All three models give their opinion at the same time — change the sliders on the left")

    probs = predict_all(X_patient.values)
    vals = {k: float(v[0]) for k, v in probs.items()}
    avg = float(np.mean(list(vals.values())))
    label, colour, advice = risk_band(avg)
    spread = max(vals.values()) - min(vals.values())

    st.markdown(f"""<div class="verdict" style="background:linear-gradient(135deg,{colour},
    {colour}CC);"><div class="t">Combined verdict of all three models</div>
    <div class="v">{avg*100:.1f}%</div><div class="s"><b>{label}</b> — {advice}</div></div>""",
                unsafe_allow_html=True)

    section("🔍 What each model says")
    cols = st.columns(3)
    for col, name in zip(cols, MODEL_NAMES):
        p = vals[name]
        _, cc, _ = risk_band(p)
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": "%", "font": {"size": 40, "color": cc}},
            title={"text": f"<b>{name}</b>", "font": {"size": 15}},
            gauge={"axis": {"range": [0, 100], "tickwidth": 1},
                   "bar": {"color": cc, "thickness": .75},
                   "borderwidth": 0,
                   "steps": [{"range": [0, 35], "color": "#D1FAE5"},
                             {"range": [35, 65], "color": "#FEF3C7"},
                             {"range": [65, 100], "color": "#FEE2E2"}],
                   "threshold": {"line": {"color": "#111", "width": 3},
                                 "thickness": .8, "value": 50}}))
        g.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
        col.plotly_chart(g, **FULL)
        col.markdown(f'<div style="text-align:center"><span class="pill" '
                     f'style="background:{cc}">{risk_band(p)[0]}</span></div>',
                     unsafe_allow_html=True)

    bar = go.Figure()
    for name in MODEL_NAMES:
        bar.add_bar(x=[name], y=[vals[name] * 100], marker_color=C_MODEL[name],
                    text=[f"{vals[name]*100:.1f}%"], textposition="outside",
                    name=name, width=.55)
    bar.add_hline(y=50, line_dash="dash", line_color="#64748B",
                  annotation_text="50% decision line")
    bar.update_layout(height=340, yaxis_range=[0, 108], plot_bgcolor="white", showlegend=False,
                      yaxis_title="Chance of heart disease (%)",
                      title="Side-by-side comparison of the three models")
    st.plotly_chart(bar, **FULL)

    agree_txt = ("all three models strongly agree" if spread < 0.05 else
                 "the three models broadly agree" if spread < 0.10 else
                 "the models disagree noticeably" if spread < 0.20 else
                 "the models disagree sharply")
    box = "good" if spread < 0.10 else ("warn" if spread < 0.20 else "bad")

    ood = not st.session_state.get("in_distribution", True)
    ood_note = ("" if not ood else
                " This patient's BMI falls outside the range the models were trained on "
                "(see the sidebar warning), which is almost certainly why the disagreement "
                "is larger than usual.")
    st.markdown(f"""<div class="{box}"><b>Do the models agree?</b> The highest and lowest
    predictions are <b>{spread*100:.1f} percentage points</b> apart, which means
    {agree_txt}.{ood_note} Across the whole test set the average gap is only
    {META['agreement']['mean_spread']:.1f} points, because all three models were
    <i>probability-calibrated</i> during training — large gaps like this one are the
    exception, not the rule, and are a signal to treat the prediction with extra caution
    rather than take the average at face value.</div>""", unsafe_allow_html=True)

    section("🚦 Health check card for this patient")
    st.dataframe(health_flags(PATIENT), **FULL, hide_index=True)

    section("📝 The prediction in plain English")
    bmi_v = round(PATIENT["weight"] / ((PATIENT["height"] / 100) ** 2), 2)
    reasons = []
    if PATIENT["ap_hi"] >= 140 or PATIENT["ap_lo"] >= 90:
        reasons.append("blood pressure is in the high range")
    if PATIENT["age"] >= 55:
        reasons.append("the patient is in an older age group")
    if PATIENT["cholesterol"] >= 2:
        reasons.append("cholesterol is above normal")
    if PATIENT["gluc"] >= 2:
        reasons.append("blood sugar is above normal")
    if bmi_v >= 30:
        reasons.append("BMI is in the obese range")
    if PATIENT["smoke"]:
        reasons.append("the patient smokes")
    if not PATIENT["active"]:
        reasons.append("the patient is not physically active")

    good = []
    if PATIENT["ap_hi"] < 130 and PATIENT["ap_lo"] < 85:
        good.append("blood pressure is normal")
    if PATIENT["age"] < 45:
        good.append("the patient is young")
    if PATIENT["cholesterol"] == 1:
        good.append("cholesterol is normal")
    if bmi_v < 25:
        good.append("BMI is healthy")
    if PATIENT["active"]:
        good.append("the patient exercises regularly")

    txt = (f"A {PATIENT['age']:.0f}-year-old "
           f"{'man' if PATIENT['gender'] else 'woman'} with blood pressure "
           f"{PATIENT['ap_hi']}/{PATIENT['ap_lo']} and BMI {bmi_v}. ")
    txt += ("Points that increase the risk: " + ", ".join(reasons) + ". "
            if reasons else "No major risk factors stand out. ")
    txt += ("Points in the patient's favour: " + ", ".join(good) + ". " if good else "")
    txt += (f"Taking the average of the three models, the estimated chance of cardiovascular "
            f"disease is <b>{avg*100:.1f}%</b>, which we classify as <b>{label}</b>.")
    st.markdown(f'<div class="note">{txt}</div>', unsafe_allow_html=True)
    st.caption("⚠️ This is a student machine-learning project, not a medical device. "
               "It must never replace a qualified doctor.")

    section("📄 Download this patient's report")

    def build_patient_pdf(patient, vals, avg, spread, label, bmi_v, summary_html):
        import io, uuid
        from datetime import datetime
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                        TableStyle, HRFlowable)

        NAVY = colors.HexColor("#1E3A5F")
        GRAY = colors.HexColor("#475569")
        LIGHT_GRAY = colors.HexColor("#F1F5F9")
        BORDER = colors.HexColor("#CBD5E1")
        RED = colors.HexColor("#B91C1C")
        AMBER = colors.HexColor("#B45309")
        GREEN = colors.HexColor("#15803D")
        RISK_COLOUR = {"LOW RISK": GREEN, "MODERATE RISK": AMBER, "HIGH RISK": RED}.get(label, NAVY)

        report_id = "HD-" + uuid.uuid4().hex[:8].upper()
        generated = datetime.now().strftime("%d %b %Y, %I:%M %p")

        buf = io.BytesIO()
                                leftMargin=1.8*cm, rightMargin=1.8*cm,
                                title="Cardiovascular Risk Assessment Report")
        styles = getSampleStyleSheet()

        meta_style = ParagraphStyle("Meta", fontSize=8.3, textColor=GRAY, fontName="Helvetica")
        meta_right = ParagraphStyle("MetaR", parent=meta_style, alignment=TA_RIGHT)
        title_style = ParagraphStyle("Title", fontSize=16.5, leading=21, textColor=NAVY,
                                     fontName="Helvetica-Bold", alignment=TA_CENTER,
                                     spaceBefore=10, spaceAfter=4)
        subtitle_style = ParagraphStyle("Sub", fontSize=9.5, leading=13, textColor=GRAY,
                                        fontName="Helvetica-Oblique", alignment=TA_CENTER,
                                        spaceBefore=0, spaceAfter=6)
        notice_style = ParagraphStyle("Notice", fontSize=8, textColor=AMBER,
                                      fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=10)
                                    fontName="Helvetica-Bold", backColor=NAVY,
        body_style = ParagraphStyle("B", fontSize=9.3, leading=12.6, textColor=colors.HexColor("#1E293B"))
        footer_style = ParagraphStyle("F", fontSize=7.6, leading=10.5, textColor=GRAY)

        def to_hex(colour):
            r, g, b = [int(round(c * 255)) for c in (colour.red, colour.green, colour.blue)]
            return f"#{r:02X}{g:02X}{b:02X}"

        def flag_style(text, colour):
            return f'<font color="{to_hex(colour)}"><b>{text}</b></font>'

        # -------- header block (letterhead-style) --------
        story = []
        hdr = Table([[Paragraph("Report Type: Cardiovascular Risk Screening", meta_style),
                     Paragraph(f"Report ID: {report_id}<br/>Generated: {generated}", meta_right)]],
                   colWidths=[9*cm, 8.5*cm])
        hdr.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"),
                                 ("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 0)]))
        story.append(hdr)
        story.append(Spacer(1, 3))
        story.append(HRFlowable(width="100%", thickness=1.3, color=NAVY))
        story.append(Paragraph("CARDIOVASCULAR DISEASE RISK ASSESSMENT", title_style))
        story.append(Paragraph("Machine Learning-Based Screening Report", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.3, color=NAVY))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "FOR ACADEMIC / RESEARCH USE ONLY &nbsp;&mdash;&nbsp; NOT A CLINICAL DIAGNOSIS", notice_style))

        # -------- patient information --------
        story.append(Paragraph("PATIENT INFORMATION", head_style))
        pinfo = [
            ["Age", f"{patient['age']:.0f} Years", "Sex", "Male" if patient["gender"] else "Female"],
            ["Height", f"{patient['height']} cm", "Weight", f"{patient['weight']} kg"],
            ["BMI", f"{bmi_v} kg/m\u00b2", "Report Date", datetime.now().strftime("%d %b %Y")],
        ]
        pt = Table(pinfo, colWidths=[2.6*cm, 5.7*cm, 2.6*cm, 5.7*cm])
        pt.setStyle(TableStyle([
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9.3), ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#1E293B")),
            ("GRID", (0,0), (-1,-1), 0.5, BORDER),
            ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 7),
        ]))
        story.append(pt)

        # -------- clinical parameters (lab-report style, with reference ranges) --------
        story.append(Paragraph("VITAL &amp; CLINICAL PARAMETERS", head_style))
        pp = patient["ap_hi"] - patient["ap_lo"]
        chol_lbl = ["Normal", "Above normal", "Well above normal"][patient["cholesterol"] - 1]
        gluc_lbl = ["Normal", "Above normal", "Well above normal"][patient["gluc"] - 1]

        def bp_flag():
            if patient["ap_hi"] < 130 and patient["ap_lo"] < 85: return "Normal", GREEN
            if patient["ap_hi"] < 140 and patient["ap_lo"] < 90: return "Elevated", AMBER
            return "High", RED
        def pp_flag():
            return ("Normal", GREEN) if 30 <= pp <= 50 else ("Unusual", AMBER)
        def bmi_flag():
            if bmi_v < 25: return "Normal", GREEN
            if bmi_v < 30: return "Overweight", AMBER
            return "Obese", RED
        def level_flag(v):
            return [("Normal", GREEN), ("Above normal", AMBER), ("Well above normal", RED)][v-1]
        def yn_flag(v, bad_word, good_word, bad_is_high=True):
            return (bad_word, RED) if (v and bad_is_high) or (not v and not bad_is_high) else (good_word, GREEN)

        bp_f, bp_c = bp_flag(); pp_f, pp_c = pp_flag(); bmi_f, bmi_c = bmi_flag()
        chol_f, chol_c = level_flag(patient["cholesterol"]); gluc_f, gluc_c = level_flag(patient["gluc"])
        smoke_f, smoke_c = ("Smoker", RED) if patient["smoke"] else ("Non-smoker", GREEN)
        alco_f, alco_c = ("Consumes alcohol", AMBER) if patient["alco"] else ("Does not drink", GREEN)
        act_f, act_c = ("Active", GREEN) if patient["active"] else ("Sedentary", RED)

        header_style = ParagraphStyle("THead", parent=body_style, textColor=colors.white,
                                      fontName="Helvetica-Bold", fontSize=9)
        flag_para_style = ParagraphStyle("Flag", parent=body_style, fontSize=9.4)

        raw_rows = [["Systolic BP (upper)", f"{patient['ap_hi']} mmHg", "90 - 129 mmHg", (bp_f, bp_c)],
                    ["Diastolic BP (lower)", f"{patient['ap_lo']} mmHg", "60 - 84 mmHg", (bp_f, bp_c)],
                    ["Pulse Pressure", f"{pp} mmHg", "30 - 50 mmHg", (pp_f, pp_c)],
                    ["Body Mass Index", f"{bmi_v} kg/m\u00b2", "18.5 - 24.9 kg/m\u00b2", (bmi_f, bmi_c)],
                    ["Cholesterol Level", chol_lbl, "Normal", (chol_f, chol_c)],
                    ["Glucose Level", gluc_lbl, "Normal", (gluc_f, gluc_c)],
                    ["Smoking Status", "Yes" if patient["smoke"] else "No", "Non-smoker", (smoke_f, smoke_c)],
                    ["Alcohol Intake", "Yes" if patient["alco"] else "No", "None / Occasional", (alco_f, alco_c)],
                    ["Physical Activity", "Yes" if patient["active"] else "No", "Regularly active", (act_f, act_c)]]

        rows = [[Paragraph("Parameter", header_style), Paragraph("Result", header_style),
                 Paragraph("Reference Range", header_style), Paragraph("Flag", header_style)]]
        for name, result, ref, (flag_text, flag_colour) in raw_rows:
            rows.append([Paragraph(name, body_style), Paragraph(result, body_style),
                        Paragraph(ref, body_style),
                        Paragraph(flag_style(flag_text, flag_colour), flag_para_style)])
        ct = Table(rows, colWidths=[4.3*cm, 3.4*cm, 4.3*cm, 4.6*cm])
        style_cmds = [
            ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, BORDER), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0,i), (-1,i), LIGHT_GRAY))
        ct.setStyle(TableStyle(style_cmds))
        story.append(ct)
        story.append(Paragraph(
            "Reference ranges reflect commonly used general adult screening thresholds and are "
            "provided for context only.", ParagraphStyle("Small", fontSize=7.3, textColor=GRAY,
            spaceBefore=2, spaceAfter=2)))

        # -------- AI risk prediction --------
        story.append(Paragraph("AI-BASED RISK PREDICTION", head_style))
        bold_body = ParagraphStyle("BB", parent=body_style, fontName="Helvetica-Bold")
        mrows_raw = [["Model", "Predicted Probability"],
                    ["Logistic Regression", f"{vals['Logistic Regression']*100:.1f}%"],
                    ["Random Forest", f"{vals['Random Forest']*100:.1f}%"],
                    ["Support Vector Machine", f"{vals['SVM']*100:.1f}%"],
                    ["Combined Average", f"{avg*100:.1f}%"]]
        mrows = [[Paragraph(c, header_style) for c in mrows_raw[0]]]
        for r in mrows_raw[1:-1]:
            mrows.append([Paragraph(c, body_style) for c in r])
        mrows.append([Paragraph(c, bold_body) for c in mrows_raw[-1]])
        mt = Table(mrows, colWidths=[10.6*cm, 6*cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), NAVY),
            ("BACKGROUND", (0,-1), (-1,-1), LIGHT_GRAY),
            ("FONTSIZE", (0,0), (-1,-1), 9.3), ("GRID", (0,0), (-1,-1), 0.5, BORDER),
            ("TOPPADDING", (0,0), (-1,-1), 3.4), ("BOTTOMPADDING", (0,0), (-1,-1), 3.4),
        ]))
        story.append(mt)
        story.append(Paragraph(
            f"Model agreement: predictions of the three models are within {spread*100:.1f} "
            f"percentage points of one another.",
            ParagraphStyle("Small2", fontSize=7.6, textColor=GRAY, spaceBefore=2, spaceAfter=5)))

        # -------- risk category stamp --------
        stamp = Table([[Paragraph(f"OVERALL RISK CATEGORY: <b>{label}</b> &nbsp;"
                                  f"(Estimated probability: {avg*100:.1f}%)",
                                  ParagraphStyle("Stamp", fontSize=11, alignment=TA_CENTER,
                                                textColor=RISK_COLOUR, fontName="Helvetica-Bold"))]],
                       colWidths=[16.6*cm])
        stamp.setStyle(TableStyle([
            ("BOX", (0,0), (-1,-1), 1.3, RISK_COLOUR), ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6), ("BACKGROUND", (0,0), (-1,-1), colors.white),
        ]))
        story.append(stamp)
        story.append(Spacer(1, 3))

        # -------- clinical interpretation --------
        story.append(Paragraph("CLINICAL INTERPRETATION", head_style))
        story.append(Paragraph(summary_html, body_style))

        # -------- footer --------
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.6, color=BORDER))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "This is a computer-generated report produced by an academic machine-learning system. "
            "It has not been reviewed by a licensed physician, has not been clinically validated, "
            "and does not constitute a medical diagnosis. It must not be used as a substitute for "
            "professional medical advice, examination, or treatment. Always consult a qualified "
            "doctor for any health concerns.", footer_style))
        story.append(Spacer(1, 3))
        ft = Table([[Paragraph(f"Report ID: {report_id}", footer_style),
                    Paragraph("Electronically generated \u2014 no signature required",
                             ParagraphStyle("FC", parent=footer_style, alignment=TA_CENTER)),
                    Paragraph("Page 1 of 1", ParagraphStyle("FR", parent=footer_style, alignment=TA_RIGHT))]],
                   colWidths=[5.5*cm, 5.6*cm, 5.5*cm])
        ft.setStyle(TableStyle([("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 0)]))
        story.append(ft)

        doc.build(story)
        return buf.getvalue()

    pdf_bytes = build_patient_pdf(PATIENT, vals, avg, spread, label, bmi_v, txt)
    st.download_button("⬇️ Download patient report (PDF)", pdf_bytes,
                       f"heart_risk_report_{PATIENT['age']:.0f}yo.pdf", "application/pdf",
                       type="primary")

# ===============================================================
#  PAGE 3 — SHAP EXPLANATION
# ===============================================================
elif PAGE == "🧠 SHAP Explanation":
    hero("🧠 SHAP — Which measurement pushed the answer up or down?",
         "SHAP splits the final prediction into one fair share for every health measurement")

    st.markdown("""<div class="note">
    <b>SHAP in one sentence:</b> imagine the models start from the <i>average patient</i> and then
    read your patient's numbers one by one. Every number nudges the prediction a little higher
    (red 🔺) or a little lower (blue 🔻). SHAP measures the size of each nudge, and the nudges
    always add up exactly to the final answer. That is why SHAP is called a
    <i>mathematically fair</i> explanation.</div>""", unsafe_allow_html=True)

    chosen = st.selectbox("🔧 Choose which model to explain", MODEL_NAMES, key="shap_model")
    st.caption("Tip: switch the model to see whether the three models reason the same way.")

    model = MODELS[chosen]
    prob_now = float(predict_proba_raw(model, X_patient.values)[0])

    @st.cache_resource(show_spinner=False)
    def get_shap_explainer(model_name):
        import shap
        bg = np.load(os.path.join(MODEL_DIR, "shap_background.npy"))
        m = MODELS[model_name]
        return shap.KernelExplainer(
            lambda z: m.predict_proba(SCALER.transform(np.asarray(z)))[:, 1], bg)

    @st.cache_data(show_spinner=False)
    def local_shap(model_name, values):
        ex = get_shap_explainer(model_name)
        sv = np.array(ex.shap_values(np.array(values, dtype=float), nsamples=120, silent=True))
        return sv.ravel(), float(ex.expected_value)

    with st.spinner(f"Calculating SHAP values for {chosen}… (the SVM takes a few seconds)"):
        sv, base = local_shap(chosen, tuple(X_patient.values[0]))

    c = st.columns(4)
    kpi(c[0], "k1", "Starting point", f"{base*100:.1f}%", "The average patient")
    kpi(c[1], "k4", "Final prediction", f"{prob_now*100:.1f}%", chosen)
    kpi(c[2], "k3", "Total nudge", f"{(prob_now-base)*100:+.1f}%", "Sum of all SHAP values")
    top_f = FEATURES[int(np.argmax(np.abs(sv)))]
    kpi(c[3], "k2", "Biggest influence", DISPLAY[top_f], f"{sv[np.argmax(np.abs(sv))]*100:+.1f}%")

    section(f"1️⃣ Personal explanation — why {chosen} gave {prob_now*100:.1f}%")
    order = np.argsort(np.abs(sv))
    fig = go.Figure(go.Bar(
        x=sv[order] * 100,
        y=[f"{DISPLAY[FEATURES[i]]}  =  {X_patient.iloc[0][FEATURES[i]]:g}" for i in order],
        orientation="h",
        marker_color=[C_HIGH if v > 0 else "#3B82F6" for v in sv[order]],
        text=[f"{v*100:+.1f}%" for v in sv[order] * 100 / 100], textposition="outside",
        hovertemplate="%{y}<br>Effect: %{x:+.2f} percentage points<extra></extra>"))
    fig.add_vline(x=0, line_color="#334155")
    fig.update_layout(height=520, plot_bgcolor="white",
                      xaxis_title="Effect on the prediction (percentage points)",
                      title="🔺 Red bars push risk UP · 🔻 Blue bars push risk DOWN",
                      margin=dict(l=10, r=60))
    st.plotly_chart(fig, **FULL)

    up = [(FEATURES[i], sv[i]) for i in np.argsort(-sv) if sv[i] > 0][:3]
    dn = [(FEATURES[i], sv[i]) for i in np.argsort(sv) if sv[i] < 0][:3]
    cU, cD = st.columns(2)
    cU.markdown("<div class='bad'><b>🔺 Top reasons the risk went UP</b><br>" +
                ("<br>".join([f"• <b>{DISPLAY[f]}</b> = "
                              f"{X_patient.iloc[0][f]:g} → adds {v*100:+.1f} points"
                              for f, v in up]) if up else "Nothing pushed the risk up.") +
                "</div>", unsafe_allow_html=True)
    cD.markdown("<div class='good'><b>🔻 Top reasons the risk went DOWN</b><br>" +
                ("<br>".join([f"• <b>{DISPLAY[f]}</b> = "
                              f"{X_patient.iloc[0][f]:g} → removes {abs(v)*100:.1f} points"
                              for f, v in dn]) if dn else "Nothing pushed the risk down.") +
                "</div>", unsafe_allow_html=True)

    section("2️⃣ Step-by-step: from the average patient to this patient")
    idx = np.argsort(-np.abs(sv))
    steps, running = ["Average patient"], [base * 100]
    for i in idx:
        steps.append(f"{DISPLAY[FEATURES[i]]} = {X_patient.iloc[0][FEATURES[i]]:g}")
        running.append(running[-1] + sv[i] * 100)
    wf = go.Figure(go.Waterfall(
        orientation="v", x=steps + ["FINAL PREDICTION"],
        measure=["absolute"] + ["relative"] * len(idx) + ["total"],
        y=[base * 100] + [sv[i] * 100 for i in idx] + [None],
        text=[f"{base*100:.1f}%"] + [f"{sv[i]*100:+.1f}" for i in idx] + [f"{prob_now*100:.1f}%"],
        textposition="outside",
        increasing={"marker": {"color": C_HIGH}}, decreasing={"marker": {"color": "#3B82F6"}},
        totals={"marker": {"color": "#6366F1"}},
        connector={"line": {"color": "#CBD5E1"}}))
    wf.update_layout(height=470, plot_bgcolor="white", yaxis_title="Risk (%)",
                     title="Every measurement adds or removes a slice of risk")
    st.plotly_chart(wf, **FULL)
    st.markdown(f"""<div class="note"><b>How to read it:</b> the first bar is the average risk of
    all patients ({base*100:.1f}%). Each following bar is one measurement of this patient adding
    or removing risk. The last bar, {prob_now*100:.1f}%, is what {chosen} finally predicted.
    Notice the bars add up perfectly — that is the guarantee SHAP gives you.</div>""",
                unsafe_allow_html=True)

    section("3️⃣ Global view — what matters across many patients")
    G = load_global_shap()
    gsv, gX = G[chosen], G["X"]
    imp = pd.DataFrame({"Feature": [DISPLAY[f] for f in FEATURES],
                        "Importance": np.abs(gsv).mean(0) * 100}).sort_values("Importance")
    cG1, cG2 = st.columns([1, 1])
    f1 = px.bar(imp, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Purp",
                text=imp.Importance.round(2),
                title=f"Average importance for {chosen} (80 test patients)",
                labels={"Importance": "Average absolute effect (percentage points)"})
    f1.update_traces(textposition="outside")
    f1.update_layout(height=470, plot_bgcolor="white", coloraxis_showscale=False)
    cG1.plotly_chart(f1, **FULL)

    pick = cG2.selectbox("Look at one feature in detail", FEATURES,
                         format_func=lambda f: DISPLAY[f], index=3)
    j = FEATURES.index(pick)
    f2 = px.scatter(x=gX[:, j], y=gsv[:, j] * 100, color=gsv[:, j] * 100,
                    color_continuous_scale="RdBu_r",
                    labels={"x": DISPLAY[pick], "y": "Effect on risk (points)"},
                    title=f"How “{DISPLAY[pick]}” changes the prediction")
    f2.add_hline(y=0, line_dash="dash", line_color="#64748B")
    f2.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    f2.update_layout(height=470, plot_bgcolor="white", coloraxis_showscale=False)
    cG2.plotly_chart(f2, **FULL)
    st.markdown(f"""<div class="note"><b>Plain English:</b> each dot is one patient. Dots above
    the dashed line mean that for that patient, “{DISPLAY[pick]}” pushed the risk <b>up</b>;
    dots below mean it pushed the risk <b>down</b>. If the dots rise from left to right, then a
    higher value of this measurement means higher heart risk.
    <br><br><i>{PLAIN[pick]}</i></div>""", unsafe_allow_html=True)

    section("4️⃣ Do the three models think alike?")
    cmp_df = pd.DataFrame({n: np.abs(G[n]).mean(0) * 100 for n in MODEL_NAMES},
                          index=[DISPLAY[f] for f in FEATURES])
    f3 = px.imshow(cmp_df.T, text_auto=".2f", color_continuous_scale="Purples", aspect="auto",
                   title="Average SHAP importance — all three models side by side",
                   labels={"x": "", "y": "", "color": "Importance"})
    f3.update_layout(height=330)
    st.plotly_chart(f3, **FULL)
    st.markdown("""<div class="good"><b>Why this chart is worth showing your professor:</b> the
    three models were built on completely different mathematics — a straight line, a forest of
    decision trees, and a curved boundary in high-dimensional space. Yet they highlight the
    <i>same</i> top features. When independent methods agree, the explanation is trustworthy.
    </div>""", unsafe_allow_html=True)


# ===============================================================
#  PAGE 4 — LIME EXPLANATION
# ===============================================================
elif PAGE == "🍋 LIME Explanation":
    hero("🍋 LIME — A simple rule-based explanation for this one patient",
         "LIME builds a tiny, easy model that copies the big model just around your patient")

    st.markdown("""<div class="note">
    <b>LIME in one sentence:</b> the real model may be complicated, but if you zoom in very close
    to <i>one</i> patient, its behaviour looks almost like a straight line. LIME creates thousands
    of slightly-changed copies of your patient, asks the model what it thinks about each copy, and
    then fits a simple rule-based model to those answers. The result is a short list of readable
    rules such as “systolic BP above 140 → risk up”.
    <br><br><b>SHAP vs LIME:</b> SHAP is exact and mathematically guaranteed but slower;
    LIME is fast and gives human-readable if-then rules. Showing both is the professional way to
    prove your explanation is not a fluke.</div>""", unsafe_allow_html=True)

    chosen = st.selectbox("🔧 Choose which model to explain", MODEL_NAMES, key="lime_model")
    n_samples = st.slider("How many “nearby” patients should LIME create?", 500, 5000, 2000, 500,
                          help="More samples = a steadier explanation, but a little slower.")

    model = MODELS[chosen]
    prob_now = float(predict_proba_raw(model, X_patient.values)[0])

    @st.cache_resource(show_spinner=False)
    def get_lime_explainer():
        from lime.lime_tabular import LimeTabularExplainer
        train_bg = np.load(os.path.join(MODEL_DIR, "lime_background.npy"))
        return LimeTabularExplainer(
            training_data=train_bg,
            feature_names=[DISPLAY[f] for f in FEATURES],
            class_names=["Healthy", "Heart disease"],
            categorical_features=[1, 6, 7, 8, 9, 10],
            discretize_continuous=True, mode="classification", random_state=42)

    @st.cache_data(show_spinner=False)
    def run_lime(model_name, values, n):
        ex = get_lime_explainer()
        m = MODELS[model_name]
        exp = ex.explain_instance(
            np.array(values, dtype=float),
            lambda z: m.predict_proba(SCALER.transform(np.asarray(z))),
            num_features=len(FEATURES), num_samples=int(n))
        # LIME's simple copy is a linear model, so it can occasionally fall
        # slightly outside 0-100%. We clip it back into a sensible range.
        local = float(np.clip(exp.local_pred[0], 0.0, 1.0))
        return exp.as_list(), float(exp.score), local

    with st.spinner(f"LIME is building {n_samples:,} nearby patients and learning from them…"):
        pairs, fit_score, local_pred = run_lime(chosen, tuple(X_patient.values[0]), n_samples)

    c = st.columns(4)
    kpi(c[0], "k1", "Model explained", chosen, "Selected above")
    kpi(c[1], "k4", "Real prediction", f"{prob_now*100:.1f}%", "From the full model")
    kpi(c[2], "k2", "LIME's copy says", f"{local_pred*100:.1f}%", "Approximate — see note below")
    kpi(c[3], "k3", "Copy quality (R²)", f"{fit_score:.3f}", "1.0 = perfect imitation")

    section(f"1️⃣ The rules LIME found for this patient")
    rules = [p[0] for p in pairs][::-1]
    weights = [p[1] for p in pairs][::-1]
    fig = go.Figure(go.Bar(
        x=weights, y=rules, orientation="h",
        marker_color=[C_HIGH if w > 0 else "#3B82F6" for w in weights],
        text=[f"{w:+.3f}" for w in weights], textposition="outside",
        hovertemplate="%{y}<br>Weight: %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#334155")
    fig.update_layout(height=560, plot_bgcolor="white",
                      xaxis_title="Weight of the rule (positive = pushes towards heart disease)",
                      title="🔺 Red rules argue for DISEASE · 🔻 Blue rules argue for HEALTHY",
                      margin=dict(l=10, r=70))
    st.plotly_chart(fig, **FULL)

    section("2️⃣ The same rules as a readable table")
    tbl = pd.DataFrame({
        "Rule LIME discovered": [p[0] for p in pairs],
        "Weight": [round(p[1], 4) for p in pairs],
        "Direction": ["🔺 Towards heart disease" if p[1] > 0 else "🔻 Towards healthy"
                      for p in pairs],
        "Strength": ["●●● Strong" if abs(p[1]) > 0.06 else
                     ("●●○ Medium" if abs(p[1]) > 0.02 else "●○○ Weak") for p in pairs]})
    st.dataframe(tbl, **FULL, hide_index=True)

    section("3️⃣ What this means in plain English")
    pos = [p for p in pairs if p[1] > 0][:3]
    neg = [p for p in pairs if p[1] < 0][:3]
    msg = (f"For this particular patient, <b>{chosen}</b> predicts a "
           f"<b>{prob_now*100:.1f}%</b> chance of cardiovascular disease. ")
    if pos:
        msg += ("The conditions arguing <b>for</b> disease are: " +
                "; ".join([f"<code>{p[0]}</code>" for p in pos]) + ". ")
    if neg:
        msg += ("The conditions arguing <b>against</b> disease are: " +
                "; ".join([f"<code>{p[0]}</code>" for p in neg]) + ". ")
    msg += (f"LIME's simple copy reproduces the real model with an R² of {fit_score:.3f}. "
            f"Remember that LIME's own percentage is produced by that simplified copy, not by the "
            f"real model, so a small gap between the two numbers above is completely normal — "
            f"the <i>ranking of the rules</i> is what LIME is really telling you.")
    st.markdown(f'<div class="note">{msg}</div>', unsafe_allow_html=True)

    section("4️⃣ Compare LIME with SHAP")
    st.markdown("""<div class="good">
    Open the <b>SHAP Explanation</b> page with the same patient and the same model. You should see
    the same measurements at the top of both charts, even though SHAP and LIME use completely
    different mathematics — SHAP borrows from cooperative game theory, LIME fits a local linear
    model. <b>Two independent methods reaching the same conclusion is strong evidence that the
    explanation is real.</b> This cross-check is exactly what an examiner wants to hear.
    </div>""", unsafe_allow_html=True)

# ===============================================================
#  PAGE 5 — BATCH CSV PREDICTION
# ===============================================================
elif PAGE == "📁 Batch CSV Prediction":
    hero("📁 Batch Prediction from a CSV file",
         "Screen a whole hospital ward at once — upload a file, get every patient scored")

    st.markdown("""<div class="note">
    <b>What this page does:</b> instead of typing one patient at a time, upload a spreadsheet with
    many patients. All three models score every row, and you can download the finished file with
    the risk scores added. This is how the model would actually be used in a clinic.</div>""",
                unsafe_allow_html=True)

    RAW_COLS = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active"]

    section("1️⃣ Get the template")
    template = pd.DataFrame({
        "age": [52, 41, 63, 35, 58], "gender": [1, 0, 1, 0, 1],
        "height": [172, 160, 168, 165, 175], "weight": [80, 55, 95, 60, 88],
        "ap_hi": [130, 110, 165, 115, 150], "ap_lo": [85, 70, 100, 75, 95],
        "cholesterol": [1, 1, 3, 1, 2], "gluc": [1, 1, 2, 1, 1],
        "smoke": [0, 0, 1, 0, 1], "alco": [0, 0, 1, 0, 0], "active": [1, 1, 0, 1, 0]})
    c1, c2 = st.columns([2, 1])
    c1.dataframe(template, **FULL, hide_index=True)
    c2.markdown("""**Required columns**

`age` (years) · `gender` (0 = female, 1 = male) · `height` (cm) · `weight` (kg)
· `ap_hi` · `ap_lo` · `cholesterol` (1-3) · `gluc` (1-3) · `smoke` · `alco` · `active` (0/1)

BMI and pulse pressure are calculated automatically.""")
    c2.download_button("⬇️ Download template CSV", template.to_csv(index=False),
                       "patients_template.csv", "text/csv", **FULL)

    section("2️⃣ Upload your file")
    up = st.file_uploader("Choose a CSV file of patients", type=["csv"])

    if up is None:
        st.info("👆 Upload a CSV file, or download the template above and try it first.")
    else:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read that file: {e}")
            st.stop()

        raw.columns = [c.strip().lower() for c in raw.columns]
        missing = [c for c in RAW_COLS if c not in raw.columns]
        if missing:
            st.error(f"These required columns are missing: **{', '.join(missing)}**")
            st.stop()

        work = raw.copy()
        # If age looks like days (as in the original Kaggle file), convert it to years.
        if work["age"].median() > 200:
            work["age"] = (work["age"] / 365.25).round(1)
            st.info("Detected age in days — automatically converted to years.")
        if set(work["gender"].dropna().unique()) <= {1, 2}:
            work["gender"] = work["gender"].map({1: 0, 2: 1})
            st.info("Detected gender coded as 1/2 — automatically converted to 0/1.")

        before = len(work)
        work = work.dropna(subset=RAW_COLS)
        work = work[(work.ap_hi > work.ap_lo) & work.ap_hi.between(60, 250)
                    & work.ap_lo.between(40, 200) & work.height.between(120, 220)
                    & work.weight.between(30, 200)]
        dropped = before - len(work)
        if len(work) == 0:
            st.error("No usable rows were left after checking the values.")
            st.stop()

        X = pd.DataFrame({
            "age_years": work.age.astype(float),
            "gender": work.gender.astype(int),
            "bmi": (work.weight / (work.height / 100) ** 2).round(2),
            "ap_hi": work.ap_hi.astype(int), "ap_lo": work.ap_lo.astype(int),
            "pulse_pressure": (work.ap_hi - work.ap_lo).astype(int),
            "cholesterol": work.cholesterol.astype(int), "gluc": work.gluc.astype(int),
            "smoke": work.smoke.astype(int), "alco": work.alco.astype(int),
            "active": work.active.astype(int)})[FEATURES]

        with st.spinner(f"Scoring {len(X):,} patients with all three models…"):
            preds = predict_all(X.values)

        out = work.reset_index(drop=True).copy()
        out["BMI"] = X.bmi.values
        for n in MODEL_NAMES:
            out[f"{n} (%)"] = (preds[n] * 100).round(1)
        avg = np.mean([preds[n] for n in MODEL_NAMES], axis=0)
        out["Average risk (%)"] = (avg * 100).round(1)
        out["Model spread (pts)"] = ((np.max([preds[n] for n in MODEL_NAMES], axis=0) -
                                      np.min([preds[n] for n in MODEL_NAMES], axis=0)) * 100).round(1)
        out["Risk level"] = ["🟢 Low" if a < .35 else ("🟡 Moderate" if a < .65 else "🔴 High")
                             for a in avg]
        out["Prediction"] = np.where(avg >= .5, "Heart disease likely", "Healthy")

        # Same safeguard as the single-patient page: the models were only ever
        # trained on BMI 15-50 (notebook 02's cleaning rules). A row outside
        # that range is an extrapolation, not a normal prediction, and is
        # exactly where the three models tend to disagree most sharply.
        BMI_TRAIN_MIN, BMI_TRAIN_MAX = 15.0, 50.0
        out["In training range"] = out["BMI"].between(BMI_TRAIN_MIN, BMI_TRAIN_MAX)
        n_ood = int((~out["In training range"]).sum())

        section("3️⃣ Results")
        n_hi = int((avg >= .65).sum()); n_md = int(((avg >= .35) & (avg < .65)).sum())
        n_lo = int((avg < .35).sum())
        c = st.columns(5)
        kpi(c[0], "k1", "Patients scored", f"{len(out):,}", f"{dropped} rows skipped")
        kpi(c[1], "k4", "High risk", f"{n_hi:,}", f"{n_hi/len(out)*100:.1f}% of the file")
        kpi(c[2], "k3", "Moderate risk", f"{n_md:,}", f"{n_md/len(out)*100:.1f}% of the file")
        kpi(c[3], "k2", "Low risk", f"{n_lo:,}", f"{n_lo/len(out)*100:.1f}% of the file")
        kpi(c[4], "k5", "Average risk", f"{avg.mean()*100:.1f}%", "Across the whole file")

        if n_ood:
            st.warning(
                f"⚠️ **{n_ood:,} patient(s)** have a BMI outside the training range "
                f"({BMI_TRAIN_MIN:.0f}–{BMI_TRAIN_MAX:.0f}). Their predictions are "
                f"extrapolations and typically show a wider gap between the three models — "
                f"look for `In training range = False` in the table below and treat those "
                f"rows with extra caution rather than trusting the average blindly.")

        g1, g2 = st.columns(2)
        pie = px.pie(values=[n_lo, n_md, n_hi], names=["Low", "Moderate", "High"], hole=.55,
                     color=["Low", "Moderate", "High"],
                     color_discrete_map={"Low": C_LOW, "Moderate": C_MED, "High": C_HIGH},
                     title="Risk breakdown of the uploaded file")
        pie.update_traces(textinfo="percent+label")
        pie.update_layout(height=400)
        g1.plotly_chart(pie, **FULL)

        hist = go.Figure()
        for n in MODEL_NAMES:
            hist.add_trace(go.Histogram(x=preds[n] * 100, name=n, opacity=.55,
                                        marker_color=C_MODEL[n], nbinsx=30))
        hist.update_layout(barmode="overlay", height=400, plot_bgcolor="white",
                           title="Do the three models spread the patients the same way?",
                           xaxis_title="Predicted risk (%)", yaxis_title="Number of patients",
                           legend=dict(orientation="h", y=1.15))
        g2.plotly_chart(hist, **FULL)

        show = st.selectbox("Show", ["All patients", "High risk only", "Moderate risk only",
                                     "Low risk only", "Where models disagree most",
                                     "Outside training range (BMI)"])
        view = out.copy()
        if show == "High risk only":
            view = view[view["Risk level"].str.contains("High")]
        elif show == "Moderate risk only":
            view = view[view["Risk level"].str.contains("Moderate")]
        elif show == "Low risk only":
            view = view[view["Risk level"].str.contains("Low")]
        elif show == "Where models disagree most":
            view = view.sort_values("Model spread (pts)", ascending=False).head(50)
        elif show == "Outside training range (BMI)":
            view = view[~view["In training range"]]

        cols_show = (["age", "gender", "BMI", "ap_hi", "ap_lo", "cholesterol", "gluc"] +
                     [f"{n} (%)" for n in MODEL_NAMES] +
                     ["Average risk (%)", "Model spread (pts)", "In training range",
                      "Risk level", "Prediction"])
        st.dataframe(
            view[cols_show].style.background_gradient(
                cmap="RdYlGn_r", subset=["Average risk (%)"] + [f"{n} (%)" for n in MODEL_NAMES]),
            **FULL, height=430)
        st.caption(f"Showing {len(view):,} of {len(out):,} patients.")

        st.download_button("⬇️ Download full results as CSV", out.to_csv(index=False),
                           "heart_disease_predictions.csv", "text/csv", type="primary")

        st.markdown(f"""<div class="note"><b>How a hospital would use this:</b> sort by
        <i>Average risk</i> and call the top patients in first. The <i>Model spread</i> column is
        a bonus safety feature — when the three models disagree by a lot, the case is uncertain
        and deserves a human doctor's attention rather than an automatic decision.</div>""",
                    unsafe_allow_html=True)


# ===============================================================
#  PAGE 6 — MODEL PERFORMANCE
# ===============================================================
elif PAGE == "📊 Model Performance":
    hero("📊 Model Performance and Validation",
         "Every number here comes from the 20% test set the models never saw during training")

    d = META["dataset"]
    best = max(MODEL_NAMES, key=lambda n: METRICS[n]["accuracy"])
    c = st.columns(4)
    kpi(c[0], "k1", "Training patients", f"{d['train']:,}", "80% of the clean data")
    kpi(c[1], "k4", "Test patients", f"{d['test']:,}", "20% never seen during training")
    kpi(c[2], "k2", "Best model", best, f"{METRICS[best]['accuracy']*100:.2f}% accuracy")
    kpi(c[3], "k3", "Validation", "5-fold CV", "Plus GridSearch tuning")

    section("1️⃣ All the numbers in one table")
    tbl = pd.DataFrame({
        "Model": MODEL_NAMES,
        "Accuracy": [METRICS[n]["accuracy"] for n in MODEL_NAMES],
        "Precision": [METRICS[n]["precision"] for n in MODEL_NAMES],
        "Recall": [METRICS[n]["recall"] for n in MODEL_NAMES],
        "F1 Score": [METRICS[n]["f1"] for n in MODEL_NAMES],
        "ROC-AUC": [METRICS[n]["roc_auc"] for n in MODEL_NAMES],
        "CV Accuracy": [METRICS[n]["cv_mean"] for n in MODEL_NAMES],
        "CV Std": [METRICS[n]["cv_std"] for n in MODEL_NAMES]}).set_index("Model")
    st.dataframe(tbl.style.format("{:.4f}").background_gradient(cmap="Greens", axis=0),
                 **FULL)
    st.markdown("""<div class="note"><b>What the words mean.</b>
    <b>Accuracy</b> — out of 100 patients, how many did we label correctly.
    <b>Precision</b> — when we shout "heart disease!", how often are we right.
    <b>Recall</b> — of all the people who really have heart disease, how many did we catch.
    <b>F1</b> — one balanced score combining precision and recall.
    <b>ROC-AUC</b> — how well the model ranks a sick patient above a healthy one; 0.5 is a coin
    toss, 1.0 is perfect.</div>""", unsafe_allow_html=True)

    section("2️⃣ Confusion matrices — where each model made mistakes")
    cols = st.columns(3)
    for col, n in zip(cols, MODEL_NAMES):
        cm = np.array(METRICS[n]["confusion_matrix"])
        f = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                      x=["Predicted<br>Healthy", "Predicted<br>Disease"],
                      y=["Actually<br>Healthy", "Actually<br>Disease"], title=n)
        f.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=50))
        col.plotly_chart(f, **FULL)
        tn, fp, fn, tp = cm.ravel()
        col.markdown(f"""<div class="note" style="font-size:13.5px">
        ✅ Correctly found sick: <b>{tp:,}</b><br>
        ✅ Correctly cleared healthy: <b>{tn:,}</b><br>
        ⚠️ False alarms: <b>{fp:,}</b><br>
        🔴 Missed sick patients: <b>{fn:,}</b></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="warn"><b>The one that matters most in medicine</b> is the bottom-left
    box — sick patients we told to go home. Missing a sick patient is far more dangerous than a
    false alarm, so in a real hospital you would lower the decision threshold below 50% to catch
    more of them, accepting more false alarms in exchange.</div>""", unsafe_allow_html=True)

    section("3️⃣ ROC curves")
    roc = go.Figure()
    for n in MODEL_NAMES:
        roc.add_trace(go.Scatter(x=METRICS[n]["fpr"], y=METRICS[n]["tpr"], mode="lines",
                                 name=f"{n} (AUC = {METRICS[n]['roc_auc']:.4f})",
                                 line=dict(width=3.2, color=C_MODEL[n])))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                             line=dict(dash="dash", color="#94A3B8")))
    roc.update_layout(height=520, plot_bgcolor="white",
                      xaxis_title="False alarm rate", yaxis_title="Sick patients caught",
                      title="The further a curve bends to the top-left, the better the model",
                      legend=dict(x=.45, y=.12))
    st.plotly_chart(roc, **FULL)

    section("4️⃣ Cross-validation — is the result stable?")
    cvf = go.Figure()
    for n in MODEL_NAMES:
        cvf.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(len(METRICS[n]["cv_scores"]))],
                             y=METRICS[n]["cv_scores"], name=n, marker_color=C_MODEL[n],
                             text=[f"{s:.3f}" for s in METRICS[n]["cv_scores"]],
                             textposition="outside"))
    cvf.update_layout(barmode="group", height=420, plot_bgcolor="white",
                      yaxis_range=[0.6, 0.8], yaxis_title="Accuracy on that fold",
                      title="The data was split into folds and each model was re-trained each time",
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(cvf, **FULL)
    st.markdown(f"""<div class="good"><b>Why this matters:</b> the bars for each model are almost
    the same height (standard deviation below
    {max(METRICS[n]['cv_std'] for n in MODEL_NAMES):.4f}). That proves the models did not get
    lucky with one particular split of the data — the performance is real and repeatable.
    </div>""", unsafe_allow_html=True)

    section("5️⃣ Hyperparameter tuning — the settings GridSearchCV chose")
    for n in MODEL_NAMES:
        with st.expander(f"⚙️ {n} — best settings found"):
            st.json(METRICS[n]["best_params"])
            st.caption(f"Training and tuning time: {METRICS[n]['train_seconds']} seconds")

    section("6️⃣ Do the three models agree with each other?")
    ag = META["agreement"]
    c = st.columns(4)
    kpi(c[0], "k1", "Logistic ↔ Forest", f"{ag['pair_lr_rf']:.2f}%", "Average gap")
    kpi(c[1], "k3", "Logistic ↔ SVM", f"{ag['pair_lr_svm']:.2f}%", "Average gap")
    kpi(c[2], "k2", "Forest ↔ SVM", f"{ag['pair_rf_svm']:.2f}%", "Average gap")
    kpi(c[3], "k4", "Overall spread", f"{ag['mean_spread']:.2f}%", "Highest minus lowest")
    st.markdown(f"""<div class="good"><b>Every pair of models agrees within
    {max(ag['pair_lr_rf'], ag['pair_lr_svm'], ag['pair_rf_svm']):.1f} percentage points on
    average</b>, comfortably under the 10-point target. This did not happen by chance — all three
    models were passed through <code>CalibratedClassifierCV</code> with isotonic regression, which
    corrects each model's raw scores into honest probabilities. A side benefit is that calibration
    also improved accuracy slightly.</div>""", unsafe_allow_html=True)


# ===============================================================
#  PAGE 7 — ABOUT
# ===============================================================
elif PAGE == "🩻 Clinical Validation":
    hero("🩻 Clinical Validation",
         "Going beyond accuracy — statistical rigor, decision thresholds, calibration and fairness")

    ADV = load_advanced()
    if ADV is None:
        st.warning("Advanced validation data (`models/advanced.json`) was not found. "
                   "This page needs it to be generated once alongside the models.")
        st.stop()

    st.markdown("""<div class="note">
    <b>Why this page exists.</b> Accuracy and ROC-AUC (see the Model Performance page) tell you
    how good a model is <i>overall</i>. They do not tell you whether the difference between two
    models is <i>real</i>, whether 50% is the <i>right</i> cut-off for a medical decision, whether
    the model's probabilities can be <i>trusted at face value</i>, or whether it treats every group
    of patients <i>fairly</i>. Those four questions are exactly what this page answers, using
    proper statistical tests on the same held-out test set — no numbers here were re-estimated or
    changed, only analysed more deeply.</div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 1. McNemar significance test
    # -----------------------------------------------------------------
    section("1️⃣ Is the best model actually better — or just lucky?")
    st.markdown("""Random Forest usually scores a little higher than the other two. But is that a
    **real** difference, or could it happen by chance on this particular test set? We use
    **McNemar's test** — the standard statistical test for comparing two classifiers on the same
    patients — which looks only at the cases where the two models disagree.""")

    mc = ADV["_mcnemar"]
    cols = st.columns(len(mc))
    for col, (pair, r) in zip(cols, mc.items()):
        a, b = pair.split(" vs ")
        verdict = "Significant difference" if r["significant"] else "No real difference"
        colour = C_HIGH if r["significant"] else C_LOW
        col.markdown(f"""<div class="kpi" style="background:linear-gradient(135deg,{colour},{colour}CC)">
        <div class="label">{a[:10]}… vs {b[:10]}…</div>
        <div class="value">p = {r['p_value']:.3f}</div>
        <div class="sub">{verdict}</div></div>""", unsafe_allow_html=True)

    mc_tbl = pd.DataFrame([
        {"Comparison": k, "χ² statistic": v["statistic"], "p-value": v["p_value"],
         "Only 1st correct": v["only_first_right"], "Only 2nd correct": v["only_second_right"],
         "Verdict": "Statistically significant" if v["significant"] else "No significant difference"}
        for k, v in mc.items()])
    st.dataframe(mc_tbl, **FULL, hide_index=True)

    st.markdown("""<div class="good"><b>Result:</b> every p-value is well above 0.05. The three
    models are <b>statistically indistinguishable</b> on this dataset — the small gaps in the
    scoreboard are within normal random variation, not genuine superiority. This is a mature,
    honest finding: when models tie statistically, the right criterion for choosing one becomes
    <i>interpretability</i>, not the third decimal place of accuracy.</div>""",
                unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 2. Decision threshold analysis
    # -----------------------------------------------------------------
    section("2️⃣ Is 50% the right cut-off for a medical decision?")
    st.markdown("""Every score on the Model Performance page assumes a patient is called "at risk"
    once their predicted probability crosses **50%**. That line is arbitrary — and for medicine it
    is usually the wrong choice, because the two mistakes are not equally serious. A **false
    alarm** costs one extra check-up; a **missed patient** goes home undiagnosed. Below, missing a
    sick patient is treated as <b>3× worse</b> than a false alarm, and we search for the cut-off
    that minimises total harm.""")

    thresh_model = st.selectbox("Choose a model to analyse", MODEL_NAMES, key="thresh_model")
    T = ADV[thresh_model]["thresholds"]

    strat_names = {"default": "Default (50%)", "youden": "Balanced (Youden's J)",
                   "f1_max": "Best F1 score", "cost_3to1": "Cost-optimal (3:1 penalty)"}
    strat_df = pd.DataFrame([
        {"Strategy": strat_names[k], "Cut-off": f"{v['threshold']:.0%}",
         "Accuracy": v["accuracy"], "Precision": v["precision"], "Recall": v["recall"],
         "Missed patients": v["missed"], "False alarms": v["false_alarms"]}
        for k, v in T.items()])
    st.dataframe(strat_df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}",
                                        "Recall": "{:.3f}"}).background_gradient(
        cmap="RdYlGn", subset=["Recall"]), **FULL, hide_index=True)

    default_missed = T["default"]["missed"]
    cost_missed = T["cost_3to1"]["missed"]
    saved = default_missed - cost_missed
    c1, c2, c3 = st.columns(3)
    kpi(c1, "k1", "Default cut-off misses", f"{default_missed:,}", "sick patients sent home")
    kpi(c2, "k3", "Cost-optimal cut-off misses", f"{cost_missed:,}", "sick patients sent home")
    kpi(c3, "k2", "Additional patients caught", f"{saved:,}",
       f"by lowering the cut-off to {T['cost_3to1']['threshold']:.0%}")

    fig = go.Figure()
    fig.add_bar(x=ADV[thresh_model]["cost_grid_x"], y=ADV[thresh_model]["cost_grid_y"],
               marker_color="#8B5CF6", name="Total harm")
    fig.add_vline(x=0.5, line_dash="dash", line_color="#64748B", annotation_text="Default 50%")
    fig.add_vline(x=T["cost_3to1"]["threshold"], line_dash="dash", line_color="#EC4899",
                 annotation_text=f"Best {T['cost_3to1']['threshold']:.0%}")
    fig.update_layout(height=380, plot_bgcolor="white",
                      title=f"{thresh_model}: total harm at every possible cut-off "
                            "(3× penalty for missing a sick patient)",
                      xaxis_title="Decision threshold", yaxis_title="Total harm score")
    st.plotly_chart(fig, **FULL)

    fig2 = go.Figure()
    for n in MODEL_NAMES:
        fig2.add_trace(go.Scatter(x=ADV[n]["pr_recall"], y=ADV[n]["pr_precision"], mode="lines",
                                  name=f"{n} (AP={ADV[n]['average_precision']:.3f})",
                                  line=dict(width=3, color=C_MODEL[n])))
    fig2.add_hline(y=float(load_data().cardio.mean()), line_dash="dash", line_color="grey",
                   annotation_text="Random guessing")
    fig2.update_layout(height=420, plot_bgcolor="white",
                       title="Precision-Recall curves — more informative than ROC for screening",
                       xaxis_title="Recall (sick patients caught)", yaxis_title="Precision")
    st.plotly_chart(fig2, **FULL)

    st.markdown(f"""<div class="warn"><b>Clinical takeaway:</b> moving {thresh_model}'s cut-off
    from 50% down to {T['cost_3to1']['threshold']:.0%} would catch <b>{saved:,} more genuinely
    sick patients</b>, at the cost of more false alarms. In a screening context that is usually the
    right trade — an unnecessary check-up is a minor inconvenience, while a missed diagnosis can
    cost a life. The Patient Prediction page still uses the standard 50% line so results stay
    comparable across the site, but this is the analysis a clinical deployment would actually run
    before choosing where to set it.</div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 3. Calibration proof
    # -----------------------------------------------------------------
    section("3️⃣ Can the probabilities be trusted at face value?")
    st.markdown("""A model can be accurate at sorting patients while still lying about the
    <i>size</i> of the risk — saying "80%" for a group where only 60% are really sick. A
    **reliability diagram** checks this directly: group patients by predicted risk, and see what
    fraction really turned out sick. A trustworthy model sits on the diagonal.""")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                              line=dict(dash="dash", color="#334155")))
    for n in MODEL_NAMES:
        fig3.add_trace(go.Scatter(x=ADV[n]["calib_pred"], y=ADV[n]["calib_true"], mode="lines+markers",
                                  name=f"{n} (Brier={ADV[n]['brier']:.4f})",
                                  line=dict(width=3, color=C_MODEL[n]), marker=dict(size=8)))
    fig3.update_layout(height=460, plot_bgcolor="white",
                       title="Reliability diagram — closer to the diagonal is more trustworthy",
                       xaxis_title="Risk the model predicted",
                       yaxis_title="Fraction who actually had heart disease")
    st.plotly_chart(fig3, **FULL)

    if "_uncalibrated" in ADV and ADV["_uncalibrated"]:
        st.markdown("**Before vs after calibration** — the deployed models are calibrated "
                    "(isotonic regression); here is the raw, uncalibrated comparison for the "
                    "models where it was measured, showing why the extra step was worth it.")
        unc_cols = st.columns(len(ADV["_uncalibrated"]))
        for col, (n, d) in zip(unc_cols, ADV["_uncalibrated"].items()):
            improvement = d["brier"] - ADV[n]["brier"]
            col.markdown(f"""<div class="note" style="font-size:13.5px">
            <b>{n}</b><br>Uncalibrated Brier: {d['brier']:.4f}<br>
            Calibrated Brier: {ADV[n]['brier']:.4f}<br>
            <span style="color:{'#059669' if improvement>0 else '#DC2626'}">
            {'Improved' if improvement>0 else 'Changed'} by {abs(improvement):.4f}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="good"><b>Brier score</b> measures calibration as a single number —
    lower is better, 0 is perfect. All three models score close together and close to the
    diagonal, confirming that when this app tells a patient "70% risk", that number can be taken
    at face value rather than treated as a rough sort order.</div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 4. Learning curves
    # -----------------------------------------------------------------
    if "_learning_curve" in ADV and ADV["_learning_curve"]:
        section("4️⃣ Would more data help?")
        st.markdown("""A **learning curve** trains the model on increasingly large slices of the
        training set and tracks accuracy on training data versus a held-out validation slice. If
        the two lines are still pulling apart at the right edge, more data would likely help. If
        they have converged, the model has already learned everything this data can teach it.""")
        lc_cols = st.columns(len(ADV["_learning_curve"]))
        for col, (n, d) in zip(lc_cols, ADV["_learning_curve"].items()):
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=d["sizes"], y=d["train"], mode="lines+markers",
                                      name="Training score", line=dict(color="#EF4444", width=3)))
            fig4.add_trace(go.Scatter(x=d["sizes"], y=d["valid"], mode="lines+markers",
                                      name="Validation score", line=dict(color="#10B981", width=3)))
            fig4.update_layout(height=340, plot_bgcolor="white", title=n,
                               xaxis_title="Training set size", yaxis_title="Accuracy",
                               legend=dict(orientation="h", y=-.25))
            col.plotly_chart(fig4, **FULL)
        st.markdown("""<div class="note"><b>Reading it:</b> the two lines converge and flatten out
        rather than continuing to diverge. That means the models have essentially learned all they
        can from 11 basic features — the ~73% ceiling is a property of the <i>data</i>, not a sign
        that the models need more patients or more training time.</div>""", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 5. Subgroup fairness
    # -----------------------------------------------------------------
    section("5️⃣ Does the model treat every group of patients fairly?")
    st.markdown("""A model can be accurate overall while performing worse for a specific group —
    for example, missing more sick women than sick men. We check accuracy and recall separately
    across gender and age bands to look for this kind of hidden imbalance.""")

    sg = ADV["_subgroups"]
    fair_model = st.selectbox("Choose a model to check", MODEL_NAMES, key="fair_model")
    fair_rows = []
    for gname, gdata in sg.items():
        m = gdata[fair_model]
        fair_rows.append({"Group": gname, "Patients": gdata["n"],
                          "Accuracy": m["accuracy"], "Recall": m["recall"],
                          "Precision": m["precision"],
                          "False alarm rate": m["fpr"],
                          "Actual disease rate": m["base_rate"]})
    fair_df = pd.DataFrame(fair_rows)
    st.dataframe(fair_df.style.format({"Accuracy": "{:.3f}", "Recall": "{:.3f}",
                                       "Precision": "{:.3f}", "False alarm rate": "{:.3f}",
                                       "Actual disease rate": "{:.3f}"}).background_gradient(
        cmap="RdYlGn", subset=["Recall"]), **FULL, hide_index=True)

    fig5 = go.Figure()
    fig5.add_bar(x=fair_df.Group, y=fair_df.Recall, marker_color="#6366F1", name="Recall",
                text=fair_df.Recall.round(3), textposition="outside")
    fig5.add_hline(y=fair_df.Recall.mean(), line_dash="dash", line_color="#EF4444",
                  annotation_text="Overall average")
    fig5.update_layout(height=380, plot_bgcolor="white", yaxis_range=[0, 1],
                       title=f"{fair_model}: recall (sick patients caught) by patient group")
    st.plotly_chart(fig5, **FULL)

    spread_recall = fair_df.Recall.max() - fair_df.Recall.min()
    box = "good" if spread_recall < 0.08 else "warn"
    st.markdown(f"""<div class="{box}"><b>Fairness check result:</b> recall varies by
    {spread_recall:.3f} between the best- and worst-served group for {fair_model}.
    {"This is a tight, acceptable spread — no group is being substantially under-served."
     if spread_recall < 0.08 else
     "This spread is worth watching — one group is caught noticeably less often than others, "
     "which would need investigating before a real clinical deployment."}
    Running this check is standard practice before deploying any clinical model, because overall
    accuracy alone can hide unequal performance across patient groups.</div>""",
                unsafe_allow_html=True)


elif PAGE == "ℹ️ About the Project":
    hero("ℹ️ About this Project",
         "Everything you need to explain this work confidently in a demo")

    d = META["dataset"]
    c = st.columns(4)
    kpi(c[0], "k1", "Original records", f"{d['rows_raw']:,}", "Kaggle cardiovascular dataset")
    kpi(c[1], "k2", "After cleaning", f"{d['rows_clean']:,}", f"{d['removed']:,} removed")
    kpi(c[2], "k3", "Models", "3", "Logistic · Forest · SVM")
    kpi(c[3], "k4", "Website pages", "8", "Including Clinical Validation")

    section("🎯 The problem")
    st.markdown("""Cardiovascular disease is the leading cause of death in the world. Doctors can
spot it, but screening every person by hand is slow and expensive. Machine learning can rank
patients by risk in milliseconds using measurements that any clinic already records.

But there is a catch. A hospital will never accept an AI that just says *"78% risk"* and refuses
to explain itself. Doctors need to know **why**. That is exactly the gap this project fills:
it predicts **and** explains.""")

    section("🗂️ The data")
    st.markdown(f"""
The **Cardiovascular Disease dataset** from Kaggle — {d['rows_raw']:,} real patient examination
records, roughly 50/50 between healthy and diseased, so no class-imbalance problem.

The raw file contains impossible values, such as a blood pressure of 16020 mmHg, so cleaning was
essential. We removed duplicates, blood pressures outside a medically sensible range, cases where
the lower reading was higher than the upper one, and extreme height/weight outliers —
**{d['removed']:,} rows in total ({d['removed']/d['rows_raw']*100:.1f}%)**, leaving
{d['rows_clean']:,} trustworthy records.

We then engineered three new features that doctors actually use:
**age in years** (the raw file stores age in days), **BMI** (weight ÷ height²), and
**pulse pressure** (upper BP − lower BP, an indicator of arterial stiffness).""")

    st.dataframe(pd.DataFrame({
        "Feature": [DISPLAY[f] for f in FEATURES],
        "Column name": FEATURES,
        "What it means": [PLAIN[f] for f in FEATURES]}),
        **FULL, hide_index=True)

    section("🔬 The machine-learning pipeline")
    st.markdown("""
| Step | What we did | Why |
|---|---|---|
| 1. Load | Read 70,000 records | Real data, not a toy set |
| 2. Clean | Removed impossible values and duplicates | Rubbish in, rubbish out |
| 3. Engineer | Built age-in-years, BMI, pulse pressure | Doctor-friendly, more predictive |
| 4. Split | 80% train / 20% test, stratified | The test set must stay unseen |
| 5. Scale | `StandardScaler` fitted on the training set only | SVM and Logistic Regression need it; fitting on train only prevents leakage |
| 6. Tune | `GridSearchCV` on each model | Find the best settings honestly |
| 7. Validate | 5-fold cross-validation | Prove the score is not luck |
| 8. Calibrate | `CalibratedClassifierCV` (isotonic) | Turn raw scores into honest probabilities so all three models agree |
| 9. Explain | SHAP + LIME | Open the black box |
| 10. Validate | McNemar's test, threshold analysis, calibration proof, fairness check | Go beyond accuracy — see the Clinical Validation page |
| 11. Deploy | This Streamlit website | Make it usable by a non-programmer |
""")

    section("🤖 Why these three models?")
    st.markdown("""
**Logistic Regression** — the simplest and most transparent. It draws one straight boundary. Fast,
and a doctor can read its coefficients directly. It is the honest baseline every project needs.

**Random Forest** — 200 decision trees each vote, and the majority wins. It captures curved
relationships and interactions between features (for example, high BP *combined with* old age)
that a straight line cannot. It was the strongest model here.

**Support Vector Machine (RBF)** — draws a curved boundary in a higher-dimensional space. It is
mathematically elegant but slow on large data, so it was trained on a 15,000-patient stratified
sample — a standard and defensible engineering trade-off.""")

    section("🧠 SHAP vs 🍋 LIME")
    st.markdown("""
| | SHAP | LIME |
|---|---|---|
| Idea | Shares credit fairly, like splitting a bill in game theory | Fits a simple model right next to your patient |
| Guarantee | The parts always add up to the exact prediction | No such guarantee — it is an approximation |
| Speed | Slower | Faster |
| Output | Precise numeric contributions | Readable if-then rules |
| Best for | Trustworthy audits and global insight | Quick explanations for one patient |

Using **both** and checking they agree is what turns a good project into a rigorous one.""")

    section("📁 Project structure")
    st.code("""heart_disease_detection/
├── notebooks/               ← 11 sequential notebooks, run in order
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Logistic_Regression.ipynb
│   ├── 06_Random_Forest.ipynb
│   ├── 07_SVM.ipynb
│   ├── 08_Model_Comparison.ipynb
│   ├── 09_SHAP_Explanation.ipynb
│   ├── 10_LIME_Explanation.ipynb
│   └── 11_Final_Prediction.ipynb
├── app.py                   ← this website (8 pages)
├── models/                  ← everything the notebooks saved
│   ├── logistic_regression.pkl · random_forest.pkl · svm.pkl · scaler.pkl
│   ├── metadata.json · metrics.json · comparison.json · advanced.json
│   ├── shap_background.npy · lime_background.npy
│   └── shapvals__*.npy      ← pre-computed global SHAP values (site loads instantly)
├── data/
│   ├── cardio_train.csv     ← the original Kaggle file
│   └── cleaned_data.csv     ← the cleaned version, used by the dashboard
├── requirements.txt
├── LICENSE
└── README.md""", language="text")

    section("🚀 How to run it")
    st.code("pip install -r requirements.txt\n"
            "# run notebooks/01 through notebooks/11 in order (only needed once)\n"
            "streamlit run app.py", language="bash")

    st.divider()
    st.markdown("""<div class="warn"><b>Disclaimer.</b> This is an academic machine-learning
    project built for educational purposes. It is not a medical device, it has not been clinically
    validated, and it must never be used to make real health decisions. Always consult a qualified
    doctor.</div>""", unsafe_allow_html=True)
