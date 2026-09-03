# ❤️ Heart Disease Detection using Explainable AI

**Logistic Regression · Random Forest · Support Vector Machine — explained with SHAP and LIME**

A complete, end-to-end machine learning project that predicts cardiovascular disease from 11
simple health measurements, and — more importantly — **explains every single prediction** using
two independent explainable-AI techniques.

---

## 🚀 Quick start (3 commands)

```bash
pip install -r requirements.txt
# run notebooks/01 through notebooks/11 in order — only needed once
streamlit run app.py
```

The website opens automatically at `http://localhost:8501`.

> The `models/` folder already contains trained models, so you can skip the notebooks and run the
> website straight away. Run them in order when you want to see (or demonstrate) the full
> training process — each one builds on the file the previous one saved.

---

## 📁 Project structure

```
heart_disease_detection/
│
├── app.py                    ← THE WEBSITE: 8 interactive Streamlit pages
├── requirements.txt          ← everything you need to install
├── README.md                 ← this file
├── .streamlit/config.toml    ← forces a light theme so headings are always readable
│
├── notebooks/                 ← the full pipeline, split into 11 sequential notebooks
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Logistic_Regression.ipynb
│   ├── 06_Random_Forest.ipynb
│   ├── 07_SVM.ipynb
│   ├── 08_Model_Comparison.ipynb      ← McNemar test, agreement, calibration proof
│   ├── 09_SHAP_Explanation.ipynb
│   ├── 10_LIME_Explanation.ipynb
│   └── 11_Final_Prediction.ipynb      ← assembles everything and verifies the saved files
│
├── models/                   ← everything the notebooks save
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── scaler.pkl                     the fitted StandardScaler
│   ├── metadata.json                  feature names + plain-English descriptions
│   ├── metrics.json                   every score, confusion matrix, ROC point, CV result
│   ├── comparison.json                agreement stats, Brier scores, optimal thresholds
│   ├── advanced.json                  McNemar tests, PR curves, calibration, fairness data
│   ├── shap_background.npy            20 representative patients for live SHAP
│   ├── lime_background.npy            5,000 training patients for live LIME
│   ├── shap_X_sample.npy              the 80 patients used for global SHAP
│   ├── shapvals__*.npy                pre-computed global SHAP values (site loads instantly)
│   └── shapbase__*.npy                each model's baseline prediction
│
└── data/
    ├── cardio_train.csv      ← the original Kaggle file (70,000 rows, semicolon separated)
    └── cleaned_data.csv      ← the cleaned version, used by the dashboard
```

---

## 📊 Results

All numbers below come from the **20% test set that the models never saw during training**.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | 5-fold CV |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7311 | 0.7375 | 0.7086 | 0.7227 | 0.7896 | 0.7274 ± 0.0028 |
| **Random Forest** | **0.7338** | 0.7515 | 0.6904 | 0.7197 | **0.7977** | 0.7345 ± 0.0032 |
| SVM (RBF) | 0.7318 | 0.7626 | 0.6646 | 0.7102 | 0.7920 | 0.7323 ± 0.0108 |

### Do the three models agree?

Yes — comfortably inside the 10-percentage-point target:

| Model pair | Average difference |
|---|---|
| Logistic Regression ↔ Random Forest | **5.22 pp** |
| Logistic Regression ↔ SVM | **5.53 pp** |
| Random Forest ↔ SVM | **4.64 pp** |
| Average spread across all three | **7.70 pp** |

This is not a coincidence. Every model is wrapped in `CalibratedClassifierCV` with **isotonic
regression**, which converts each model's raw score into an honest probability. Once all three
speak the same language, they naturally line up — and calibration slightly improved accuracy too.

---

## 🧠 The machine learning pipeline

| Step | What happens | Why it matters |
|---|---|---|
| 1. Load | 70,000 records from the Kaggle cardiovascular dataset | Real data, not a toy set |
| 2. Explore | Check for missing values, class balance, impossible values | Found blood pressures of 16,020 mmHg |
| 3. Clean | Remove duplicates and medically impossible values | 1,948 rows removed (2.78%), 68,052 kept |
| 4. Engineer | Build `age_years`, `bmi`, `pulse_pressure` | Doctor-friendly features that predict better |
| 5. Split | 80% train / 20% test, stratified | The test set must stay unseen |
| 6. Scale | `StandardScaler` **fitted on training data only** | Prevents data leakage |
| 7. Train | 3 models, each with `GridSearchCV` + 5-fold CV | No hand-picked settings |
| 8. Calibrate | `CalibratedClassifierCV` (isotonic) | Honest probabilities that agree |
| 9. Evaluate | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | The full picture, not just accuracy |
| 10. Explain | SHAP (local + global) and LIME, cross-checked | Open the black box |
| 11. Save | Everything into `models/` | So the website starts instantly |

### Best hyperparameters found by GridSearchCV

| Model | Settings |
|---|---|
| Logistic Regression | `C = 0.01` |
| Random Forest | `max_depth = 10`, `min_samples_leaf = 10`, `n_estimators = 200` |
| SVM | `C = 1.0`, `gamma = 0.05`, `kernel = rbf` |

---

## 🩺 The 11 features

| Feature | Meaning |
|---|---|
| `age_years` | Age in years (converted from days in the raw file) |
| `gender` | 0 = female, 1 = male |
| `bmi` | **Engineered**: weight ÷ height² |
| `ap_hi` | Systolic blood pressure (upper number) |
| `ap_lo` | Diastolic blood pressure (lower number) |
| `pulse_pressure` | **Engineered**: `ap_hi − ap_lo`, an indicator of arterial stiffness |
| `cholesterol` | 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc` | Blood sugar: 1 = normal, 2 = above, 3 = well above |
| `smoke` | 0 = no, 1 = yes (self-reported) |
| `alco` | 0 = no, 1 = yes (self-reported) |
| `active` | 0 = no, 1 = yes (self-reported) |

**Target:** `cardio` — 0 = healthy, 1 = has cardiovascular disease.

---

## 🩻 Beyond accuracy — clinical validation

Most student projects stop at a scoreboard. This one goes further, because a screening model
that will never actually be deployed still has to be validated the way a deployed one would be:

* **McNemar's test** — is the best-scoring model actually better, or could the difference be
  chance? Every pairwise comparison came back **not statistically significant** (p > 0.05).
* **Decision threshold analysis** — the default 50% cut-off is arbitrary. We modelled missing a
  sick patient as 3× worse than a false alarm and found the cut-off that minimises total harm,
  catching hundreds more true cases at the cost of more false alarms.
* **Calibration proof** — reliability diagrams and Brier scores confirm that when a model says
  "70% risk", roughly 70% of those patients really are sick, not just that the ranking is right.
* **Learning curves** — training and validation accuracy converge and flatten, showing the ~73%
  ceiling comes from the data, not from under-training.
* **Subgroup fairness** — accuracy and recall checked separately across gender and age bands, to
  catch a model that is accurate overall but under-serves a specific group.
* **Out-of-distribution detection** — the training data only ever contained BMI between 15 and 50
  (values outside that were removed as data-entry errors in notebook 02). Both the Patient
  Prediction page and the Batch CSV page detect when a user builds a patient outside that range,
  warn that the prediction is an extrapolation rather than a normal inference, and automatically
  explain why the three models are likely to disagree more sharply for that patient.

All of this lives on the website's **🩻 Clinical Validation** page and is generated entirely by
`notebooks/08_Model_Comparison.ipynb` — nothing about this page depends on a separate script.
Delete everything in `models/`, rerun the 11 notebooks in order, and both the models and the full
clinical validation suite (`comparison.json`, `advanced.json`) rebuild themselves from scratch.

## 🌐 The website — 8 pages

| Page | What it does |
|---|---|
| 🏠 **Dashboard** | KPI cards, model comparison, and four tabs of interactive Plotly charts exploring the data |
| 🩺 **Patient Prediction** | Move the sliders and get **all three models' probabilities at once** — three gauges, a comparison bar chart, a traffic-light health card and a plain-English summary |
| 🧠 **SHAP Explanation** | Choose any model. See a personal contribution chart, a step-by-step waterfall from average patient to this patient, global feature importance, a per-feature dependence plot, and a heat map comparing all three models |
| 🍋 **LIME Explanation** | Choose any model. See readable if-then rules, a weighted bar chart, and a rules table with strength indicators |
| 📁 **Batch CSV Prediction** | Upload a spreadsheet of many patients. All three models score every row. Includes a downloadable template, automatic format detection, risk breakdown charts, filtering, and a CSV download |
| 📊 **Model Performance** | Full metrics table, three confusion matrices, overlaid ROC curves, cross-validation fold chart, tuned hyperparameters, and the model-agreement analysis |
| 🩻 **Clinical Validation** | McNemar significance testing between models, optimal decision-threshold analysis with a cost model, precision-recall curves, calibration reliability diagrams, learning curves, and a subgroup fairness check across gender and age |
| ℹ️ **About the Project** | The problem, the data, the pipeline, SHAP vs LIME, and the project structure |

Every chart is interactive (hover, zoom, pan) and every technical result is followed by a
plain-English explanation of what it means.

---

## 🔍 SHAP vs LIME — why we use both

|  | 🧠 SHAP | 🍋 LIME |
|---|---|---|
| Core idea | Splits credit fairly, like dividing a prize in game theory | Fits a simple model right next to your patient |
| Guarantee | Contributions add up **exactly** to the prediction | An approximation — no such guarantee |
| Speed | Slower | Faster |
| Output | Precise numeric contributions | Readable if-then rules |
| Best for | Trustworthy audits and global insight | Quick explanations for one patient |

They rest on completely different mathematics. **When two independent methods point at the same
features, the explanation is real** — not an artefact of one technique. Both agree here that
**systolic blood pressure, age and cholesterol** drive the predictions, which is exactly what a
cardiologist would tell you.

---

## 🔧 Troubleshooting

**The website says a model file is missing.**
Run `notebooks/01` through `notebooks/11` in order first, or check that the `models/` folder sits
next to `app.py`.

**The SHAP page takes a few seconds.**
That is expected for the SVM — Kernel SHAP evaluates the model hundreds of times per patient. The
Logistic Regression and Random Forest pages are almost instant, and the *global* SHAP charts load
immediately because they were pre-computed in `notebooks/09_SHAP_Explanation.ipynb`.

**LIME gives a slightly different answer each time.**
LIME generates random nearby patients, so a little variation is normal. Increase the sample slider
on the LIME page for a steadier result.

**The dataset is missing.**
Download `cardio_train.csv` from
[Kaggle — Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
and place it in the `data/` folder. It must be semicolon-separated.

---

## 📚 Dataset credit

Cardiovascular Disease dataset by Svetlana Ulianova, hosted on Kaggle — 70,000 patient
examination records with 11 features and a binary target.

---

## ⚠️ Disclaimer

This is an **academic machine learning project built for educational purposes**. It is not a
medical device, it has not been clinically validated, and it must never be used to make real
health decisions. Always consult a qualified doctor.
