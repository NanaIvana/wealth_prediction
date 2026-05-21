"""
Wealth Quintile Predictor — Streamlit app
Cameroon DHS 2018 · Ordinal Logistic Regression (mord.LogisticAT)

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import base64
import os
import pickle
from pathlib import Path

import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wealth Quintile Predictor — Cameroon DHS",
    page_icon="₣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).parent
BG_PATH = APP_DIR / "wealth-bg.jpg"
DATA_PATH = APP_DIR / "CMHR71FL.SAV"
MODEL_PATH = APP_DIR / "ordinal_logit.pkl"

# -----------------------------------------------------------------------------
# i18n
# -----------------------------------------------------------------------------
REGIONS = [
    (1, "Adamaoua", "Adamaoua"),
    (2, "Yaoundé", "Yaoundé"),
    (3, "Centre (excl. Yaoundé)", "Centre (hors Yaoundé)"),
    (4, "Est", "Est"),
    (5, "Far North", "Extrême-Nord"),
    (6, "Littoral (excl. Douala)", "Littoral (hors Douala)"),
    (7, "North", "Nord"),
    (8, "North-West", "Nord-Ouest"),
    (9, "West", "Ouest"),
    (10, "South", "Sud"),
    (11, "South-West", "Sud-Ouest"),
    (12, "Douala", "Douala"),
]
EDUCATION = [
    (0, "None", "Aucun"),
    (1, "Primary", "Primaire"),
    (2, "Secondary", "Secondaire"),
    (3, "Higher", "Supérieur"),
]

T = {
    "en": {
        "title": "Wealth Quintile Predictor",
        "subtitle": "Ordinal logistic prediction · Cameroon DHS 2018",
        "method": "Method",
        "about_text": (
            "This application predicts the wealth quintile of a Cameroonian household "
            "based on six characteristics of the head of household and the dwelling. "
            "The underlying model is an ordinal logistic regression (LogisticAT) fitted "
            "on the DHS 2018 weighted sample."
        ),
        "tab_predict": "Predictor",
        "tab_report": "Analysis report",
        "tab_variables": "Variables",
        "section_profile": "Household profile",
        "section_result": "Prediction",
        "region": "Region",
        "milieu": "Place of residence",
        "urban": "Urban", "rural": "Rural",
        "sex": "Head of household — Sex",
        "male": "Male", "female": "Female",
        "education": "Head of household — Education level",
        "size": "Household size", "size_help": "Number of members (1–20)",
        "age": "Head of household — Age", "age_help": "Years (18–95)",
        "predict": "Predict wealth quintile", "reset": "Reset",
        "result_label": "Most likely quintile",
        "confidence": "Confidence",
        "distribution": "Probability across quintiles",
        "fill_required": "Please fill all fields with valid values.",
        "quintiles": ["Q1 — Poorest", "Q2 — Poor", "Q3 — Middle", "Q4 — Rich", "Q5 — Super Rich"],
        "footer": "Model: mord.LogisticAT (ordinal). Trained on Cameroon DHS 2018.",
        "placeholder_result": "Fill out the household profile and click Predict to see the wealth quintile estimate.",
        "conf_high": "High confidence",
        "conf_moderate": "Moderate confidence",
        "conf_low": "Low confidence",
        "conf_high_txt": "The model is fairly certain — the top quintile clearly dominates the others.",
        "conf_mod_txt": "The profile sits near a boundary between two quintiles. The top guess is reasonable but the neighbouring quintile is also plausible.",
        "conf_low_txt": "The model is uncertain — probability is spread across several quintiles. Use the result only as a rough indication.",
        "conf_note": (
            "Wealth is shaped by many factors beyond what the model sees (income shocks, "
            "informal assets, social networks, etc.). The model therefore returns a probability "
            "for each quintile rather than a single certainty. Treat the result as a guided estimate, "
            "not an absolute classification."
        ),
        "report_title": "What we learned from the data",
        "report_lead": "Plain-language summary of the regression analysis and machine learning carried out on the Cameroon DHS 2018 dataset (CMHR71FL).",
        "report_sections": [
            ("1. The question",
             "Which characteristics of a household and its head best explain whether the household ends up in the poorest, middle or richest part of the population? The DHS wealth index is split into five equal groups (quintiles) — Q1 is the poorest 20% and Q5 the richest 20%."),
            ("2. The data",
             "14,173 Cameroonian households from the 2018 Demographic and Health Survey. After cleaning and weighting, we kept six predictors: region, urban/rural milieu, sex of the head, education of the head, household size, and age of the head."),
            ("3. What we tried",
             "We compared multinomial logistic regression, an ordinal logistic regression (which respects the natural Q1<Q2<…<Q5 order), a Random Forest and an XGBoost classifier. Random Forest had the highest raw accuracy, but it treated the quintiles as unrelated labels."),
            ("4. Why we kept the ordinal model",
             "Wealth quintiles are ordered — being off by one quintile (predicting Q3 instead of Q4) is a much smaller mistake than jumping from Q1 to Q5. The ordinal logistic model (mord.LogisticAT) is the only one that uses that order, which is why we use it here even though its top-1 accuracy is slightly below Random Forest."),
            ("5. What drives wealth",
             "Education of the head is by far the strongest positive driver: each step up (none → primary → secondary → higher) sharply increases the probability of landing in a higher quintile. Living in an urban area, especially Douala or Yaoundé, also lifts households toward Q4–Q5. Conversely, large household size and living in the Far North, North or rural areas push toward Q1–Q2. Sex and age of the head have only a modest effect once education and region are accounted for."),
            ("6. How well it predicts",
             "On a held-out test set, the ordinal model reaches roughly 45–50% exact-quintile accuracy and over 80% within-one-quintile accuracy. That is good for a social-survey model with only six inputs — but it confirms wealth is partly explained by factors the survey does not capture."),
            ("7. Take-away",
             "Education and geography matter most. Public policy aimed at expanding secondary and higher education, and at reducing the urban/rural gap, would have the largest measurable impact on the wealth distribution of Cameroonian households."),
        ],
        "var_title": "About the variables",
        "var_lead": "The model uses six predictors taken from the Cameroon DHS 2018 household questionnaire. Here is what each one means and why it was kept.",
        "var_meaning": "Meaning",
        "var_why": "Why we kept it",
        "variables": [
            ("Region",
             "The administrative region where the household lives (12 categories, Yaoundé and Douala isolated).",
             "Captures large geographic differences in infrastructure, markets and services."),
            ("Place of residence (milieu)",
             "Whether the household is in an urban or a rural area.",
             "Urban households have systematically higher access to electricity, piped water and salaried jobs."),
            ("Sex of head of household",
             "Whether the head of household is male or female.",
             "Female-headed households are over-represented in lower quintiles in many DHS surveys; we test whether that holds in Cameroon."),
            ("Education of head",
             "Highest education level completed: none, primary, secondary, higher.",
             "Strongest single predictor of wealth — directly linked to earning capacity."),
            ("Household size",
             "Total number of usual residents in the household (1 to 20+).",
             "Larger households spread the same income over more people, which lowers the per-capita wealth score."),
            ("Age of head",
             "Age in completed years of the household head (18–95).",
             "Proxy for life-cycle accumulation of assets."),
            ("Wealth quintile (target)",
             "The variable we are predicting. DHS builds a wealth index from durable goods, housing quality and utilities, then splits households into 5 equal groups: Q1 poorest → Q5 richest.",
             "Standard international indicator of household economic status."),
        ],
    },
    "fr": {
        "title": "Prédicteur de Quintile de Richesse",
        "subtitle": "Régression logistique ordinale · EDS Cameroun 2018",
        "method": "Méthode",
        "about_text": (
            "Cette application prédit le quintile de richesse d'un ménage camerounais à partir "
            "de six caractéristiques du chef de ménage et du logement. Le modèle sous-jacent est "
            "une régression logistique ordinale (LogisticAT) ajustée sur l'échantillon pondéré de l'EDS 2018."
        ),
        "tab_predict": "Prédicteur",
        "tab_report": "Rapport d'analyse",
        "tab_variables": "Variables",
        "section_profile": "Profil du ménage",
        "section_result": "Prédiction",
        "region": "Région",
        "milieu": "Milieu de résidence",
        "urban": "Urbain", "rural": "Rural",
        "sex": "Chef de ménage — Sexe",
        "male": "Masculin", "female": "Féminin",
        "education": "Chef de ménage — Niveau d'éducation",
        "size": "Taille du ménage", "size_help": "Nombre de membres (1–20)",
        "age": "Chef de ménage — Âge", "age_help": "Années (18–95)",
        "predict": "Prédire le quintile", "reset": "Réinitialiser",
        "result_label": "Quintile le plus probable",
        "confidence": "Confiance",
        "distribution": "Probabilités par quintile",
        "fill_required": "Veuillez remplir tous les champs avec des valeurs valides.",
        "quintiles": ["Q1 — Très pauvre", "Q2 — Pauvre", "Q3 — Moyen", "Q4 — Riche", "Q5 — Très riche"],
        "footer": "Modèle : mord.LogisticAT (ordinal). Entraîné sur l'EDS Cameroun 2018.",
        "placeholder_result": "Remplissez le profil du ménage puis cliquez sur Prédire pour voir l'estimation.",
        "conf_high": "Confiance élevée",
        "conf_moderate": "Confiance modérée",
        "conf_low": "Confiance faible",
        "conf_high_txt": "Le modèle est assez sûr — le quintile de tête domine nettement.",
        "conf_mod_txt": "Le profil est proche d'une frontière entre deux quintiles. L'estimation principale est raisonnable mais le quintile voisin reste plausible.",
        "conf_low_txt": "Le modèle est incertain — la probabilité est répartie sur plusieurs quintiles. À utiliser comme simple indication.",
        "conf_note": (
            "La richesse dépend de nombreux facteurs que le modèle ne voit pas (chocs de revenu, "
            "actifs informels, réseaux sociaux, etc.). Le modèle renvoie donc une probabilité par "
            "quintile plutôt qu'une certitude. Le résultat est une estimation indicative, pas une "
            "classification absolue."
        ),
        "report_title": "Ce que les données nous ont appris",
        "report_lead": "Résumé en langage simple du travail de régression et de machine learning mené sur les données EDS Cameroun 2018 (CMHR71FL).",
        "report_sections": [
            ("1. La question",
             "Quelles caractéristiques d'un ménage et de son chef expliquent le mieux son appartenance aux 20 % les plus pauvres, à la classe moyenne ou aux 20 % les plus riches ? L'indice de richesse EDS découpe la population en cinq groupes égaux (quintiles)."),
            ("2. Les données",
             "14 173 ménages camerounais issus de l'EDS 2018. Après nettoyage et pondération, six variables ont été retenues : région, milieu, sexe du chef, éducation, taille du ménage, âge du chef."),
            ("3. Les modèles testés",
             "Régression logistique multinomiale, régression logistique ordinale, Random Forest et XGBoost. Le Random Forest a la meilleure précision brute, mais traite les quintiles comme des étiquettes indépendantes."),
            ("4. Pourquoi garder le modèle ordinal",
             "Les quintiles sont ordonnés : se tromper d'un quintile (prédire Q3 au lieu de Q4) est bien moins grave que sauter de Q1 à Q5. Le modèle ordinal (mord.LogisticAT) est le seul à exploiter cet ordre."),
            ("5. Ce qui détermine la richesse",
             "L'éducation du chef est de loin le moteur positif le plus fort. Vivre en zone urbaine, surtout Douala ou Yaoundé, pousse aussi vers Q4–Q5. À l'inverse, une grande taille de ménage et les régions Extrême-Nord, Nord et zones rurales tirent vers Q1–Q2. Le sexe et l'âge n'ont qu'un effet modeste."),
            ("6. Qualité de prédiction",
             "Sur l'échantillon test : environ 45–50 % d'exactitude exacte et plus de 80 % à un quintile près. Satisfaisant pour six variables, mais la richesse dépend aussi de facteurs non mesurés."),
            ("7. À retenir",
             "L'éducation et la géographie pèsent le plus. Une politique d'expansion du secondaire/supérieur et de réduction de l'écart urbain/rural aurait l'impact mesurable le plus fort."),
        ],
        "var_title": "À propos des variables",
        "var_lead": "Le modèle utilise six variables tirées du questionnaire ménage EDS Cameroun 2018.",
        "var_meaning": "Signification",
        "var_why": "Pourquoi la garder",
        "variables": [
            ("Région",
             "Région administrative (12 catégories, Yaoundé et Douala isolées).",
             "Capture les fortes différences géographiques."),
            ("Milieu de résidence",
             "Ménage urbain ou rural.",
             "Les ménages urbains ont un meilleur accès à l'électricité, à l'eau courante et au salariat."),
            ("Sexe du chef",
             "Homme ou femme.",
             "Les ménages dirigés par des femmes sont sur-représentés dans les quintiles bas."),
            ("Éducation du chef",
             "Aucun, primaire, secondaire, supérieur.",
             "Variable la plus liée à la richesse."),
            ("Taille du ménage",
             "Nombre de résidents habituels (1 à 20+).",
             "Un même revenu réparti sur plus de personnes baisse le score par tête."),
            ("Âge du chef",
             "Âge en années révolues (18–95).",
             "Indicateur d'accumulation d'actifs sur le cycle de vie."),
            ("Quintile de richesse (cible)",
             "Variable à prédire. L'EDS construit un indice à partir des biens, du logement et des services, puis répartit en 5 groupes égaux.",
             "Indicateur international standard du statut économique."),
        ],
    },
}

# -----------------------------------------------------------------------------
# Background + global CSS (matches the React design)
# -----------------------------------------------------------------------------
def _bg_data_uri() -> str:
    if not BG_PATH.exists():
        return ""
    data = base64.b64encode(BG_PATH.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"

def inject_css() -> None:
    bg_uri = _bg_data_uri()
    st.markdown(f"""
    <style>
    /* ---- Background (fixed image + dark navy gradient overlay) ---- */
    .stApp {{
        background:
            linear-gradient(135deg,
                oklch(0.15 0.04 250 / 0.85),
                oklch(0.2 0.06 240 / 0.62) 50%,
                oklch(0.13 0.04 250 / 0.88)
            ),
            url("{bg_uri}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: oklch(0.97 0.01 240);
    }}
    .block-container {{ max-width: 1200px; padding-top: 1.2rem; padding-bottom: 3rem; }}
    section[data-testid="stSidebar"] {{ display: none; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* ---- Glass cards ---- */
    .glass {{
        background: oklch(0.23 0.05 250 / 0.72);
        border: 1px solid oklch(0.78 0.14 85 / 0.22);
        border-radius: 24px;
        padding: 2rem;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        margin-bottom: 1.25rem;
    }}
    .glass-soft {{
        background: oklch(0.22 0.05 250 / 0.55);
        border: 1px solid oklch(0.78 0.14 85 / 0.22);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
    }}

    /* ---- Typography ---- */
    h1,h2,h3,h4 {{ color: oklch(0.97 0.01 240); letter-spacing: -0.01em; }}
    .eyebrow {{
        font-size: 11px; text-transform: uppercase; letter-spacing: .25em;
        color: oklch(0.72 0.13 200); font-weight: 600;
    }}
    .muted {{ color: oklch(0.78 0.02 240); font-size: .9rem; }}
    .tiny  {{ color: oklch(0.78 0.02 240 / 0.85); font-size: .78rem; }}

    /* ---- Header brand ---- */
    .brand {{
        display:flex; align-items:center; gap: .85rem;
    }}
    .brand-mark {{
        width: 42px; height: 42px; border-radius: 14px;
        display:grid; place-items:center; font-weight: 800; font-size: 18px;
        background: linear-gradient(135deg, oklch(0.78 0.14 85), oklch(0.72 0.13 200));
        color: oklch(0.18 0.04 250);
    }}
    .brand-meta {{ display:flex; flex-direction:column; line-height:1.1; }}
    .brand-meta small {{
        font-size: 11px; letter-spacing: .25em; text-transform: uppercase;
        color: oklch(0.78 0.02 240);
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px; background: oklch(0.23 0.05 250 / 0.6);
        padding: 6px; border-radius: 16px;
        border: 1px solid oklch(0.78 0.14 85 / 0.22);
        backdrop-filter: blur(14px);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px; padding: 0 18px; border-radius: 12px;
        color: oklch(0.78 0.02 240); font-weight: 500;
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, oklch(0.78 0.14 85), oklch(0.72 0.13 200)) !important;
        color: oklch(0.18 0.04 250) !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ display:none; }}
    .stTabs [data-baseweb="tab-border"] {{ display:none; }}

    /* ---- Inputs ---- */
    div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {{
        background: oklch(0.3 0.05 245 / 0.55) !important;
        border: 1px solid oklch(0.78 0.14 85 / 0.22) !important;
        border-radius: 12px !important;
        color: oklch(0.97 0.01 240) !important;
    }}
    label, .stSelectbox label, .stNumberInput label {{
        color: oklch(0.78 0.02 240) !important;
        font-size: .72rem !important; letter-spacing: .12em; text-transform: uppercase;
        font-weight: 600 !important;
    }}

    /* ---- Buttons ---- */
    .stButton > button {{
        border-radius: 14px; padding: .65rem 1.2rem; font-weight: 600;
        border: 1px solid oklch(0.78 0.14 85 / 0.22);
        background: oklch(0.28 0.05 240 / 0.7);
        color: oklch(0.97 0.01 240);
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, oklch(0.78 0.14 85), oklch(0.72 0.13 200));
        color: oklch(0.18 0.04 250); border: none;
    }}

    /* ---- Result hero card ---- */
    .result-hero {{
        border-radius: 20px; padding: 1.5rem; text-align:center;
        background: linear-gradient(135deg, oklch(0.82 0.16 85 / 0.18), oklch(0.65 0.16 165 / 0.18));
        border: 1px solid oklch(0.82 0.16 85 / 0.35);
    }}
    .result-hero .q {{ font-size: 3rem; font-weight: 800; margin: .3rem 0 .15rem; }}
    .pill {{
        display:inline-flex; align-items:center; gap:.4rem;
        padding: .25rem .8rem; border-radius: 9999px;
        border: 1px solid oklch(0.78 0.14 85 / 0.22);
        background: oklch(0.23 0.05 250 / 0.6); font-size: .8rem;
    }}
    .pill b {{ color: oklch(0.78 0.14 85); }}

    /* ---- Confidence interpretation ---- */
    .conf-card {{
        border-radius: 18px; padding: 1rem 1.1rem;
        background: oklch(0.23 0.05 250 / 0.55);
        margin-top: .9rem;
    }}
    .conf-dot {{
        display:inline-block; width: 10px; height:10px; border-radius:50%;
        margin-right: .5rem;
    }}

    /* ---- Probability bars ---- */
    .bar-row {{ display:flex; align-items:center; gap: .75rem; margin: .35rem 0; }}
    .bar-label {{ width: 130px; font-size: .78rem; color: oklch(0.78 0.02 240); }}
    .bar-label b {{ color: oklch(0.97 0.01 240); }}
    .bar-track {{
        flex:1; height: 22px; border-radius: 8px;
        background: oklch(0.28 0.05 240 / 0.7); overflow:hidden; position:relative;
    }}
    .bar-fill {{
        position:absolute; top:0; bottom:0; left:0;
        background: oklch(0.5 0.06 160 / 0.7);
    }}
    .bar-fill.active {{
        background: linear-gradient(90deg, oklch(0.78 0.14 85), oklch(0.72 0.13 200));
    }}
    .bar-pct {{ width: 56px; text-align:right; font-size: .78rem; font-variant-numeric: tabular-nums; }}
    .bar-pct.active {{ color: oklch(0.78 0.14 85); font-weight: 700; }}

    /* ---- Footer ---- */
    .foot {{
        margin-top: 1.5rem; text-align:center; font-size: .72rem;
        color: oklch(0.78 0.02 240);
        background: oklch(0.23 0.05 250 / 0.45);
        border: 1px solid oklch(0.78 0.14 85 / 0.18);
        border-radius: 14px; padding: .8rem;
        backdrop-filter: blur(10px);
    }}

    /* Variable card numbered badge */
    .var-badge {{
        display:inline-grid; place-items:center;
        width: 28px; height: 28px; border-radius: 10px;
        background: linear-gradient(135deg, oklch(0.78 0.14 85), oklch(0.72 0.13 200));
        color: oklch(0.18 0.04 250); font-weight:800; font-size: .8rem;
        margin-right: .6rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model loading / training
# -----------------------------------------------------------------------------
FEATURES = ["Taille_Menage", "Age_Chef", "Education_Chef", "Mil", "Sexe_Chef", "Region"]

def _find_col(df_cols, candidates):
    """Return the first column name in df_cols that matches any candidate."""
    norm = {c.upper().replace("$", "_"): c for c in df_cols}

    for cand in candidates:
        key = cand.upper().replace("$", "_")

        if key in norm:
            return norm[key]

        for k, original in norm.items():
            if k.startswith(key + "_") or k == key:
                return original

    return None


@st.cache_resource(show_spinner=True)
def load_or_train_model():
    """Load cached pickle or train mord.LogisticAT on CMHR71FL.SAV."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    if not DATA_PATH.exists():
        st.warning(
            f"Place CMHR71FL.SAV next to streamlit_app.py to train the model. "
            f"Expected: {DATA_PATH}"
        )
        return None

    import pyreadstat
    from sklearn.preprocessing import StandardScaler
    import mord

    df, meta = pyreadstat.read_sav(str(DATA_PATH))

    # Auto-detect actual DHS column names (they can be HV106_01, HV106$01, etc.)
    mapping_specs = {
        "Quintile":      ["HV270"],
        "Region":        ["HV024"],
        "Mil":           ["HV025"],
        "Sexe_Chef":     ["HV219"],
        "Age_Chef":      ["HV220"],
        "Taille_Menage": ["HV009"],
        # education of head: try the head-specific col first, then plain HV106 / HV108
        "Education_Chef": ["HV106_01", "HV106$01", "HV106_1", "HV106", "HV108_01", "HV108$01", "HV108"],
    }

    rename = {}
    missing = []
    for target, candidates in mapping_specs.items():
        found = _find_col(df.columns, candidates)
        if found is None:
            missing.append(f"{target} (tried {candidates})")
        else:
            rename[found] = target

    if missing:
        st.error(
            "Some expected DHS variables were not found in your SAV file:\n\n- "
            + "\n- ".join(missing)
            + "\n\nColumns present in the file (first 60):\n"
            + ", ".join(list(df.columns)[:60])
        )
        return None

    df = df.rename(columns=rename)
    keep = list(mapping_specs.keys())
    df = df[keep].dropna()
    df["Quintile"] = df["Quintile"].astype(int)
    df["Education_Chef"] = df["Education_Chef"].astype(int).clip(0, 3)

    X = pd.get_dummies(
        df[FEATURES],
        columns=["Mil", "Sexe_Chef", "Region"],
        drop_first=True,
    ).astype(float)
    y = df["Quintile"].astype(int).values

    scaler = StandardScaler()
    cont = ["Taille_Menage", "Age_Chef", "Education_Chef"]
    X[cont] = scaler.fit_transform(X[cont])

    model = mord.LogisticAT(alpha=1.0)
    model.fit(X.values, y)

    bundle = {
        "model": model,
        "scaler": scaler,
        "columns": list(X.columns),
        "cont": cont,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle

def build_input_row(bundle, region, milieu, sex, education, size, age):
    row = pd.DataFrame([{
        "Taille_Menage": size,
        "Age_Chef": age,
        "Education_Chef": education,
        "Mil": milieu,
        "Sexe_Chef": sex,
        "Region": region,
    }])
    row = pd.get_dummies(row, columns=["Mil", "Sexe_Chef", "Region"], drop_first=True)
    # align to training columns
    for c in bundle["columns"]:
        if c not in row.columns:
            row[c] = 0
    row = row[bundle["columns"]].astype(float)
    row[bundle["cont"]] = bundle["scaler"].transform(row[bundle["cont"]])
    return row.values


def predict(bundle, region, milieu, sex, education, size, age):
    x = build_input_row(bundle, region, milieu, sex, education, size, age)
    model = bundle["model"]
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
    else:
        # fallback: hard prediction → one-hot
        yhat = int(model.predict(x)[0])
        proba = np.zeros(len(classes))
        proba[classes.index(yhat)] = 1.0
    probs5 = np.zeros(5)
    for i, c in enumerate(classes):
        if 1 <= int(c) <= 5:
            probs5[int(c) - 1] = proba[i]
    if probs5.sum() > 0:
        probs5 = probs5 / probs5.sum()
    q = int(np.argmax(probs5)) + 1
    return q, probs5

# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def confidence_tier(top: float, L: dict):
    if top >= 0.55:
        return L["conf_high"], L["conf_high_txt"], "oklch(0.7 0.17 150)"
    if top >= 0.35:
        return L["conf_moderate"], L["conf_mod_txt"], "oklch(0.78 0.14 85)"
    return L["conf_low"], L["conf_low_txt"], "oklch(0.65 0.2 30)"

def render_result(L, q, probs):
    top = float(probs[q - 1])
    label = L["quintiles"][q - 1]
    st.markdown(f"""
    <div class="result-hero">
      <div class="eyebrow">{L["result_label"]}</div>
      <div class="q">Q{q}</div>
      <div class="muted">{label}</div>
      <div style="margin-top:.6rem;"><span class="pill">{L["confidence"]}: <b>{top*100:.1f}%</b></span></div>
    </div>
    """, unsafe_allow_html=True)

    tier_label, tier_txt, tier_color = confidence_tier(top, L)
    st.markdown(f"""
    <div class="conf-card" style="border:1px solid {tier_color};">
      <div>
        <span class="conf-dot" style="background:{tier_color}; box-shadow: 0 0 12px {tier_color};"></span>
        <b style="color:{tier_color};">{tier_label}</b>
      </div>
      <p class="muted" style="margin-top:.5rem;">{tier_txt}</p>
      <p class="tiny" style="margin-top:.4rem;">{L["conf_note"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
    f"<div class='eyebrow' style='margin-top:1.2rem;'>{L['distribution']}</div>",
    unsafe_allow_html=True
    )

    labels = L["quintiles"]
    values = [float(p) for p in probs]

    fig = px.bar(
         x=labels,
         y=values,
         text=[f"{v*100:.1f}%" for v in values]
    )

    fig.update_layout(
    yaxis=dict(title="Probability", range=[0, 1]),
    xaxis=dict(title="Wealth Quintile"),
    showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
inject_css()

# language toggle
if "lang" not in st.session_state:
    st.session_state.lang = "en"

head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown("""
    <div class="brand">
      <div class="brand-mark">₣</div>
      <div class="brand-meta">
        <small>DHS · 2018</small>
        <strong style="font-size:1.05rem;">Wealth Quintile Predictor</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    lang = st.radio(
        " ", ["EN", "FR"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True, label_visibility="collapsed",
        key="lang_radio",
    )
    st.session_state.lang = "en" if lang == "EN" else "fr"

L = T[st.session_state.lang]

# Hero
st.markdown(f"""
<div class="glass">
  <div class="eyebrow">{L["method"]}</div>
  <h2 style="margin:.4rem 0 .2rem;">{L["title"]}</h2>
  <div class="muted">{L["subtitle"]}</div>
  <p class="muted" style="margin-top: .9rem; max-width: 820px;">{L["about_text"]}</p>
</div>
""", unsafe_allow_html=True)

bundle = load_or_train_model()

tab_predict, tab_report, tab_variables = st.tabs([L["tab_predict"], L["tab_report"], L["tab_variables"]])

# -------------------- PREDICTOR --------------------
with tab_predict:
    left, right = st.columns([1.1, 1])
    with left:
        with st.container():
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown(f"### {L['section_profile']}")

            region_labels = [r[1] if st.session_state.lang == "en" else r[2] for r in REGIONS]
            region_idx = st.selectbox(L["region"], options=list(range(len(REGIONS))),
                                      format_func=lambda i: region_labels[i], index=0)
            region = REGIONS[region_idx][0]

            c1, c2 = st.columns(2)
            with c1:
                milieu = st.selectbox(L["milieu"], options=[1, 2],
                                      format_func=lambda v: L["urban"] if v == 1 else L["rural"])
            with c2:
                sex = st.selectbox(L["sex"], options=[1, 2],
                                   format_func=lambda v: L["male"] if v == 1 else L["female"])

            edu_idx = st.selectbox(L["education"], options=list(range(len(EDUCATION))),
                                   format_func=lambda i: EDUCATION[i][1] if st.session_state.lang == "en" else EDUCATION[i][2])
            education = EDUCATION[edu_idx][0]

            c3, c4 = st.columns(2)
            with c3:
                size = st.number_input(L["size"], min_value=1, max_value=20, value=5, step=1, help=L["size_help"])
            with c4:
                age = st.number_input(L["age"], min_value=18, max_value=95, value=40, step=1, help=L["age_help"])

            b1, b2 = st.columns([2, 1])
            with b1:
                submit = st.button(L["predict"], type="primary", use_container_width=True,
                                   disabled=(bundle is None))
            with b2:
                if st.button(L["reset"], use_container_width=True):
                    for k in ("pred_q", "pred_probs"):
                        st.session_state.pop(k, None)
                    st.rerun()

            if submit and bundle is not None:
                q, probs = predict(bundle, region, milieu, sex, education, int(size), int(age))
                st.session_state.pred_q = q
                st.session_state.pred_probs = probs.tolist()

            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(f"### {L['section_result']}")
        if "pred_q" in st.session_state:
            render_result(L, st.session_state.pred_q, np.array(st.session_state.pred_probs))
        else:
            st.markdown(
                f"<div class='glass-soft' style='text-align:center; min-height: 240px; "
                f"display:flex; align-items:center; justify-content:center;'>"
                f"<span class='muted'>{L['placeholder_result']}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------- ANALYSIS REPORT --------------------
with tab_report:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f"## {L['report_title']}")
    st.markdown(f"<p class='muted'>{L['report_lead']}</p>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (h, p) in enumerate(L["report_sections"]):
        with cols[i % 2]:
            st.markdown(
                f"<div class='glass-soft'>"
                f"<div style='color: oklch(0.78 0.14 85); font-weight:700; font-size:.9rem;'>{h}</div>"
                f"<p class='muted' style='margin-top:.4rem;'>{p}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- VARIABLES --------------------
with tab_variables:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f"## {L['var_title']}")
    st.markdown(f"<p class='muted'>{L['var_lead']}</p>", unsafe_allow_html=True)
    for i, (name, meaning, why) in enumerate(L["variables"], start=1):
        st.markdown(f"""
        <div class="glass-soft">
          <div style="display:flex; align-items:center;">
            <span class="var-badge">{i}</span>
            <strong style="font-size:1.02rem;">{name}</strong>
          </div>
          <p style="margin:.55rem 0 .2rem;">
            <span class="eyebrow">{L["var_meaning"]}</span><br>
            <span>{meaning}</span>
          </p>
          <p style="margin:.4rem 0 0;">
            <span class="eyebrow">{L["var_why"]}</span><br>
            <span class="muted">{why}</span>
          </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f"<div class='foot'>{L['footer']}</div>", unsafe_allow_html=True)
