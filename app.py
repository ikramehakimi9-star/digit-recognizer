import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import center_of_mass, shift
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Reconnaissance de chiffres manuscrits | KNN",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Style personnalisé (inspiration Bootstrap)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 55%, #f8fafc 100%);
        color: #172033;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(120deg, #172554 0%, #3730a3 55%, #4f46e5 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 45px rgba(49,46,129,.18);
    }
    .hero-kicker {
        color: #c7d2fe;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }
    .hero h1 {
        color: white;
        font-size: clamp(1.8rem, 4vw, 2.65rem);
        line-height: 1.12;
        font-weight: 800;
        margin: 0 0 .75rem 0;
    }
    .hero p {
        color: #e0e7ff;
        font-size: 1.02rem;
        margin: 0;
        max-width: 760px;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1e1b4b;
        margin: .25rem 0 .35rem 0;
    }
    .section-subtitle {
        color: #64748b;
        font-size: .92rem;
        margin-bottom: 1rem;
    }
    .panel {
        background: rgba(255,255,255,.92);
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 8px 25px rgba(15,23,42,.045);
    }
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 100px;
    }
    .metric-label {
        color: #64748b;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    .metric-value {
        color: #312e81;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: .2rem;
    }
    .result-card {
        background: linear-gradient(145deg, #eef2ff, #ffffff);
        border: 1px solid #c7d2fe;
        border-radius: 20px;
        padding: 1.35rem;
        text-align: center;
    }
    .result-number {
        color: #3730a3;
        font-size: 5rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .result-caption {
        color: #64748b;
        font-size: .88rem;
        font-weight: 600;
    }
    .hint {
        background: #eff6ff;
        border-left: 4px solid #6366f1;
        padding: .8rem 1rem;
        border-radius: 10px;
        color: #334155;
        font-size: .9rem;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: .8rem;
        padding-top: 1.8rem;
    }
    div.stButton > button {
        border-radius: 12px;
        border: 0;
        font-weight: 700;
        min-height: 2.8rem;
        transition: .2s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #4338ca, #6366f1);
        color: white;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(100deg, #3730a3, #4f46e5);
        box-shadow: 0 8px 18px rgba(79,70,229,.2);
    }
    [data-testid="stCanvas"] {
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Entraînement du modèle
# ---------------------------------------------------------
@st.cache_resource
def train_knn():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.30,
        random_state=42,
        stratify=digits.target,
    )
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    return model, score, digits

model, model_accuracy, digits_data = train_knn()

# ---------------------------------------------------------
# Prétraitement du dessin : fond noir, chiffre clair, 8x8
# ---------------------------------------------------------
def prepare_drawing(image_data):
    """Transforme le dessin du canvas en vecteur 8x8 compatible avec load_digits."""
    if image_data is None:
        return None

    rgba = Image.fromarray(image_data.astype("uint8")).convert("RGBA")
    white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white_bg.alpha_composite(rgba)
    gray = white_bg.convert("L")

    # Le canvas utilise un chiffre sombre sur fond blanc.
    # Le dataset sklearn utilise un chiffre clair sur fond noir.
    inverted = Image.eval(gray, lambda px: 255 - px)
    arr = np.asarray(inverted, dtype=np.uint8)

    # Nettoyage léger et détection de la zone contenant le chiffre.
    mask = arr > 25
    if not np.any(mask):
        return None

    ys, xs = np.where(mask)
    cropped = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop_img = Image.fromarray(cropped)

    # Redimensionnement proportionnel dans une zone 7x7,
    # puis placement au centre d'une image 8x8.
    h, w = cropped.shape
    scale = 7.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (8, 8), 0)
    offset = ((8 - new_w) // 2, (8 - new_h) // 2)
    canvas.paste(resized, offset)

    digit = np.asarray(canvas, dtype=np.float32)

    # Recentre le centre de gravité du tracé.
    if digit.sum() > 0:
        cy, cx = center_of_mass(digit)
        digit = shift(
            digit,
            shift=(3.5 - cy, 3.5 - cx),
            order=1,
            mode="constant",
            cval=0.0,
        )

    # sklearn load_digits contient des intensités de 0 à 16.
    digit = np.clip(digit / 255.0 * 16.0, 0, 16)
    return digit.reshape(1, 64), digit


# ---------------------------------------------------------
# En-tête
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Machine Learning · Classification</div>
        <h1>Reconnaissance des chiffres manuscrits</h1>
        <p>Dessinez un chiffre de 0 à 9 et laissez l’algorithme K plus proches voisins
        (KNN) analyser votre écriture et prédire le résultat.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Indicateurs
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Algorithme</div>'
        f'<div class="metric-value">KNN</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Nombre de voisins (K)</div>'
        '<div class="metric-value">5</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Précision sur le test</div>'
        f'<div class="metric-value">{model_accuracy * 100:.1f}%</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------
# Interface de dessin et prédiction
# ---------------------------------------------------------
left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.markdown('<div class="section-title">✍️ 1. Dessinez votre chiffre</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Utilisez votre souris ou votre doigt pour écrire un seul chiffre.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        stroke = st.slider("Épaisseur du trait", min_value=8, max_value=30, value=18, step=2)
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = 0

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke,
            stroke_color="#111827",
            background_color="#FFFFFF",
            background_image=None,
            update_streamlit=True,
            height=300,
            width=300,
            drawing_mode="freedraw",
            point_display_radius=0,
            key=f"digit_canvas_{st.session_state.canvas_key}",
        )

        st.markdown(
            '<div class="hint">Conseil : écrivez au milieu de la zone, assez grand, '
            'et évitez de dessiner plusieurs chiffres en même temps.</div>',
            unsafe_allow_html=True,
        )

        btn_col1, btn_col2 = st.columns([1.35, 1])
        with btn_col1:
            predict_clicked = st.button("🔍  Reconnaître le chiffre", type="primary", use_container_width=True)
        with btn_col2:
            clear_clicked = st.button("↻  Effacer", use_container_width=True)

        if clear_clicked:
            st.session_state.canvas_key += 1
            st.session_state.pop("prediction", None)
            st.rerun()

        if predict_clicked:
            processed = prepare_drawing(canvas_result.image_data if canvas_result else None)
            if processed is None:
                st.warning("Veuillez d’abord dessiner un chiffre dans la zone blanche.")
                st.session_state.pop("prediction", None)
            else:
                sample, preview = processed
                predicted_digit = int(model.predict(sample)[0])
                probabilities = model.predict_proba(sample)[0]
                confidence = float(np.max(probabilities))
                st.session_state.prediction = {
                    "digit": predicted_digit,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "preview": preview,
                }

with right:
    st.markdown('<div class="section-title">🎯 2. Résultat de la prédiction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Le chiffre reconnu et le niveau de confiance du modèle.</div>',
        unsafe_allow_html=True,
    )

    prediction = st.session_state.get("prediction")
    if prediction:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-caption">CHIFFRE DÉTECTÉ</div>
                <div class="result-number">{prediction['digit']}</div>
                <div class="result-caption">Confiance du modèle : {prediction['confidence'] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**Prétraitement du dessin (format 8 × 8)**")
        st.image(
            prediction["preview"],
            width=190,
            clamp=True,
            channels="GRAY",
        )
        st.markdown("**Probabilité estimée pour chaque chiffre**")
        probability_df = pd.DataFrame({
            "Chiffre": [str(i) for i in range(10)],
            "Probabilité": prediction["probabilities"],
        }).set_index("Chiffre")
        st.bar_chart(probability_df, y="Probabilité", height=220)
    else:
        st.markdown(
            """
            <div class="panel" style="min-height:310px; display:flex; flex-direction:column;
                 align-items:center; justify-content:center; text-align:center;">
                <div style="font-size:3.5rem;">🔢</div>
                <h3 style="color:#312e81; margin:.5rem 0;">En attente de votre dessin</h3>
                <p style="color:#64748b; max-width:300px;">
                    Votre prédiction apparaîtra ici après avoir dessiné un chiffre
                    et cliqué sur « Reconnaître le chiffre ».
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Explication courte
# ---------------------------------------------------------
st.write("")
with st.expander("ℹ️ Comment fonctionne cette application ?"):
    st.markdown(
        """
        1. **Dataset :** le modèle utilise `load_digits()` de Scikit-learn, qui contient
           des images de chiffres manuscrits de 0 à 9 au format 8 × 8.
        2. **Prétraitement :** le dessin est converti en niveaux de gris, recadré,
           redimensionné en 8 × 8 et adapté au format du dataset.
        3. **Classification :** KNN compare le dessin aux exemples appris et recherche
           les 5 voisins les plus proches.
        4. **Prédiction :** le chiffre ayant le meilleur vote est affiché, avec les
           probabilités estimées par le modèle.

        **À noter :** la confiance affichée correspond aux probabilités de vote de KNN ;
        elle ne garantit pas que la prédiction est correcte.
        """
    )

st.markdown(
    '<div class="footer">Projet de Machine Learning · Scikit-learn + Streamlit · KNN</div>',
    unsafe_allow_html=True,
)
