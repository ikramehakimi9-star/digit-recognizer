
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


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="KNN - Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)


# =====================================================
# 2. DESIGN CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #f5f7ff,
        #eef2ff,
        #f8fafc
    );
    color: #172033;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(
        120deg,
        #172554,
        #3730a3,
        #4f46e5
    );
    color: white;
    padding: 2.5rem;
    border-radius: 24px;
    margin-bottom: 25px;
    box-shadow: 0 18px 45px rgba(49,46,129,.18);
}

.hero h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 800;
}

.hero p {
    color: #e0e7ff;
    font-size: 1rem;
}

.kicker {
    color: #c7d2fe;
    font-size: .8rem;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #1e1b4b;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #64748b;
    font-size: .9rem;
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px;
    min-height: 100px;
}

.metric-label {
    color: #64748b;
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
}

.metric-value {
    color: #3730a3;
    font-size: 1.7rem;
    font-weight: 800;
}

.result-card {
    background: linear-gradient(145deg, #eef2ff, white);
    border: 1px solid #c7d2fe;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}

.result-number {
    color: #3730a3;
    font-size: 6rem;
    font-weight: 800;
    line-height: 1.2;
}

.result-caption {
    color: #64748b;
    font-size: .9rem;
    font-weight: 600;
}

.hint {
    background: #eff6ff;
    border-left: 4px solid #6366f1;
    padding: 12px;
    border-radius: 10px;
    color: #334155;
    font-size: .9rem;
    margin-top: 12px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: .8rem;
    padding-top: 30px;
}

div.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 45px;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #4338ca, #6366f1);
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. CHARGEMENT ET ENTRAINEMENT DU MODELE KNN
# =====================================================

@st.cache_resource
def train_model():

    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


model, accuracy = train_model()


# =====================================================
# 4. PRETRAITEMENT DU DESSIN
# =====================================================

def prepare_drawing(image_data):

    if image_data is None:
        return None

    # Conversion en image RGBA
    image = Image.fromarray(
        image_data.astype(np.uint8)
    ).convert("RGBA")

    # Fond blanc
    background = Image.new(
        "RGBA",
        image.size,
        (255, 255, 255, 255)
    )

    background.alpha_composite(image)

    # Conversion niveaux de gris
    gray = background.convert("L")

    # Inversion : chiffre blanc, fond noir
    inverted = Image.eval(
        gray,
        lambda pixel: 255 - pixel
    )

    array = np.asarray(inverted, dtype=np.uint8)

    # Détection du chiffre
    mask = array > 25

    if not np.any(mask):
        return None

    # Recadrage
    ys, xs = np.where(mask)

    cropped = array[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    # Redimensionnement proportionnel
    h, w = cropped.shape

    scale = 7.0 / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = Image.fromarray(cropped).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    # Image finale 8x8
    final_image = Image.new(
        "L",
        (8, 8),
        0
    )

    offset = (
        (8 - new_w) // 2,
        (8 - new_h) // 2
    )

    final_image.paste(resized, offset)

    digit = np.asarray(
        final_image,
        dtype=np.float32
    )

    # Centrage du chiffre
    if digit.sum() > 0:

        cy, cx = center_of_mass(digit)

        digit = shift(
            digit,
            shift=(3.5 - cy, 3.5 - cx),
            order=1,
            mode="constant",
            cval=0
        )

    # Normalisation de 0 à 16
    digit = np.clip(
        digit / 255.0 * 16.0,
        0,
        16
    )

    return digit.reshape(1, 64), digit


# =====================================================
# 5. INITIALISATION SESSION
# =====================================================

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# =====================================================
# 6. HEADER
# =====================================================

st.markdown("""
<div class="hero">

    <div class="kicker">
        MACHINE LEARNING · CLASSIFICATION
    </div>

    <h1>
        Handwritten Digit Recognition
    </h1>

    <p>
        Dessinez un chiffre de 0 à 9 et laissez
        l'algorithme KNN reconnaître votre écriture
        grâce au Machine Learning.
    </p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 7. STATISTIQUES
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            Algorithme
        </div>
        <div class="metric-value">
            KNN
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">
            Nombre de voisins
        </div>
        <div class="metric-value">
            K = 5
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            Accuracy
        </div>
        <div class="metric-value">
            {accuracy * 100:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# =====================================================
# 8. INTERFACE PRINCIPALE
# =====================================================

left, right = st.columns(
    [1.08, 0.92],
    gap="large"
)


# =====================================================
# 9. ZONE DE DESSIN
# =====================================================

with left:

    st.markdown(
        '<div class="section-title">✍️ Draw your digit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Écrivez un chiffre dans la zone ci-dessous.'
        '</div>',
        unsafe_allow_html=True
    )

    stroke = st.slider(
        "Épaisseur du stylo",
        min_value=8,
        max_value=30,
        value=18,
        step=2
    )

    canvas_result = st_canvas(

        fill_color="rgba(255,255,255,0)",

        stroke_width=stroke,

        stroke_color="#111827",

        background_color="#FFFFFF",

        height=300,

        width=300,

        drawing_mode="freedraw",

        update_streamlit=True,

        key=f"canvas_{st.session_state.canvas_key}"

    )

    st.markdown("""
    <div class="hint">
        💡 Conseil : dessinez un seul chiffre,
        assez grand et au centre de la zone.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    button1, button2 = st.columns(2)

    with button1:

        predict_clicked = st.button(
            "🔍 Reconnaître le chiffre",
            type="primary",
            use_container_width=True
        )

    with button2:

        clear_clicked = st.button(
            "↻ Effacer",
            use_container_width=True
        )


# =====================================================
# 10. BOUTON EFFACER
# =====================================================

if clear_clicked:

    st.session_state.canvas_key += 1

    st.session_state.prediction = None

    st.rerun()


# =====================================================
# 11. PREDICTION KNN - CORRECTION CANVAS
# =====================================================

if predict_clicked:

    processed = None

    try:

        if canvas_result is not None:

            # Lecture sécurisée des données du canvas
            image_data = canvas_result.image_data

            if image_data is not None:

                processed = prepare_drawing(image_data)

    except (RuntimeError, ValueError, OSError, AttributeError):

        processed = None

    if processed is None:

        st.session_state.prediction = None

        st.warning(
            "Veuillez dessiner un chiffre avant "
            "de lancer la reconnaissance."
        )

    else:

        sample, preview = processed

        # Prédiction
        predicted_digit = int(
            model.predict(sample)[0]
        )

        # Probabilités de vote KNN
        probabilities = model.predict_proba(sample)[0]

        confidence = float(
            np.max(probabilities)
        )

        st.session_state.prediction = {

            "digit": predicted_digit,

            "confidence": confidence,

            "probabilities": probabilities,

            "preview": preview

        }


# =====================================================
# 12. AFFICHAGE DU RESULTAT
# =====================================================

with right:

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Résultat de la reconnaissance par KNN.'
        '</div>',
        unsafe_allow_html=True
    )

    prediction = st.session_state.prediction

    if prediction is not None:

        st.markdown(f"""
        <div class="result-card">

            <div class="result-caption">
                CHIFFRE DETECTE
            </div>

            <div class="result-number">
                {prediction["digit"]}
            </div>

            <div class="result-caption">
                Confiance : {prediction["confidence"] * 100:.1f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown("**Image après prétraitement (8 × 8)**")

        preview_image = (
            prediction["preview"] / 16.0 * 255.0
        ).clip(0, 255).astype(np.uint8)

        st.image(
            preview_image,
            width=160
        )

        st.markdown("**Probabilités par chiffre**")

        probability_df = pd.DataFrame({

            "Chiffre": [str(i) for i in range(10)],

            "Probabilité": prediction["probabilities"]

        }).set_index("Chiffre")

        st.bar_chart(
            probability_df,
            y="Probabilité",
            height=250
        )

    else:

        st.markdown("""
        <div class="result-card">

            <div style="font-size:60px;">
                🔢
            </div>

            <h3 style="color:#3730a3;">
                En attente de votre dessin
            </h3>

            <p style="color:#64748b;">
                Votre résultat apparaîtra ici
                après avoir dessiné un chiffre
                et cliqué sur le bouton de reconnaissance.
            </p>

        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 13. EXPLICATION DU FONCTIONNEMENT
# =====================================================

st.write("")

with st.expander("ℹ️ Comment fonctionne l'application ?"):

    st.markdown("""
    ### 1. Dataset

    Le modèle utilise le dataset `load_digits()`
    de Scikit-learn contenant des chiffres manuscrits
    de 0 à 9.

    ### 2. Prétraitement

    Le dessin de l'utilisateur est :

    - Converti en niveaux de gris.
    - Inversé pour obtenir un fond noir.
    - Recadré automatiquement.
    - Redimensionné en 8 × 8 pixels.
    - Normalisé entre 0 et 16.

    ### 3. Algorithme KNN

    L'algorithme recherche les 5 voisins
    les plus proches du chiffre dessiné.

    ### 4. Classification

    Le modèle effectue un vote parmi les voisins
    pour déterminer le chiffre reconnu.

    ### 5. Résultat

    L'application affiche le chiffre prédit
    et les probabilités de vote pour chaque classe.

    **Remarque :** les probabilités de vote ne
    constituent pas une garantie de précision.
    """)


# =====================================================
# 14. FOOTER
# =====================================================

st.markdown("""
<div class="footer">

    Machine Learning Project · KNN Algorithm

    <br>

    Built with Python · Scikit-learn · Streamlit

</div>
""", unsafe_allow_html=True)
