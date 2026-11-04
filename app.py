
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


# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Digit Recognition - KNN",
    page_icon="✍️",
    layout="wide"
)


# ==================================================
# 2. STYLE STREAMLIT NATIF (SANS HTML)
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #f4f6fc;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #25245e;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: bold;
    min-height: 45px;
}

div.stButton > button[kind="primary"] {
    background-color: #4f46e5;
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. ENTRAINEMENT DU MODELE KNN
# ==================================================

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

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


model, accuracy = train_model()


# ==================================================
# 4. PRETRAITEMENT DU DESSIN
# ==================================================

def prepare_drawing(image_data):

    if image_data is None:
        return None

    # Conversion RGBA
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

    # Niveaux de gris
    gray = background.convert("L")

    # Inversion des couleurs
    inverted = Image.eval(
        gray,
        lambda pixel: 255 - pixel
    )

    array = np.asarray(
        inverted,
        dtype=np.uint8
    )

    # Détection des pixels du chiffre
    mask = array > 25

    if not np.any(mask):
        return None

    # Recadrage automatique
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

    # Création image 8x8
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

    # Normalisation entre 0 et 16
    digit = np.clip(
        digit / 255.0 * 16.0,
        0,
        16
    )

    return digit.reshape(1, 64), digit


# ==================================================
# 5. INITIALISATION SESSION
# ==================================================

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ==================================================
# 6. HEADER
# ==================================================

st.title("✍️ Handwritten Digit Recognition")

st.markdown(
    "### Reconnaissance des chiffres manuscrits avec KNN"
)

st.write(
    "Dessinez un chiffre de 0 à 9 et laissez "
    "l'intelligence artificielle le reconnaître."
)

st.divider()


# ==================================================
# 7. STATISTIQUES
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Algorithme",
        "KNN"
    )

with col2:
    st.metric(
        "Nombre de voisins",
        "K = 5"
    )

with col3:
    st.metric(
        "Précision du modèle",
        f"{accuracy * 100:.2f}%"
    )


st.divider()


# ==================================================
# 8. INTERFACE PRINCIPALE
# ==================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# ==================================================
# 9. ZONE DE DESSIN
# ==================================================

with left:

    with st.container(border=True):

        st.subheader("✍️ Dessinez votre chiffre")

        st.caption(
            "Utilisez votre souris pour dessiner un chiffre."
        )

        stroke = st.slider(
            "Épaisseur du stylo",
            min_value=8,
            max_value=30,
            value=18
        )

        canvas_result = st_canvas(

            fill_color="rgba(255,255,255,0)",

            stroke_width=stroke,

            stroke_color="#111111",

            background_color="#FFFFFF",

            height=300,

            width=300,

            drawing_mode="freedraw",

            update_streamlit=True,

            key=f"canvas_{st.session_state.canvas_key}"

        )

        st.info(
            "Conseil : dessinez un seul chiffre "
            "au centre de la zone."
        )

        col_a, col_b = st.columns(2)

        with col_a:

            predict_clicked = st.button(
                "🔍 Reconnaître",
                type="primary",
                use_container_width=True
            )

        with col_b:

            clear_clicked = st.button(
                "↻ Effacer",
                use_container_width=True
            )


# ==================================================
# 10. EFFACER LE DESSIN
# ==================================================

if clear_clicked:

    st.session_state.canvas_key += 1

    st.session_state.prediction = None

    st.rerun()


# ==================================================
# 11. RECONNAISSANCE KNN
# ==================================================

if predict_clicked:

    processed = None

    try:

        if canvas_result is not None:

            # Vérifier que le canvas contient un dessin
            json_data = canvas_result.json_data

            if (
                json_data is not None
                and len(json_data.get("objects", [])) > 0
            ):

                # Lecture de l'image uniquement après
                # vérification du contenu du canvas
                image_data = canvas_result.image_data

                if image_data is not None:

                    processed = prepare_drawing(
                        image_data
                    )

    except Exception as error:

        st.error(
            f"Impossible de lire le dessin : {error}"
        )

        processed = None

    if processed is None:

        st.warning(
            "Veuillez dessiner un chiffre avant "
            "de lancer la reconnaissance."
        )

        st.session_state.prediction = None

    else:

        sample, preview = processed

        # Prédiction
        predicted_digit = int(
            model.predict(sample)[0]
        )

        # Probabilités
        probabilities = model.predict_proba(sample)[0]

        confidence = float(
            np.max(probabilities)
        )

        # Sauvegarde du résultat
        st.session_state.prediction = {

            "digit": predicted_digit,

            "confidence": confidence,

            "probabilities": probabilities,

            "preview": preview

        }


# ==================================================
# 12. AFFICHAGE RESULTAT
# ==================================================

with right:

    with st.container(border=True):

        st.subheader("🎯 Résultat de la reconnaissance")

        prediction = st.session_state.prediction

        if prediction is not None:

            st.success("Chiffre reconnu avec succès !")

            st.markdown(
                f"<h1 style='text-align:center;"
                f"font-size:100px;color:#4f46e5;'>"
                f"{prediction['digit']}</h1>",
                unsafe_allow_html=True
            )

            st.metric(
                "Confiance du modèle",
                f"{prediction['confidence'] * 100:.1f}%"
            )

            st.divider()

            st.write("Image après prétraitement (8 × 8)")

            preview = (
                prediction["preview"] * 255 / 16
            ).clip(0, 255).astype(np.uint8)

            st.image(
                preview,
                width=150
            )

            st.divider()

            st.write("Probabilités par chiffre")

            df = pd.DataFrame({

                "Chiffre": [
                    str(i) for i in range(10)
                ],

                "Probabilité": prediction["probabilities"]

            }).set_index("Chiffre")

            st.bar_chart(
                df,
                y="Probabilité"
            )

        else:

            st.info(
                "En attente de votre dessin. "
                "Le résultat apparaîtra ici."
            )

            st.markdown(
                "### 🔢",
            )

            st.caption(
                "Dessinez un chiffre puis cliquez "
                "sur Reconnaître."
            )


# ==================================================
# 13. EXPLICATION DU FONCTIONNEMENT
# ==================================================

st.divider()

with st.expander("ℹ️ Comprendre le fonctionnement de KNN"):

    st.markdown("""
    **1. Dataset**

    Utilisation de `load_digits()` de Scikit-learn,
    contenant des images de chiffres manuscrits de 0 à 9.

    **2. Prétraitement**

    Le dessin est converti en niveaux de gris,
    recadré et redimensionné en 8 × 8 pixels.

    **3. Classification KNN**

    Le modèle recherche les 5 exemples les plus proches
    du dessin de l'utilisateur.

    **4. Prédiction**

    Le modèle effectue un vote pour identifier
    le chiffre le plus probable.

    **5. Résultat**

    L'application affiche le chiffre reconnu
    et les probabilités estimées par le modèle.

    """)


# ==================================================
# 14. FOOTER
# ==================================================

st.divider()

st.caption(
    "Machine Learning Project | KNN Algorithm | "
    "Python - Scikit-learn - Streamlit"
)
