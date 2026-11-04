
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
    page_title="Reconnaissance de Chiffres",
    page_icon="✍️",
    layout="wide"
)


# =====================================================
# 2. MODELE KNN
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

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, X, y


model, accuracy, X, y = train_model()


# =====================================================
# 3. SESSION STATE
# =====================================================

defaults = {
    "canvas_key": 0,
    "prediction": None,
    "mode": "interactive",
    "language": "FR",
    "show_error": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# 4. TRADUCTIONS
# =====================================================

translations = {

    "FR": {
        "header": "Reconnaissance de Chiffres",
        "title": "Reconnaissance de Chiffres Manuscrits",
        "subtitle": "Dessinez un chiffre de 0 à 9 dans la boîte ci-dessous, puis cliquez sur Prédire.",
        "draw": "DESSINEZ ICI",
        "result": "RÉSULTAT",
        "thickness": "Épaisseur du stylo",
        "tip": "Conseil : Dessinez un seul chiffre, en grand et bien centré pour de meilleurs résultats.",
        "predict": "Prédire",
        "clear": "Effacer",
        "detected": "Chiffre prédit",
        "confidence": "Confiance",
        "processed": "Image transmise au modèle (8×8)",
        "votes": "VOTES DES VOISINS",
        "neighbors": "voisins",
        "waiting": "En attente de votre dessin...",
        "empty": "Veuillez dessiner un chiffre avant de lancer la prédiction.",
        "error": "Erreur : impossible de lire le dessin. Veuillez réessayer.",
        "example": "Exemple 7",
        "show_error": "Afficher erreur",
        "about": "À propos",
        "interactive": "Interactif",
        "footer": "Modèle : KNN (k=5) entraîné sur le jeu de données scikit-learn digits · Images 8×8",
        "pen": "Épaisseur du trait",
        "success": "Prédiction effectuée avec succès !",
        "initial": "1. État initial (vide)",
        "state2": "2. Résultat (Chiffre 7)",
        "state3": "3. Résultat (Chiffre 3)",
        "state4": "4. État d'erreur",
        "about_text": "Cette application utilise l'algorithme KNN pour reconnaître les chiffres manuscrits. Dessinez un chiffre et le modèle recherchera ses 5 voisins les plus proches.",
        "no_result": "Dessinez un chiffre puis cliquez sur Prédire."
    },

    "EN": {
        "header": "Digit Recognition",
        "title": "Handwritten Digit Recognition",
        "subtitle": "Draw a digit from 0 to 9 in the box below, then click Predict.",
        "draw": "DRAW HERE",
        "result": "RESULT",
        "thickness": "Pen thickness",
        "tip": "Tip: Draw one digit clearly, large and centered for better results.",
        "predict": "Predict",
        "clear": "Clear",
        "detected": "Predicted digit",
        "confidence": "Confidence",
        "processed": "Image sent to model (8×8)",
        "votes": "NEIGHBOR VOTES",
        "neighbors": "neighbors",
        "waiting": "Waiting for your drawing...",
        "empty": "Please draw a digit before making a prediction.",
        "error": "Error: unable to read the drawing. Please try again.",
        "example": "Example 7",
        "show_error": "Show error",
        "about": "About",
        "interactive": "Interactive",
        "footer": "Model: KNN (k=5) trained on scikit-learn digits dataset · 8×8 images",
        "pen": "Pen thickness",
        "success": "Prediction completed successfully!",
        "initial": "1. Initial state (empty)",
        "state2": "2. Result (Digit 7)",
        "state3": "3. Result (Digit 3)",
        "state4": "4. Error state",
        "about_text": "This application uses KNN to recognize handwritten digits. Draw a digit and the model will search for its 5 nearest neighbors.",
        "no_result": "Draw a digit and click Predict."
    }
}

t = translations[st.session_state.language]


# =====================================================
# 5. PRETRAITEMENT
# =====================================================

def prepare_drawing(image_data):

    if image_data is None:
        return None

    image = Image.fromarray(
        image_data.astype(np.uint8)
    ).convert("RGBA")

    background = Image.new(
        "RGBA",
        image.size,
        (255, 255, 255, 255)
    )

    background.alpha_composite(image)

    gray = background.convert("L")

    inverted = Image.eval(
        gray,
        lambda p: 255 - p
    )

    array = np.asarray(inverted, dtype=np.uint8)

    mask = array > 25

    if not np.any(mask):
        return None

    ys, xs = np.where(mask)

    cropped = array[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    h, w = cropped.shape

    scale = 7 / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = Image.fromarray(cropped).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    final_image = Image.new("L", (8, 8), 0)

    offset = (
        (8 - new_w) // 2,
        (8 - new_h) // 2
    )

    final_image.paste(resized, offset)

    digit = np.asarray(
        final_image,
        dtype=np.float32
    )

    if digit.sum() > 0:

        cy, cx = center_of_mass(digit)

        digit = shift(
            digit,
            shift=(3.5 - cy, 3.5 - cx),
            order=1,
            mode="constant",
            cval=0
        )

    digit = np.clip(
        digit / 255 * 16,
        0,
        16
    )

    return digit.reshape(1, 64), digit


# =====================================================
# 6. FONCTION DE PREDICTION
# =====================================================

def make_prediction(sample):

    digit = int(model.predict(sample)[0])

    probabilities = model.predict_proba(sample)[0]

    confidence = float(np.max(probabilities))

    distances, indices = model.kneighbors(sample)

    neighbor_labels = y[indices[0]]

    unique, counts = np.unique(
        neighbor_labels,
        return_counts=True
    )

    votes = sorted(
        zip(unique, counts),
        key=lambda item: item[1],
        reverse=True
    )

    st.session_state.prediction = {
        "digit": digit,
        "confidence": confidence,
        "probabilities": probabilities,
        "image": sample.reshape(8, 8),
        "votes": votes
    }

    st.session_state.show_error = False


# =====================================================
# 7. NAVIGATION
# =====================================================

st.title("✍️ " + t["header"])

nav1, nav2, nav3, nav4, nav5 = st.columns(
    [1.2, 1, 1.2, 0.8, 0.8]
)

with nav1:
    if st.button(t["interactive"], use_container_width=True):
        st.session_state.mode = "interactive"
        st.session_state.prediction = None
        st.session_state.show_error = False
        st.rerun()

with nav2:
    if st.button(t["example"], use_container_width=True):
        st.session_state.mode = "example"
        make_prediction(X[7].reshape(1, 64))
        st.rerun()

with nav3:
    if st.button(t["show_error"], use_container_width=True):
        st.session_state.mode = "error"
        st.session_state.show_error = True
        st.rerun()

with nav4:
    if st.button("🇫🇷 FR", use_container_width=True):
        st.session_state.language = "FR"
        st.rerun()

with nav5:
    if st.button("🇬🇧 EN", use_container_width=True):
        st.session_state.language = "EN"
        st.rerun()

with st.expander("ℹ️ " + t["about"]):
    st.write(t["about_text"])

st.divider()


# =====================================================
# 8. TITRE PRINCIPAL
# =====================================================

st.header(t["title"])

st.caption(t["subtitle"])

st.write("")


# =====================================================
# 9. STATISTIQUES
# =====================================================

stat1, stat2, stat3 = st.columns(3)

stat1.metric("🤖 Algorithm", "KNN")

stat2.metric("👥 Number of Neighbors", "5")

stat3.metric(
    "🎯 Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.write("")


# =====================================================
# 10. LES DEUX RECTANGLES PRINCIPAUX
# =====================================================

left, right = st.columns(2, gap="medium")


# -----------------------------------------------------
# RECTANGLE GAUCHE : DESSIN
# -----------------------------------------------------

with left:

    with st.container(border=True):

        st.subheader("✍️ " + t["draw"])

        st.caption("280 × 280 px")

        stroke = st.slider(
            t["thickness"],
            8,
            30,
            18
        )

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=stroke,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_{st.session_state.canvas_key}"
        )

        st.caption(t["tip"])

        btn1, btn2 = st.columns([2, 1])

        with btn1:
            predict_clicked = st.button(
                "▶ " + t["predict"],
                type="primary",
                use_container_width=True
            )

        with btn2:
            clear_clicked = st.button(
                t["clear"],
                use_container_width=True
            )


# -----------------------------------------------------
# RECTANGLE DROIT : RESULTAT
# -----------------------------------------------------

with right:

    with st.container(border=True):

        st.subheader("🎯 " + t["result"])

        prediction = st.session_state.prediction

        if prediction is not None:

            st.success(t["success"])

            st.markdown(
                f"<h1 style='text-align:center;"
                f"font-size:90px;color:#1677ff;'>"
                f"{prediction['digit']}</h1>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"**{t['detected']} : {prediction['digit']}**"
            )

            st.write(
                f"**{t['confidence']} : "
                f"{prediction['confidence'] * 100:.1f}%**"
            )

            st.progress(prediction["confidence"])

            st.divider()

            st.caption(t["processed"])

            preview = (
                prediction["image"] / 16 * 255
            ).clip(0, 255).astype(np.uint8)

            st.image(
                preview,
                width=100
            )

            st.divider()

            st.caption(t["votes"])

            for label, count in prediction["votes"]:

                st.write(
                    f"Chiffre {label} : "
                    f"{count} / 5 {t['neighbors']}"
                )

                st.progress(count / 5)

        else:

            st.info(t["waiting"])

            st.caption(t["no_result"])


# =====================================================
# 11. TRAITEMENT DES BOUTONS
# =====================================================

if clear_clicked:

    st.session_state.canvas_key += 1

    st.session_state.prediction = None

    st.session_state.show_error = False

    st.rerun()


if predict_clicked:

    if st.session_state.show_error:
        st.session_state.show_error = False

    processed = None

    try:

        if canvas_result is not None:

            json_data = canvas_result.json_data

            if (
                json_data is not None
                and len(json_data.get("objects", [])) > 0
            ):

                image_data = canvas_result.image_data

                if image_data is not None:

                    processed = prepare_drawing(image_data)

    except Exception:

        processed = None

    if processed is None:

        st.warning(t["empty"])

    else:

        sample, preview = processed

        make_prediction(sample)

    st.rerun()


# =====================================================
# 12. ETAT D'ERREUR DEMO
# =====================================================

if st.session_state.show_error:

    st.error(t["error"])


# =====================================================
# 13. ETATS DE DEMONSTRATION
# =====================================================

st.divider()

st.caption("États de démonstration")

demo1, demo2, demo3, demo4 = st.columns(4)

with demo1:

    if st.button(t["initial"], use_container_width=True):

        st.session_state.prediction = None
        st.session_state.show_error = False
        st.session_state.mode = "interactive"
        st.rerun()

with demo2:

    if st.button(t["state2"], use_container_width=True):

        make_prediction(X[7].reshape(1, 64))
        st.session_state.mode = "example"
        st.rerun()

with demo3:

    if st.button(t["state3"], use_container_width=True):

        make_prediction(X[3].reshape(1, 64))
        st.session_state.mode = "example"
        st.rerun()

with demo4:

    if st.button(t["state4"], use_container_width=True):

        st.session_state.show_error = True
        st.session_state.prediction = None
        st.rerun()


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(t["footer"])
