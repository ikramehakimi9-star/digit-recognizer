
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
from scipy.ndimage import center_of_mass, shift

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
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
# 2. ENTRAINEMENT ET OPTIMISATION KNN
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

    # Recherche des meilleurs paramètres
    parameters = {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    search = GridSearchCV(
        KNeighborsClassifier(),
        parameters,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    # Meilleur modèle
    model = search.best_estimator_

    # Evaluation
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return (
        model,
        accuracy,
        X,
        y,
        X_train,
        y_train
    )


model, accuracy, X, y, X_train, y_train = train_model()

K = model.n_neighbors


# =====================================================
# 3. SESSION STATE
# =====================================================

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "result" not in st.session_state:
    st.session_state.result = None

if "demo_error" not in st.session_state:
    st.session_state.demo_error = False

if "pending_demo" not in st.session_state:
    st.session_state.pending_demo = None

if "language" not in st.session_state:
    st.session_state.language = "FR"


# =====================================================
# 4. TRADUCTIONS
# =====================================================

translations = {

    "FR": {
        "title": "Reconnaissance de Chiffres Manuscrits",
        "subtitle": "Dessinez un chiffre de 0 à 9 dans la boîte ci-dessous, puis cliquez sur Prédire.",
        "draw": "DESSINEZ ICI",
        "result": "RÉSULTAT",
        "instruction": "Dessinez le chiffre en grand et bien centré pour de meilleurs résultats.",
        "predict": "▶ Prédire",
        "clear": "Effacer",
        "digit": "Chiffre prédit",
        "agreement": "Accord des voisins",
        "processed": "Image transmise au modèle (8×8)",
        "votes": "VOTES DES VOISINS",
        "neighbors": "voisins",
        "empty": "Votre résultat apparaîtra ici après la prédiction.",
        "error": "Aucun chiffre détecté. Dessinez d'abord un chiffre.",
        "initial": "État initial (vide)",
        "state7": "Résultat (Chiffre 7)",
        "state3": "Résultat (Chiffre 3)",
        "error_state": "État d'erreur",
        "about": "À propos",
        "about_text": "Application de reconnaissance de chiffres manuscrits utilisant l'algorithme KNN et le dataset Digits de Scikit-learn.",
        "accuracy": "Précision du modèle",
        "algorithm": "Algorithme",
        "number_neighbors": "Nombre de voisins",
        "tip": "Conseil : écrivez un seul chiffre clairement au centre.",
        "success": "Chiffre reconnu avec succès !",
        "waiting": "En attente de votre dessin",
        "uncertain": "Prédiction incertaine : les voisins sont partagés.",
        "pen": "Épaisseur du stylo",
        "distance": "Métrique de distance",
        "weights": "Pondération",
        "example": 'Exemple "7"',
        "show_error": "Afficher erreur",
        "footer": "Modèle KNN entraîné sur Scikit-learn Digits"
    },

    "EN": {
        "title": "Handwritten Digit Recognition",
        "subtitle": "Draw a digit from 0 to 9 in the box below, then click Predict.",
        "draw": "DRAW HERE",
        "result": "RESULT",
        "instruction": "Draw a large and centered digit for better results.",
        "predict": "▶ Predict",
        "clear": "Clear",
        "digit": "Predicted digit",
        "agreement": "Neighbor agreement",
        "processed": "Image sent to model (8×8)",
        "votes": "NEIGHBOR VOTES",
        "neighbors": "neighbors",
        "empty": "Your result will appear here after prediction.",
        "error": "No digit detected. Please draw a digit first.",
        "initial": "Initial state (empty)",
        "state7": "Result (Digit 7)",
        "state3": "Result (Digit 3)",
        "error_state": "Error state",
        "about": "About",
        "about_text": "Handwritten digit recognition application using KNN and the Scikit-learn Digits dataset.",
        "accuracy": "Model Accuracy",
        "algorithm": "Algorithm",
        "number_neighbors": "Number of Neighbors",
        "tip": "Tip: Write one digit clearly in the center.",
        "success": "Digit successfully recognized!",
        "waiting": "Waiting for your drawing",
        "uncertain": "Uncertain prediction: neighbors disagree.",
        "pen": "Pen thickness",
        "distance": "Distance metric",
        "weights": "Weighting",
        "example": 'Example "7"',
        "show_error": "Show error",
        "footer": "KNN model trained on Scikit-learn Digits"
    }
}

lang = translations[st.session_state.language]


# =====================================================
# 5. PRETRAITEMENT DES IMAGES
# =====================================================

def prepare_drawing(image_data):

    if image_data is None:
        return None

    image = Image.fromarray(
        image_data.astype(np.uint8)
    ).convert("RGBA")

    # Fond noir
    background = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 255)
    )

    background.alpha_composite(image)

    gray = background.convert("L")

    array = np.asarray(
        gray,
        dtype=np.uint8
    )

    # Détection du chiffre
    mask = array > 30

    if not np.any(mask):
        return None

    # Recadrage
    ys, xs = np.where(mask)

    cropped = array[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    # Redimensionnement avec marge
    h, w = cropped.shape

    scale = 6.0 / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = Image.fromarray(cropped).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    # Création 8x8
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

    # Centrage automatique
    if digit.sum() > 0:

        cy, cx = center_of_mass(digit)

        digit = shift(
            digit,
            shift=(3.5 - cy, 3.5 - cx),
            order=1,
            mode="constant",
            cval=0
        )

    # Normalisation 0-16
    digit = np.clip(
        digit / 255.0 * 16.0,
        0,
        16
    )

    return digit.reshape(1, 64), digit


# =====================================================
# 6. PREDICTION ET VOTES DES VOISINS
# =====================================================

def make_prediction(sample):

    # Prédiction KNN
    predicted_digit = int(
        model.predict(sample)[0]
    )

    # Recherche des voisins les plus proches
    distances, indices = model.kneighbors(sample)

    neighbor_labels = y_train[indices[0]]

    neighbor_distances = distances[0]

    # Comptage des votes
    votes = {}

    for label in neighbor_labels:

        label = int(label)

        votes[label] = votes.get(label, 0) + 1

    # Accord des voisins
    agreement = max(votes.values()) / K

    # Sauvegarde du résultat
    st.session_state.result = {

        "digit": predicted_digit,

        "agreement": agreement,

        "votes": votes,

        "distances": neighbor_distances,

        "preview": sample.reshape(8, 8)

    }

    st.session_state.demo_error = False


# =====================================================
# 7. EXEMPLES DU DATASET
# =====================================================

def load_example(digit):

    indices = np.where(y == digit)[0]

    sample = X[indices[0]].reshape(1, 64)

    make_prediction(sample)


# =====================================================
# 8. DEMONSTRATIONS
# =====================================================

if st.session_state.pending_demo is not None:

    demo = st.session_state.pending_demo

    st.session_state.canvas_key += 1

    if demo == "7":

        load_example(7)

    elif demo == "3":

        load_example(3)

    elif demo == "initial":

        st.session_state.result = None
        st.session_state.demo_error = False

    elif demo == "error":

        st.session_state.result = None
        st.session_state.demo_error = True

    st.session_state.pending_demo = None


# =====================================================
# 9. BARRE SUPERIEURE
# =====================================================

header1, header2 = st.columns([2, 3])

with header1:

    st.markdown("### ✍️ Reconnaissance de Chiffres")

with header2:

    nav1, nav2, nav3, nav4 = st.columns(
        [1, 1, 1, 1]
    )

    with nav1:

        st.caption("Mode interactif")

    with nav2:

        if st.button(
            lang["example"],
            use_container_width=True
        ):

            st.session_state.pending_demo = "7"
            st.rerun()

    with nav3:

        if st.button(
            lang["show_error"],
            use_container_width=True
        ):

            st.session_state.pending_demo = "error"
            st.rerun()

    with nav4:

        selected_language = st.selectbox(
            "Langue",
            ["FR", "EN"],
            index=0 if st.session_state.language == "FR" else 1,
            label_visibility="collapsed"
        )

        if selected_language != st.session_state.language:

            st.session_state.language = selected_language
            st.rerun()


with st.expander("ℹ️ " + lang["about"]):

    st.write(lang["about_text"])


st.divider()


# =====================================================
# 10. TITRE PRINCIPAL
# =====================================================

st.title(lang["title"])

st.caption(lang["subtitle"])


# =====================================================
# 11. INFORMATIONS MODELE
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        lang["algorithm"],
        "KNN"
    )

with col2:

    st.metric(
        lang["number_neighbors"],
        str(K)
    )

with col3:

    st.metric(
        lang["accuracy"],
        f"{accuracy * 100:.2f}%"
    )


# =====================================================
# 12. PANNEAUX PRINCIPAUX
# =====================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# =====================================================
# 13. PANNEAU DESSIN
# =====================================================

with left:

    with st.container(border=True):

        draw_header, draw_size = st.columns([3, 1])

        with draw_header:

            st.markdown("**✍️ " + lang["draw"] + "**")

        with draw_size:

            st.caption("280 × 280 px")

        stroke = st.slider(
            lang["pen"],
            min_value=8,
            max_value=38,
            value=18,
            step=2
        )

        canvas_result = st_canvas(

            fill_color="rgba(0,0,0,0)",

            stroke_width=stroke,

            stroke_color="#FFFFFF",

            background_color="#000000",

            height=280,

            width=280,

            drawing_mode="freedraw",

            update_streamlit=True,

            key=f"canvas_{st.session_state.canvas_key}"

        )

        st.caption(lang["instruction"])

        st.caption("💡 " + lang["tip"])

        st.write("")

        btn1, btn2 = st.columns([2, 1])

        with btn1:

            predict_clicked = st.button(
                lang["predict"],
                type="primary",
                use_container_width=True
            )

        with btn2:

            clear_clicked = st.button(
                lang["clear"],
                use_container_width=True
            )

        st.caption("🖱️ Dessin tactile et souris supportés.")


# =====================================================
# 14. ACTIONS DESSIN
# =====================================================

if clear_clicked:

    st.session_state.canvas_key += 1

    st.session_state.result = None

    st.session_state.demo_error = False

    st.rerun()


if predict_clicked:

    processed = None

    try:

        if canvas_result is not None:

            image_data = canvas_result.image_data

            if image_data is not None:

                processed = prepare_drawing(image_data)

    except (RuntimeError, ValueError, OSError, AttributeError):

        processed = None

    if processed is None:

        st.session_state.result = None

        st.session_state.demo_error = True

    else:

        sample, preview = processed

        make_prediction(sample)


# =====================================================
# 15. PANNEAU RESULTAT
# =====================================================

with right:

    with st.container(border=True):

        result_header, result_badge = st.columns([3, 1])

        with result_header:

            st.markdown("**🎯 " + lang["result"] + "**")

        result = st.session_state.result

        if result is not None:

            st.success(lang["success"])

            st.markdown(
                f"<h1 style='text-align:center; font-size:90px;'>"
                f"{result['digit']}</h1>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"**{lang['digit']}**",
                text_alignment="center"
            )

            st.write(
                f"**{lang['agreement']} : "
                f"{result['agreement'] * 100:.1f}%**"
            )

            if result["agreement"] < 0.6:

                st.warning(lang["uncertain"])

            st.divider()

            st.caption(lang["processed"])

            preview = (
                result["preview"] * 255 / 16
            ).clip(0, 255).astype(np.uint8)

            st.image(
                preview,
                width=110
            )

            st.divider()

            st.caption(lang["votes"])

            # Affichage des votes réels
            sorted_votes = sorted(
                result["votes"].items(),
                key=lambda item: (-item[1], item[0])
            )

            for digit, count in sorted_votes:

                st.write(
                    f"Chiffre {digit} : "
                    f"{count} / {K} {lang['neighbors']}"
                )

                st.progress(
                    count / K
                )

        elif st.session_state.demo_error:

            st.error(lang["error"])

        else:

            st.info(lang["waiting"])

            st.markdown("### 🔢")

            st.caption(lang["empty"])


# =====================================================
# 16. ETATS DE DEMONSTRATION
# =====================================================

st.divider()

st.markdown("**États de démonstration :**")

demo1, demo2, demo3, demo4 = st.columns(4)

with demo1:

    if st.button(
        "1. " + lang["initial"],
        use_container_width=True
    ):

        st.session_state.pending_demo = "initial"
        st.rerun()

with demo2:

    if st.button(
        "2. " + lang["state7"],
        use_container_width=True
    ):

        st.session_state.pending_demo = "7"
        st.rerun()

with demo3:

    if st.button(
        "3. " + lang["state3"],
        use_container_width=True
    ):

        st.session_state.pending_demo = "3"
        st.rerun()

with demo4:

    if st.button(
        "4. " + lang["error_state"],
        use_container_width=True
    ):

        st.session_state.pending_demo = "error"
        st.rerun()


# =====================================================
# 17. INFORMATIONS TECHNIQUES
# =====================================================

st.divider()

st.caption(
    f"{lang['footer']} · "
    f"K={K} · "
    f"Métrique={model.metric} · "
    f"Pondération={model.weights} · "
    "Matrice de caractéristiques 8×8"
)
