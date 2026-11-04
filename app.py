
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
    page_title="Digit Recognition | KNN",
    page_icon="✍️",
    layout="wide"
)


# ==================================================
# 2. STYLE CSS
# ==================================================

st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f5f6fc;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Titles */
h1, h2, h3 {
    color: #25245e;
}

/* Buttons */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 43px;
}

div.stButton > button[kind="primary"] {
    background-color: #4f46e5;
    color: white;
    border: none;
}

/* Statistics cards */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Main columns alignment */
div[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
    gap: 24px;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    display: flex;
    flex-direction: column;
}

/* Equal height for the two main panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    min-height: 680px;
    height: 100%;
    box-sizing: border-box;
    border-radius: 12px;
}

/* Panel padding */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 18px;
}

/* Canvas centered */
div[data-testid="stCanvas"] {
    margin-left: auto;
    margin-right: auto;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. TRAIN KNN MODEL
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
# 4. IMAGE PREPROCESSING
# ==================================================

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
        lambda pixel: 255 - pixel
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

    scale = 7.0 / max(h, w)

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
        digit / 255.0 * 16.0,
        0,
        16
    )

    return digit.reshape(1, 64), digit


# ==================================================
# 5. SESSION STATE
# ==================================================

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ==================================================
# 6. HEADER
# ==================================================

st.title("✍️ Handwritten Digit Recognition")

st.caption(
    "Write a digit from 0 to 9 and let KNN recognize it."
)

st.divider()


# ==================================================
# 7. STATISTICS CARDS
# ==================================================

col1, col2, col3 = st.columns(3, gap="medium")

with col1:

    st.metric(
        label="🤖 Algorithm",
        value="KNN"
    )

with col2:

    st.metric(
        label="👥 Number of Neighbors",
        value="5"
    )

with col3:

    st.metric(
        label="🎯 Model Accuracy",
        value=f"{accuracy * 100:.2f}%"
    )


st.write("")


# ==================================================
# 8. MAIN INTERFACE
# ==================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# ==================================================
# 9. DRAWING PANEL
# ==================================================

with left:

    with st.container(border=True):

        st.subheader("✍️ Draw your digit")

        st.caption(
            "Use your mouse or touch screen to draw a number."
        )

        stroke = st.slider(
            "Pen thickness",
            min_value=8,
            max_value=30,
            value=18,
            step=2
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

        st.caption(
            "💡 Tip: Draw one digit clearly in the center."
        )

        st.write("")

        btn1, btn2 = st.columns(2)

        with btn1:

            predict_clicked = st.button(
                "🔍 Recognize",
                type="primary",
                use_container_width=True
            )

        with btn2:

            clear_clicked = st.button(
                "↻ Clear",
                use_container_width=True
            )


# ==================================================
# 10. CLEAR DRAWING
# ==================================================

if clear_clicked:

    st.session_state.canvas_key += 1

    st.session_state.prediction = None

    st.rerun()


# ==================================================
# 11. KNN PREDICTION
# ==================================================

if predict_clicked:

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

        st.session_state.prediction = None

        st.warning(
            "Please draw a digit before clicking Recognize."
        )

    else:

        sample, preview = processed

        predicted_digit = int(
            model.predict(sample)[0]
        )

        probabilities = model.predict_proba(sample)[0]

        confidence = float(np.max(probabilities))

        st.session_state.prediction = {

            "digit": predicted_digit,

            "confidence": confidence,

            "probabilities": probabilities,

            "preview": preview

        }


# ==================================================
# 12. RESULT PANEL
# ==================================================

with right:

    with st.container(border=True):

        st.subheader("🎯 Recognition Result")

        prediction = st.session_state.prediction

        if prediction is not None:

            st.success("Digit successfully recognized!")

            st.markdown(
                f"""
                <h1 style="
                    text-align:center;
                    font-size:100px;
                    color:#4f46e5;
                    margin:0;
                ">
                    {prediction["digit"]}
                </h1>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Confidence",
                f'{prediction["confidence"] * 100:.1f}%'
            )

            st.divider()

            st.write("**Processed image (8 × 8)**")

            preview = (
                prediction["preview"] * 255 / 16
            ).clip(0, 255).astype(np.uint8)

            st.image(
                preview,
                width=120
            )

            st.divider()

            st.write("**Prediction probabilities**")

            df = pd.DataFrame({

                "Digit": [str(i) for i in range(10)],

                "Probability": prediction["probabilities"]

            }).set_index("Digit")

            st.bar_chart(
                df,
                y="Probability",
                height=200
            )

        else:

            st.info(
                "Your result will appear here after recognition."
            )

            st.write("")

            st.markdown("### 🔢")

            st.caption(
                "Draw a digit and click on Recognize to see the result."
            )


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.caption(
    "Machine Learning Project | KNN | Python & Streamlit"
)
