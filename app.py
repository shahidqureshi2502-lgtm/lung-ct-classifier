import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Lung CT Classifier",
    page_icon="🫁",
    layout="centered"
)

MODEL_PATH = Path("lung_ct_model.keras")
LABELS_PATH = Path("class_names.json")
IMG_SIZE = (224, 224)

# ------------------------------------------------------------
# LOAD MODEL AND LABELS
# ------------------------------------------------------------

@st.cache_resource
def load_classifier():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found."
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Class labels file not found."
        )

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH
    )

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    if len(labels) != loaded_model.output_shape[-1]:
        raise ValueError(
            "Number of labels does not match model output."
        )

    return loaded_model, labels


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("🫁 Lung CT Image Classifier")

st.write(
    "Upload a CT image to see the prediction made "
    "by the experimental image-classification model."
)

st.warning(
    "Educational demonstration only. This model is not "
    "validated for medical diagnosis. Do not use its "
    "predictions to make healthcare decisions."
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

try:
    model, class_names = load_classifier()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ------------------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CT image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

    except Exception:
        st.error("The uploaded file is not a valid image.")
        st.stop()

    st.subheader("Uploaded image")

    st.image(
        image,
        caption="Image supplied for classification",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "Classify image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Processing image..."):

            resized = image.resize(
                IMG_SIZE,
                Image.Resampling.BILINEAR
            )

            image_array = np.asarray(
                resized,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            probabilities = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            predicted_label = class_names[
                predicted_index
            ]

            score = float(
                probabilities[predicted_index]
            )

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        st.subheader("Model output")

        st.metric(
            label="Predicted class",
            value=predicted_label
        )

        st.metric(
            label="Model score",
            value=f"{score * 100:.2f}%"
        )

        st.caption(
            "The score is the model's softmax output, "
            "not a calibrated probability of disease."
        )

        st.subheader("Scores for all classes")

        for index, label in enumerate(class_names):

            st.write(
                f"**{label}:** "
                f"{probabilities[index] * 100:.2f}%"
            )

            st.progress(
                float(probabilities[index])
            )

        st.info(
            "This result is a research demonstration, "
            "not a medical assessment."
        )

else:
    st.info(
        "Upload a CT image to begin."
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Lung CT Classifier | Experimental educational project"
)