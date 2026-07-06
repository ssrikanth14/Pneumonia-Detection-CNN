import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# Configure Streamlit Page
# ==========================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# Title
# ==========================================

st.title("🩺 Pneumonia Detection using Deep Learning")

st.markdown("""
Upload a **Chest X-ray** image and the trained CNN model
will predict whether the patient has **Pneumonia**
or is **Normal**.
""")

# ==========================================
# Load Trained Model
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/pneumonia_detection_model.keras")

model = load_model()

# ==========================================
# Image Preprocessing
# ==========================================

def preprocess_image(image):

    # Convert image to RGB (important)
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy array
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# ==========================================
# Upload Image
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# Prediction
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing X-ray..."):

            processed_image = preprocess_image(image)

            prediction = model.predict(processed_image)

            probability = prediction[0][0]

        st.divider()

        if probability >= 0.5:

            st.error("## 🫁 Prediction: PNEUMONIA")

            st.metric(
                label="Confidence",
                value=f"{probability*100:.2f}%"
            )

        else:

            st.success("## ✅ Prediction: NORMAL")

            st.metric(
                label="Confidence",
                value=f"{(1-probability)*100:.2f}%"
            )

        st.divider()

        st.subheader("Prediction Probability")

        st.progress(float(probability))

        st.write(f"**Pneumonia Probability:** {probability:.4f}")
        st.write(f"**Normal Probability:** {(1-probability):.4f}")

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.caption(
    "Developed using TensorFlow, Keras and Streamlit | Pneumonia Detection using CNN"
)