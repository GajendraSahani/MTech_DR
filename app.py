import os
import cv2
import numpy as np
import pandas as pd
import PIL.Image
import plotly.express as px
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Diabetic Retinopathy Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EFF_SIZE = (224, 224)

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

CLASS_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Microaneurysms only. Early stage condition.",
    2: "More than microaneurysms, but less than severe DR.",
    3: "Severe intraretinal hemorrhages or microvascular abnormalities.",
    4: "Neovascularization or vitreous/preretinal hemorrhage.",
}

CLASS_COLORS = {
    0: "#4CAF50",
    1: "#2196F3",
    2: "#FF9800",
    3: "#F44336",
    4: "#9C27B0",
}


def crop_black_border(img, tol=7):
    """Crops empty black margins from eye fundus photos."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = gray > tol
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        img = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return img


def gaussian_blur_subtraction(img, sigma=10):
    """Applies Gaussian blur subtraction to enhance vessel contrast."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def enhance_brightness_contrast(img, alpha=1.2, beta=10):
    """Adjusts contrast (alpha) and brightness (beta)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def preprocess_fundus_image(pil_img):
    """Full preprocessing pipeline matching notebook specifications."""

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cropped = crop_black_border(img)
    blurred = gaussian_blur_subtraction(cropped)
    enhanced = enhance_brightness_contrast(blurred)

    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    resized_img = cv2.resize(enhanced_rgb, EFF_SIZE)

    return resized_img, enhanced_rgb

def build_head_model():
    """Reconstructs the classification head architecture."""
    inputs = tf.keras.Input(shape=(1792,), name="features")
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(5e-5))(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(5e-5))(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(5e-5))(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


@st.cache_resource
def load_dr_model():

    model_dir = "models"
    full_model_path = os.path.join(model_dir, "eff_finetuned_final.keras")
    head_model_path = os.path.join(model_dir, "fl_best.keras")

    if os.path.exists(full_model_path):
        model = tf.keras.models.load_model(full_model_path, compile=False)
        return "full", model

    base_backbone = tf.keras.applications.EfficientNetB4(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg"
    )

    if os.path.exists(head_model_path):
        head = build_head_model()
        head.load_weights(head_model_path)
    else:
        head = build_head_model()

    return "split", (base_backbone, head)


def predict(processed_img, model_tuple, use_tta=False, tta_steps=8):
    """Runs inference on processed image batch with optional Test-Time Augmentation (TTA)."""

    img_array = np.expand_dims(processed_img, axis=0).astype(np.float32)
    input_tensor = tf.keras.applications.efficientnet.preprocess_input(img_array)

    model_type, model_obj = model_tuple

    def get_probs(x_input):
        if model_type == "full":
            return model_obj(x_input, training=False).numpy()
        else:
            backbone, head = model_obj
            features = backbone(x_input, training=False)
            return head(features, training=False).numpy()

    base_probs = get_probs(input_tensor)[0]

    if not use_tta:
        return base_probs

    all_preds = [base_probs]
    for _ in range(tta_steps - 1):
        noise = np.random.normal(0, 0.02, input_tensor.shape).astype(np.float32)
        all_preds.append(get_probs(input_tensor + noise)[0])

    return np.mean(all_preds, axis=0)

def main():
    st.sidebar.title("DR Screening Panel")
    st.sidebar.markdown(
        "Upload a fundus photograph of the retina to run automated AI screening for Diabetic Retinopathy (DR)."
    )

    st.sidebar.subheader("Model Settings")
    use_tta = st.sidebar.checkbox(
        "Enable TTA (Test-Time Augmentation)",
        value=True,
        help="Averages multiple perturbed passes for enhanced accuracy.",
    )
    tta_steps = (
        st.sidebar.slider("TTA Steps", min_value=2, max_value=16, value=8)
        if use_tta
        else 8
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model Architecture:**\n"
        "- **Backbone:** EfficientNetB4\n"
        "- **Training:** Federated Learning (FedProx)\n"
        "- **Preprocessing:** Gaussian Blur Subtraction + Contrast Normalization"
    )

    st.title("Automated Diabetic Retinopathy Classification")
    st.markdown(
        "This application uses a deep learning model trained with **Federated Learning** and **EfficientNetB4** "
        "to screen retinal fundus images into 5 severity stages."
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a Retinal Fundus Image...", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            image = PIL.Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        col1, col2 = st.columns([1, 1])

        # Preprocess Image
        with st.spinner("Applying visual enhancement & preprocessing..."):
            processed_img, enhanced_display = preprocess_fundus_image(image)

        with col1:
            st.subheader("Input vs Preprocessed Image")
            img_tabs = st.tabs(["Preprocessed (Model Input)", "Original Raw Image"])
            with img_tabs[0]:
                st.image(
                    enhanced_display,
                    caption="Enhanced Retinal View (Gaussian Subtraction)",
                    use_column_width=True,
                )
            with img_tabs[1]:
                st.image(image, caption="Original Uploaded Image", use_column_width=True)

        with st.spinner("Loading AI model & analyzing fundus image..."):
            model_tuple = load_dr_model()
            probs = predict(
                processed_img, model_tuple, use_tta=use_tta, tta_steps=tta_steps
            )
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class]) * 100

        with col2:
            st.subheader("Diagnostic Results")

            pred_label = CLASS_NAMES[pred_class]
            color = CLASS_COLORS[pred_class]

            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="margin:0; color: white;">Stage {pred_class}: {pred_label}</h2>
                    <h4 style="margin:0; opacity: 0.9; color: white;">Confidence: {confidence:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown(f"**Description:** {CLASS_DESCRIPTIONS[pred_class]}")

            df_probs = pd.DataFrame(
                {
                    "Stage": [f"Stage {k}: {v}" for k, v in CLASS_NAMES.items()],
                    "Probability (%)": probs * 100,
                    "Color": [CLASS_COLORS[i] for i in range(5)],
                }
            )

            fig = px.bar(
                df_probs,
                x="Probability (%)",
                y="Stage",
                orientation="h",
                text="Probability (%)",
                color="Stage",
                color_discrete_map={
                    f"Stage {k}: {v}": CLASS_COLORS[k] for k, v in CLASS_NAMES.items()
                },
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )
            fig.update_layout(
                showlegend=False,
                xaxis=dict(range=[0, 115]),
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.warning(
                "**Disclaimer:** This tool is an AI diagnostic assistance prototype and should not be used as a sole basis for clinical diagnosis. Please consult a licensed ophthalmologist."
            )


if __name__ == "__main__":
    main()
