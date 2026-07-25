import os
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


IMAGE_SIZE = (224, 224)

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

CLASS_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Microaneurysms only. Early stage diabetic retinopathy.",
    2: "Moderate diabetic retinopathy with increasing retinal abnormalities.",
    3: "Severe diabetic retinopathy with extensive retinal damage.",
    4: "Proliferative diabetic retinopathy with abnormal blood vessel growth.",
}

CLASS_COLORS = {
    0: "#4CAF50",
    1: "#2196F3",
    2: "#FF9800",
    3: "#F44336",
    4: "#9C27B0",
}

@st.cache_resource
def load_model():

    model_path = os.path.join(
        "models",
        "eff_finetuned_final.keras"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


def prepare_image(image):


    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    img = np.asarray(image).astype(np.float32)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img

def predict(model, image):

    input_tensor = prepare_image(image)

    predictions = model(
        input_tensor,
        training=False
    ).numpy()[0]

    predicted_class = int(np.argmax(predictions))

    confidence = float(
        predictions[predicted_class] * 100
    )

    return (
        predictions,
        predicted_class,
        confidence
    )


def main():


    st.sidebar.title("👁️ Diabetic Retinopathy Screening")

    st.sidebar.markdown(
        """
        Upload a retinal fundus image and let the trained
        EfficientNetB4 model classify the stage of
        diabetic retinopathy.
        """
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("Model Information")

    st.sidebar.info(
        """
        **Architecture:** EfficientNetB4

        **Classes:** 5

        **Input Size:** 224 × 224

        **Training:** Federated Learning

        **Inference:** Direct Model Prediction

        **Image Enhancement:** None

        **Test-Time Augmentation:** Disabled
        """
    )

    st.sidebar.markdown("---")


    st.title("👁️ Automated Diabetic Retinopathy Classification")

    st.markdown(
        """
        This application performs **direct inference**
        using the trained **EfficientNetB4** model.

        The uploaded image is **not manipulated** before
        prediction except for resizing to the model's
        required input size.
        """
    )

    st.markdown("---")


    try:

        model = load_model()

    except Exception as e:

        st.error(f"Unable to load model.\n\n{e}")

        st.stop()


    uploaded_file = st.file_uploader(
        "Upload Retinal Fundus Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:

        st.info("Please upload a retinal fundus image.")

        return

    try:

        image = PIL.Image.open(uploaded_file).convert("RGB")

    except Exception as e:

        st.error(f"Unable to open image.\n\n{e}")

        return


    with st.spinner("Running AI model..."):

        probs, pred_class, confidence = predict(
            model,
            image
        )


    col1, col2 = st.columns([1, 1])


    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )


    with col2:

        st.subheader("Prediction")

        label = CLASS_NAMES[pred_class]

        color = CLASS_COLORS[pred_class]

        st.markdown(
            f"""
            <div style="
                background:{color};
                padding:18px;
                border-radius:12px;
                color:white;
                text-align:center;
            ">
                <h2 style="margin:0;">
                    {label}
                </h2>

                <h3 style="margin-top:10px;">
                    Confidence: {confidence:.2f}%
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            f"### Description\n\n{CLASS_DESCRIPTIONS[pred_class]}"
        )

    st.markdown("---")

    st.subheader("Prediction Probability")

    df = pd.DataFrame({

        "Stage": [
            f"Stage {i}: {CLASS_NAMES[i]}"
            for i in range(5)
        ],

        "Probability": probs * 100,

    })

    fig = px.bar(

        df,

        x="Probability",

        y="Stage",

        orientation="h",

        text="Probability",

        color="Stage",

        color_discrete_map={
            f"Stage {i}: {CLASS_NAMES[i]}":
            CLASS_COLORS[i]
            for i in range(5)
        }

    )

    fig.update_traces(

        texttemplate="%{text:.2f}%",

        textposition="outside"

    )

    fig.update_layout(

        height=420,

        showlegend=False,

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),

        xaxis=dict(
            range=[0, 100]
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    with st.expander("Raw Model Output"):

        output_df = pd.DataFrame({

            "Class": list(CLASS_NAMES.values()),

            "Probability (%)": np.round(
                probs * 100,
                4
            )

        })

        st.dataframe(
            output_df,
            use_container_width=True
        )


    st.markdown("---")

    st.warning(
        """
        **Medical Disclaimer**

        This application is intended for research and
        educational purposes only.

        Predictions are generated solely by the trained
        deep learning model and should **not** be used
        as a substitute for professional diagnosis by
        an ophthalmologist.
        """
    )


if __name__ == "__main__":
    main()
