import os
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import PIL.Image
import plotly.express as px
import streamlit as st
import tensorflow as tf

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


st.set_page_config(
    page_title="RetinaScan AI — Diabetic Retinopathy Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #1a1a4e, #16213e, #0f3460, #1a1a2e);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
        color: #e6edf3;
        font-family: 'Space Grotesk', sans-serif;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp::before {
        content: ""; position: fixed; top: -10%; left: -10%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(0,229,255,0.25) 0%, transparent 70%);
        border-radius: 50%; filter: blur(60px); z-index: 0;
        animation: float1 20s ease-in-out infinite; pointer-events: none;
    }
    .stApp::after {
        content: ""; position: fixed; bottom: -10%; right: -10%;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(255,64,129,0.20) 0%, transparent 70%);
        border-radius: 50%; filter: blur(70px); z-index: 0;
        animation: float2 25s ease-in-out infinite; pointer-events: none;
    }
    @keyframes float1 { 0%,100% { transform: translate(0,0);} 50% { transform: translate(120px,80px);} }
    @keyframes float2 { 0%,100% { transform: translate(0,0);} 50% { transform: translate(-100px,-60px);} }

    .stApp, .stApp p, .stApp span, .stApp label, .stApp div { color: #e6edf3; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }

    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.75) !important;
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(0, 229, 255, 0.15);
    }
    section[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #00e5ff !important; }

    .hero-wrap {
        position: relative; padding: 42px 36px; border-radius: 24px;
        background: linear-gradient(135deg, rgba(0,229,255,0.10), rgba(255,64,129,0.08));
        border: 1px solid rgba(0, 229, 255, 0.25);
        backdrop-filter: blur(14px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.08);
        margin-bottom: 26px; overflow: hidden;
    }
    .hero-badge {
        display: inline-block; padding: 6px 14px;
        background: rgba(0, 229, 255, 0.15);
        border: 1px solid rgba(0, 229, 255, 0.4);
        color: #00e5ff; border-radius: 999px;
        font-size: 12px; font-weight: 600; letter-spacing: 0.15em;
        text-transform: uppercase; margin-bottom: 18px;
        font-family: 'JetBrains Mono', monospace;
    }
    .hero-title {
        font-size: 54px; font-weight: 700; line-height: 1.05; margin: 0 0 14px 0;
        background: linear-gradient(90deg, #ffffff 0%, #00e5ff 50%, #ff4081 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-sub { font-size: 17px; color: #b8c5d6 !important; max-width: 780px; line-height: 1.6; }
    .hero-stats { display: flex; gap: 32px; margin-top: 26px; flex-wrap: wrap; }
    .stat-box { border-left: 2px solid #00e5ff; padding-left: 14px; }
    .stat-num { font-size: 26px; font-weight: 700; color: #ffffff; font-family: 'JetBrains Mono', monospace; }
    .stat-label { font-size: 11px; color: #8a9bb0; letter-spacing: 0.12em; text-transform: uppercase; }

    .marquee-container {
        overflow: hidden; white-space: nowrap;
        background: linear-gradient(90deg, rgba(0,229,255,0.08), rgba(255,64,129,0.08));
        border-top: 1px solid rgba(0, 229, 255, 0.2);
        border-bottom: 1px solid rgba(0, 229, 255, 0.2);
        padding: 12px 0; margin: 20px 0 30px 0; border-radius: 8px;
    }
    .marquee-track {
        display: inline-block; animation: marquee 38s linear infinite;
        font-family: 'JetBrains Mono', monospace; font-size: 14px;
        color: #00e5ff; letter-spacing: 0.08em;
    }
    .marquee-track span { margin: 0 40px; }
    .marquee-track .pink { color: #ff4081; }
    .marquee-track .green { color: #4CAF50; }
    .marquee-track .amber { color: #FF9800; }
    @keyframes marquee { 0% { transform: translateX(0);} 100% { transform: translateX(-50%);} }

    .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 20px 0 30px 0; }
    .feature-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px; padding: 20px; backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover { transform: translateY(-4px); border-color: rgba(0, 229, 255, 0.5); box-shadow: 0 12px 40px rgba(0, 229, 255, 0.15); }
    .feature-icon { font-size: 24px; margin-bottom: 10px; }
    .feature-title { font-size: 15px; font-weight: 600; color: #ffffff; margin-bottom: 6px; }
    .feature-desc { font-size: 13px; color: #8a9bb0; line-height: 1.5; }

    .result-badge {
        padding: 22px; border-radius: 18px; text-align: center; color: #ffffff;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35); position: relative; overflow: hidden;
    }
    .result-badge::before {
        content: ""; position: absolute; inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.15), transparent);
        pointer-events: none;
    }
    .result-badge h2, .result-badge h4 { color: #ffffff !important; margin: 0; }
    .pulse-dot {
        display: inline-block; width: 10px; height: 10px; background: #ffffff;
        border-radius: 50%; margin-right: 8px; animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100% { opacity:1; transform:scale(1);} 50% { opacity:0.5; transform:scale(1.3);} }

    .care-wrap {
        margin-top: 26px; padding: 26px; border-radius: 20px;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(0,229,255,0.15);
        backdrop-filter: blur(12px);
    }
    .care-header { font-size: 12px; letter-spacing: 0.2em; color: #00e5ff; font-family: 'JetBrains Mono', monospace; margin-bottom: 8px; }
    .care-title { font-size: 22px; font-weight: 700; color: #ffffff !important; margin-bottom: 6px; }
    .care-sub { color: #8a9bb0; font-size: 13px; margin-bottom: 18px; line-height: 1.6; }
    .care-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    @media (max-width: 900px) { .care-cols { grid-template-columns: 1fr; } }
    .care-col { padding: 20px; border-radius: 16px; background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.06); }
    .care-col.do { border-left: 3px solid #4CAF50; }
    .care-col.dont { border-left: 3px solid #F44336; }
    .care-col h4 { font-size: 15px; margin: 0 0 12px 0; display: flex; align-items: center; gap: 8px; }
    .care-col.do h4 { color: #4CAF50 !important; }
    .care-col.dont h4 { color: #ff6b6b !important; }
    .care-list { list-style: none; padding: 0; margin: 0; }
    .care-list li {
        padding: 8px 0 8px 26px; position: relative;
        color: #cdd9e5; font-size: 13.5px; line-height: 1.55;
        border-bottom: 1px dashed rgba(255,255,255,0.05);
    }
    .care-list li:last-child { border-bottom: none; }
    .care-list.do li::before { content: "✓"; position: absolute; left: 4px; top: 8px; color: #4CAF50; font-weight: 700; }
    .care-list.dont li::before { content: "✕"; position: absolute; left: 4px; top: 8px; color: #ff6b6b; font-weight: 700; }
    .urgency-strip { margin-top: 18px; padding: 14px 18px; border-radius: 12px; font-size: 13.5px; line-height: 1.55; display: flex; align-items: flex-start; gap: 12px; }
    .urgency-strip .icon { font-size: 20px; line-height: 1; }

    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(0, 229, 255, 0.4);
        border-radius: 16px; padding: 10px; transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover { border-color: #00e5ff; background: rgba(0, 229, 255, 0.05); }

    button[data-baseweb="tab"] { color: #b8c5d6 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #00e5ff !important; }

    .section-header {
        display: flex; align-items: center; gap: 10px;
        margin: 18px 0 14px 0; padding-bottom: 8px;
        border-bottom: 1px solid rgba(0, 229, 255, 0.2);
    }
    .section-header .dot { width: 8px; height: 8px; background: #00e5ff; border-radius: 50%; box-shadow: 0 0 12px #00e5ff; }
    .section-header h3 { margin: 0 !important; font-size: 20px !important; }

    .footer {
        margin-top: 60px; padding: 24px; text-align: center;
        border-top: 1px solid rgba(0, 229, 255, 0.15);
        color: #8a9bb0; font-size: 13px;
    }
    .footer .accent { color: #00e5ff; font-family: 'JetBrains Mono', monospace; }

    div[data-testid="stAlert"] {
        background: rgba(255, 152, 0, 0.10) !important;
        border-left: 3px solid #FF9800 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stAlert"] * { color: #ffd699 !important; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #00e5ff, #ff4081); border-radius: 6px; }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .main .block-container { position: relative; z-index: 1; }
    </style>
    """,
    unsafe_allow_html=True,
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


CARE_GUIDE = {
    0: {
        "headline": "Preventive Eye Health Plan",
        "summary": "No retinopathy detected — focus on prevention through metabolic control and yearly screening.",
        "urgency": {
            "level": "Routine",
            "color": "#4CAF50",
            "icon": "🟢",
            "text": "Schedule a comprehensive dilated eye exam every 12 months.",
        },
        "do": [
            "Maintain HbA1c below 7% through diet, exercise, and prescribed medication.",
            "Keep blood pressure under 130/80 mmHg and cholesterol (LDL) below 100 mg/dL.",
            "Eat a diet rich in leafy greens, omega-3 fish, berries, nuts, and whole grains.",
            "Exercise at least 150 minutes per week (brisk walking, cycling, yoga).",
            "Get an annual dilated fundus examination even without symptoms.",
            "Stay hydrated (2–3 L water daily) and get 7–8 hours of sleep.",
        ],
        "dont": [
            "Don't skip diabetes medication or insulin doses.",
            "Avoid smoking and secondhand smoke — it accelerates vascular damage.",
            "Limit alcohol (max 1 drink/day women, 2/day men) — it destabilizes glucose.",
            "Cut back on refined sugar, sugary drinks, and processed carbs.",
            "Don't ignore blurry vision, floaters, or dark spots — report immediately.",
            "Avoid extended screen time without breaks (follow the 20-20-20 rule).",
        ],
    },
    1: {
        "headline": "Early-Stage Management",
        "summary": "Microaneurysms detected. Reversible with strict glycemic and blood pressure control.",
        "urgency": {
            "level": "Follow-up in 6–12 months",
            "color": "#2196F3",
            "icon": "🔵",
            "text": "Repeat dilated eye exam every 6–12 months. Consult an ophthalmologist.",
        },
        "do": [
            "Tighten blood glucose control — aim HbA1c ≤ 6.5–7% under doctor supervision.",
            "Monitor blood pressure daily; keep it below 130/80 mmHg.",
            "Take prescribed statins/ACE inhibitors if recommended for kidney/heart protection.",
            "Book a follow-up ophthalmology visit within 6 months.",
            "Adopt a Mediterranean or DASH diet — low glycemic, high antioxidants.",
            "Track sugar levels with a glucometer or CGM and log them.",
        ],
        "dont": [
            "Don't rely only on symptoms — early DR has no visible signs.",
            "Avoid heavy lifting or high-intensity Valsalva-type exercises.",
            "Don't consume high-sodium or fried foods — they worsen vascular stress.",
            "Avoid over-the-counter herbal remedies claiming to cure diabetes.",
            "Don't skip medications during holidays, travel, or fasting.",
            "Avoid rubbing eyes forcefully; use lubricating drops if dry.",
        ],
    },
    2: {
        "headline": "Active Monitoring & Lifestyle Intensification",
        "summary": "Progressing retinopathy. Vision is still preservable with aggressive metabolic control.",
        "urgency": {
            "level": "Follow-up in 3–6 months",
            "color": "#FF9800",
            "icon": "🟡",
            "text": "See a retina specialist within a few weeks. Fluorescein angiography or OCT may be advised.",
        },
        "do": [
            "Schedule a retina specialist consultation within 2–4 weeks.",
            "Undergo OCT (Optical Coherence Tomography) to check for macular edema.",
            "Strictly control HbA1c (< 7%), BP (< 130/80), and lipids.",
            "Report new floaters, blurred vision, or sudden vision changes immediately.",
            "Consider fenofibrate therapy if prescribed — shown to slow DR progression.",
            "Maintain kidney health — get annual urine microalbumin and eGFR tests.",
        ],
        "dont": [
            "Don't delay follow-up appointments — DR can progress silently.",
            "Avoid contact sports, boxing, or head-impact activities.",
            "Don't take aspirin without doctor approval — may increase hemorrhage risk.",
            "Avoid sudden crash diets — they cause glucose swings that worsen retinopathy.",
            "Don't ignore symptoms of low blood sugar (hypoglycemia) or wide sugar swings.",
            "Avoid steam rooms/saunas that cause rapid vasodilation.",
        ],
    },
    3: {
        "headline": "Urgent Ophthalmology Referral",
        "summary": "Severe non-proliferative DR — high risk of vision loss. Requires specialist intervention.",
        "urgency": {
            "level": "URGENT — within 1–2 weeks",
            "color": "#F44336",
            "icon": "🟠",
            "text": "Consult a retina specialist within days. Laser photocoagulation or anti-VEGF injections may be indicated.",
        },
        "do": [
            "See a retina specialist within 1–2 weeks — do not delay.",
            "Undergo fluorescein angiography and OCT for detailed staging.",
            "Ask about anti-VEGF injections (ranibizumab, aflibercept, bevacizumab).",
            "Prepare for possible pan-retinal photocoagulation (PRP) laser therapy.",
            "Maintain extremely tight glycemic control — but avoid sudden HbA1c drops.",
            "Have a family member accompany you to appointments and drive you home.",
        ],
        "dont": [
            "Do NOT skip or postpone the retina specialist appointment.",
            "Avoid strenuous exercise, weightlifting, or activities that raise eye pressure.",
            "Don't take blood thinners or NSAIDs without ophthalmologist clearance.",
            "Avoid air travel to very high altitudes without medical advice.",
            "Don't drive if you notice sudden vision changes — arrange transport.",
            "Avoid bending upside-down (inversion yoga, gardening head-down).",
        ],
    },
    4: {
        "headline": "Emergency Retinal Care Required",
        "summary": "Proliferative DR with neovascularization. Immediate treatment needed to prevent blindness.",
        "urgency": {
            "level": "EMERGENCY — seek care immediately",
            "color": "#9C27B0",
            "icon": "🔴",
            "text": "Go to a retina specialist or eye emergency service today. Vitreous hemorrhage or retinal detachment may occur without treatment.",
        },
        "do": [
            "Seek emergency ophthalmology / retina specialist care TODAY.",
            "Prepare for pan-retinal photocoagulation (PRP) or anti-VEGF injection therapy.",
            "Discuss vitrectomy surgery options if hemorrhage or detachment is present.",
            "Arrange a caregiver — do not travel or manage appointments alone.",
            "Continue strict glycemic, BP, and lipid control alongside treatment.",
            "Get evaluated for kidney disease and cardiovascular risk (often coexists).",
        ],
        "dont": [
            "Do NOT wait for symptoms to worsen — act within 24 hours.",
            "Absolutely avoid lifting weights, straining, or Valsalva maneuvers.",
            "Don't take aspirin, ibuprofen, or blood thinners without doctor approval.",
            "Avoid flying, scuba diving, or high-altitude travel until cleared.",
            "Don't drive — sudden vitreous hemorrhage can cause instant vision loss.",
            "Avoid rubbing or pressing on the eyes at all costs.",
        ],
    },
}


def crop_black_border(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = gray > tol
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        img = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return img


def gaussian_blur_subtraction(img, sigma=10):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def enhance_brightness_contrast(img, alpha=1.2, beta=10):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def preprocess_fundus_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cropped = crop_black_border(img)
    blurred = gaussian_blur_subtraction(cropped)
    enhanced = enhance_brightness_contrast(blurred)
    resized_bgr = cv2.resize(enhanced, EFF_SIZE)
    display_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    input_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return input_rgb, display_rgb


def build_head_model():
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
        full_saved = tf.keras.models.load_model(full_model_path, compile=False)
        try:
            base_backbone = tf.keras.Model(
                inputs=full_saved.input,
                outputs=full_saved.get_layer("avg_pool").output,
            )
        except ValueError:
            gap_layer = [
                l for l in full_saved.layers
                if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)
            ][0]
            base_backbone = tf.keras.Model(inputs=full_saved.input, outputs=gap_layer.output)
    else:
        base_backbone = tf.keras.applications.EfficientNetB4(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg"
        )

    head = build_head_model()
    if os.path.exists(head_model_path):
        head.load_weights(head_model_path)

    return "split", (base_backbone, head)


def predict(processed_img, model_tuple, use_tta=False, tta_steps=8):
    img_array = np.expand_dims(processed_img, axis=0).astype(np.float32)
    input_tensor = tf.keras.applications.efficientnet.preprocess_input(img_array)

    model_type, (backbone, head) = model_tuple
    features = backbone(input_tensor, training=False).numpy()
    base_probs = head(features, training=False).numpy()[0]

    if not use_tta:
        return base_probs

    all_preds = [base_probs]
    for _ in range(tta_steps - 1):
        noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
        all_preds.append(head(features + noise, training=False).numpy()[0])

    return np.mean(all_preds, axis=0)


def render_hero():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-badge">◆ AI-Powered Ophthalmology · v2.0</div>
            <div class="hero-title">RetinaScan AI<br/>Diabetic Retinopathy Detection</div>
            <div class="hero-sub">
                A federated-learning powered deep learning system that classifies retinal fundus
                imagery into 5 clinical severity stages — using an EfficientNetB4 backbone
                fine-tuned across distributed medical datasets with FedProx aggregation.
            </div>
            <div class="hero-stats">
                <div class="stat-box"><div class="stat-num">5</div><div class="stat-label">Severity Stages</div></div>
                <div class="stat-box"><div class="stat-num">EfficientNetB4</div><div class="stat-label">Backbone Model</div></div>
                <div class="stat-box"><div class="stat-num">FedProx</div><div class="stat-label">Training Protocol</div></div>
                <div class="stat-box"><div class="stat-num">TTA×8</div><div class="stat-label">Inference Robustness</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_marquee():
    items = (
        '<span>◉ REAL-TIME FUNDUS ANALYSIS</span>'
        '<span class="pink">◈ FEDERATED LEARNING PIPELINE</span>'
        '<span class="green">✓ 5-CLASS SEVERITY DETECTION</span>'
        '<span class="amber">⚡ TEST-TIME AUGMENTATION</span>'
        '<span>◉ GAUSSIAN VESSEL ENHANCEMENT</span>'
        '<span class="pink">◈ EFFICIENTNET-B4 BACKBONE</span>'
        '<span class="green">✓ CLINICAL-GRADE PREPROCESSING</span>'
        '<span class="amber">⚡ AI-ASSISTED SCREENING</span>'
    )
    st.markdown(
        f'<div class="marquee-container"><div class="marquee-track">{items}{items}</div></div>',
        unsafe_allow_html=True,
    )


def render_features():
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Deep Neural Backbone</div>
                <div class="feature-desc">EfficientNetB4 with 1792-dim feature extraction fine-tuned on curated fundus data.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">Federated Learning</div>
                <div class="feature-desc">FedProx aggregation across distributed clinical datasets — privacy preserving.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔬</div>
                <div class="feature-title">Vessel Enhancement</div>
                <div class="feature-desc">Gaussian blur subtraction accentuates microaneurysms & hemorrhages.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">TTA Ensemble</div>
                <div class="feature-desc">Test-Time Augmentation averages multiple perturbed passes for robust predictions.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pil_to_reportlab(pil_img, max_width_mm=80):
    """Converts a PIL image (or numpy array) to a ReportLab Image at fixed width."""
    if isinstance(pil_img, np.ndarray):
        pil_img = PIL.Image.fromarray(pil_img)
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    w, h = pil_img.size
    target_w = max_width_mm * mm
    target_h = target_w * (h / w)
    return RLImage(buf, width=target_w, height=target_h)


def build_care_plan_pdf(pred_class, confidence, probs, original_img, enhanced_img):
    """Builds a downloadable PDF containing the diagnosis, retina images, and care plan."""
    guide = CARE_GUIDE[pred_class]
    urgency = guide["urgency"]
    stage_color_hex = CLASS_COLORS[pred_class]
    stage_color = colors.HexColor(stage_color_hex)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="RetinaScan AI — Care Plan Report", author="RetinaScan AI",
    )

    styles = getSampleStyleSheet()
    style_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=22, leading=26,
        textColor=colors.HexColor("#0f3460"), spaceAfter=6, alignment=TA_LEFT)
    style_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, leading=18,
        textColor=colors.HexColor("#0f3460"), spaceBefore=10, spaceAfter=6)
    style_meta = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#666666"))
    style_body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15,
        textColor=colors.HexColor("#222222"), spaceAfter=4)
    style_bullet_do = ParagraphStyle("BulletDo", parent=styles["Normal"], fontSize=10, leading=14,
        textColor=colors.HexColor("#1b5e20"), leftIndent=12, spaceAfter=3)
    style_bullet_dont = ParagraphStyle("BulletDont", parent=styles["Normal"], fontSize=10, leading=14,
        textColor=colors.HexColor("#b71c1c"), leftIndent=12, spaceAfter=3)
    style_caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=6)
    style_disclaimer = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8.5, leading=12,
        textColor=colors.HexColor("#8a4b00"), spaceBefore=8)

    story = []
    story.append(Paragraph("RetinaScan AI — Diabetic Retinopathy Care Report", style_h1))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y · %I:%M %p')}  |  "
        "Model: EfficientNetB4 + FedProx  |  For educational use only",
        style_meta))
    story.append(Spacer(1, 10))

    diag_data = [[Paragraph(
        f"<b>Diagnostic Output</b><br/>"
        f"<font size=16 color='{stage_color_hex}'><b>Stage {pred_class} — {CLASS_NAMES[pred_class]}</b></font><br/>"
        f"<font size=10>Model Confidence: <b>{confidence:.2f}%</b></font><br/>"
        f"<font size=9 color='#555555'>{CLASS_DESCRIPTIONS[pred_class]}</font>",
        style_body)]]
    diag_table = Table(diag_data, colWidths=[174 * mm])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f9ff")),
        ("BOX", (0, 0), (-1, -1), 1.2, stage_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Retinal Fundus Analysis", style_h2))
    story.append(Paragraph(
        "The AI system analyzes a preprocessed retinal fundus image. "
        "Below is the original image submitted along with the enhanced version used by the model. "
        "The enhanced view applies Gaussian blur subtraction, which accentuates blood vessels, "
        "microaneurysms, and hemorrhages — key markers used to determine DR severity.",
        style_body))
    story.append(Spacer(1, 6))

    original_rl = _pil_to_reportlab(original_img, max_width_mm=80)
    enhanced_rl = _pil_to_reportlab(enhanced_img, max_width_mm=80)

    img_table = Table(
        [
            [original_rl, enhanced_rl],
            [
                Paragraph("<b>Original Fundus Image</b><br/><font size=8>As uploaded — unmodified retina photograph.</font>", style_caption),
                Paragraph("<b>Enhanced (Model Input)</b><br/><font size=8>Gaussian subtraction highlights vessels & lesions.</font>", style_caption),
            ],
        ],
        colWidths=[85 * mm, 85 * mm],
    )
    img_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("What the AI Is Looking At", style_h2))
    explanation_bullets = [
        "<b>Optic Disc:</b> The bright circular region where the optic nerve exits the retina. Neovascularization here indicates proliferative DR.",
        "<b>Macula &amp; Fovea:</b> The central dark spot responsible for sharp vision. Fluid or hemorrhage here causes vision loss (macular edema).",
        "<b>Blood Vessels:</b> Branching vessels radiating from the optic disc. In DR, they become dilated, tortuous, or leaky.",
        "<b>Microaneurysms:</b> Tiny red dots — the earliest sign of DR. Enhanced view makes them clearly visible.",
        "<b>Hemorrhages &amp; Exudates:</b> Dark red blots (bleeds) or bright yellow patches (lipid leaks) indicate moderate-to-severe DR.",
    ]
    for b in explanation_bullets:
        story.append(Paragraph(f"• {b}", style_body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Probability Distribution", style_h2))
    prob_rows = [["Stage", "Class", "Probability"]]
    for k, name in CLASS_NAMES.items():
        prob_rows.append([f"Stage {k}", name, f"{probs[k] * 100:.2f}%"])
    prob_table = Table(prob_rows, colWidths=[30 * mm, 90 * mm, 40 * mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, pred_class + 1), (-1, pred_class + 1), stage_color),
        ("TEXTCOLOR", (0, pred_class + 1), (-1, pred_class + 1), colors.white),
        ("FONTNAME", (0, pred_class + 1), (-1, pred_class + 1), "Helvetica-Bold"),
    ]))
    story.append(prob_table)

    story.append(PageBreak())
    story.append(Paragraph(f"Personalized Care Plan — {guide['headline']}", style_h1))
    story.append(Paragraph(guide["summary"], style_body))
    story.append(Spacer(1, 6))

    urgency_hex = urgency["color"]
    urgency_table = Table(
        [[Paragraph(
            f"<b>{urgency['level']}</b><br/><font size=9>{urgency['text']}</font>",
            ParagraphStyle("U", parent=styles["Normal"], fontSize=11, textColor=colors.white, leading=15),
        )]],
        colWidths=[174 * mm],
    )
    urgency_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(urgency_hex)),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(urgency_table)
    story.append(Spacer(1, 12))

    do_paras = [Paragraph(f"✓ {item}", style_bullet_do) for item in guide["do"]]
    dont_paras = [Paragraph(f"✕ {item}", style_bullet_dont) for item in guide["dont"]]
    dodont_header = [
        Paragraph("<b><font color='#1b5e20'>Recommended Actions (Do's)</font></b>", style_body),
        Paragraph("<b><font color='#b71c1c'>Things to Avoid (Don'ts)</font></b>", style_body),
    ]
    dodont_table = Table([dodont_header, [do_paras, dont_paras]], colWidths=[85 * mm, 85 * mm])
    dodont_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, 1), colors.HexColor("#e8f5e9")),
        ("BACKGROUND", (1, 0), (1, 1), colors.HexColor("#ffebee")),
        ("BOX", (0, 0), (0, 1), 0.6, colors.HexColor("#4CAF50")),
        ("BOX", (1, 0), (1, 1), 0.6, colors.HexColor("#F44336")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(dodont_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Sources & References", style_h2))
    story.append(Paragraph(
        "Care guidance summarized from: American Academy of Ophthalmology (AAO), "
        "American Diabetes Association (ADA) Standards of Care, National Eye Institute (NIH/NEI), "
        "World Health Organization (WHO), and Mayo Clinic Patient Care Guidelines.",
        style_body))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Medical Disclaimer:</b> This report is generated by an AI diagnostic assistance prototype "
        "and is not a substitute for professional medical evaluation. Care recommendations are "
        "educational summaries derived from general clinical guidance. Always consult a licensed "
        "ophthalmologist and endocrinologist for personalized medical advice and treatment decisions.",
        style_disclaimer))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def render_care_recommendations(pred_class: int):
    guide = CARE_GUIDE[pred_class]
    urgency = guide["urgency"]
    do_items = "".join(f"<li>{item}</li>" for item in guide["do"])
    dont_items = "".join(f"<li>{item}</li>" for item in guide["dont"])

    html_content = f"""<div class="care-wrap">
<div class="care-header">◆ PERSONALIZED CARE PLAN · STAGE {pred_class}</div>
<div class="care-title">{guide["headline"]}</div>
<div class="care-sub">{guide["summary"]}</div>
<div class="urgency-strip" style="background: {urgency['color']}22; border: 1px solid {urgency['color']}66;">
<div class="icon">{urgency['icon']}</div>
<div>
<div style="color:{urgency['color']};font-weight:700;font-size:13px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">
{urgency['level']}
</div>
<div style="color:#e6edf3;">{urgency['text']}</div>
</div>
</div>
<div class="care-cols" style="margin-top:18px;">
<div class="care-col do">
<h4>✅ Recommended Actions (Do's)</h4>
<ul class="care-list do">{do_items}</ul>
</div>
<div class="care-col dont">
<h4>⛔ Things to Avoid (Don'ts)</h4>
<ul class="care-list dont">{dont_items}</ul>
</div>
</div>
<div style="margin-top:16px;font-size:11.5px;color:#6b7a8f;letter-spacing:0.05em;line-height:1.5;">
Guidance summarized from American Academy of Ophthalmology (AAO), American Diabetes Association (ADA),
National Eye Institute (NIH/NEI), WHO, and Mayo Clinic. For educational use only —
always follow your physician's personalized advice.
</div>
</div>"""

    st.markdown(html_content, unsafe_allow_html=True)


def main():
    st.sidebar.markdown("### 🩺 DR Screening Panel")
    st.sidebar.markdown(
        "<div style='color:#8a9bb0;font-size:13px;line-height:1.5;'>Upload a fundus photograph of the retina to run automated AI screening for Diabetic Retinopathy.</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("### ⚙️ Inference Settings")
    use_tta = st.sidebar.checkbox(
        "Enable TTA (Test-Time Augmentation)",
        value=True,
        help="Averages multiple perturbed passes for enhanced accuracy.",
    )
    tta_steps = (
        st.sidebar.slider("TTA Steps", min_value=2, max_value=16, value=8)
        if use_tta else 8
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧬 Model Architecture")
    st.sidebar.markdown(
        "<div style='color:#8a9bb0;font-size:13px;line-height:1.7;'>"
        "▸ <b style='color:#00e5ff'>Backbone:</b> Fine-Tuned EfficientNetB4<br/>"
        "▸ <b style='color:#00e5ff'>Training:</b> Federated Learning (FedProx)<br/>"
        "▸ <b style='color:#00e5ff'>Preprocessing:</b> Gaussian Blur Subtraction<br/>"
        "▸ <b style='color:#00e5ff'>Input:</b> 224×224 BGR resized"
        "</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='color:#8a9bb0;font-size:11px;text-align:center;letter-spacing:0.1em;'>RETINASCAN · AI PROTOTYPE</div>",
        unsafe_allow_html=True,
    )

    render_hero()
    render_marquee()
    render_features()

    st.markdown(
        '<div class="section-header"><span class="dot"></span><h3>📤 Upload Retinal Fundus Image</h3></div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drag & drop or browse — supported formats: PNG, JPG, JPEG",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        try:
            image = PIL.Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        col1, col2 = st.columns([1, 1])

        with st.spinner("🔬 Applying visual enhancement & preprocessing..."):
            processed_img, enhanced_display = preprocess_fundus_image(image)

        with col1:
            st.markdown(
                '<div class="section-header"><span class="dot"></span><h3>🖼️ Image Analysis</h3></div>',
                unsafe_allow_html=True,
            )
            img_tabs = st.tabs(["🔬 Preprocessed (Model Input)", "📷 Original Raw"])
            with img_tabs[0]:
                st.image(enhanced_display, caption="Enhanced Retinal View — Gaussian Subtraction Applied", use_column_width=True)
            with img_tabs[1]:
                st.image(image, caption="Original Uploaded Image", use_column_width=True)

        with st.spinner("🧠 Loading neural backbone & analyzing fundus image..."):
            model_tuple = load_dr_model()
            probs = predict(processed_img, model_tuple, use_tta=use_tta, tta_steps=tta_steps)
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class]) * 100

        with col2:
            st.markdown(
                '<div class="section-header"><span class="dot"></span><h3>🎯 Diagnostic Results</h3></div>',
                unsafe_allow_html=True,
            )

            pred_label = CLASS_NAMES[pred_class]
            color = CLASS_COLORS[pred_class]

            st.markdown(
                f"""
                <div class="result-badge" style="background: linear-gradient(135deg, {color}, {color}cc);">
                    <div style="font-size:11px;letter-spacing:0.2em;opacity:0.85;margin-bottom:6px;">
                        <span class="pulse-dot"></span>DIAGNOSTIC OUTPUT
                    </div>
                    <h2>Stage {pred_class} — {pred_label}</h2>
                    <h4 style="opacity:0.92;margin-top:6px;">Confidence · {confidence:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='margin-top:14px;padding:14px;background:rgba(255,255,255,0.04);border-radius:12px;border-left:3px solid {color};'>"
                f"<div style='font-size:11px;color:#8a9bb0;letter-spacing:0.15em;margin-bottom:6px;'>CLINICAL DESCRIPTION</div>"
                f"<div style='color:#e6edf3;font-size:14px;line-height:1.6;'>{CLASS_DESCRIPTIONS[pred_class]}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            df_probs = pd.DataFrame({
                "Stage": [f"Stage {k}: {v}" for k, v in CLASS_NAMES.items()],
                "Probability (%)": probs * 100,
            })

            fig = px.bar(
                df_probs, x="Probability (%)", y="Stage", orientation="h",
                text="Probability (%)", color="Stage",
                color_discrete_map={f"Stage {k}: {v}": CLASS_COLORS[k] for k, v in CLASS_NAMES.items()},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
            fig.update_layout(
                showlegend=False,
                xaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)", color="#b8c5d6"),
                yaxis=dict(color="#b8c5d6"),
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk", color="#e6edf3"),
                title=dict(text="Probability Distribution Across DR Stages",
                           font=dict(size=14, color="#ffffff"), x=0.02),
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="section-header"><span class="dot"></span><h3>💡 Personalized Care Recommendations</h3></div>',
            unsafe_allow_html=True,
        )
        render_care_recommendations(pred_class)

        with st.spinner("📄 Building your downloadable care report..."):
            pdf_bytes = build_care_plan_pdf(
                pred_class=pred_class,
                confidence=confidence,
                probs=probs,
                original_img=image,
                enhanced_img=enhanced_display,
            )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.markdown(
            "<div style='margin-top:18px;padding:18px;border-radius:14px;"
            "background:linear-gradient(135deg, rgba(0,229,255,0.10), rgba(255,64,129,0.08));"
            "border:1px solid rgba(0,229,255,0.3);'>"
            "<div style='font-size:12px;letter-spacing:0.18em;color:#00e5ff;font-family:JetBrains Mono,monospace;margin-bottom:6px;'>◆ EXPORT REPORT</div>"
            "<div style='color:#ffffff;font-size:16px;font-weight:600;margin-bottom:4px;'>Download your full care report (PDF)</div>"
            "<div style='color:#8a9bb0;font-size:13px;line-height:1.5;'>Includes the diagnosis, retina images with anatomical annotations, probability breakdown, and your personalized Do's &amp; Don'ts care plan.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️  Download PDF Care Report",
            data=pdf_bytes,
            file_name=f"RetinaScan_CarePlan_Stage{pred_class}_{ts}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.warning(
            "**Medical Disclaimer:** This tool is an AI diagnostic assistance prototype and should not be used as a sole basis for clinical diagnosis. Care recommendations are educational summaries — always consult a licensed ophthalmologist and endocrinologist for personalized medical advice."
        )
    else:
        st.markdown(
            """
            <div style="padding:40px;text-align:center;background:rgba(255,255,255,0.03);border-radius:16px;border:1px dashed rgba(0,229,255,0.25);margin-top:10px;">
                <div style="font-size:48px;margin-bottom:12px;">👁️</div>
                <div style="color:#b8c5d6;font-size:16px;font-weight:500;">Awaiting Fundus Image Upload</div>
                <div style="color:#8a9bb0;font-size:13px;margin-top:6px;">Upload a retinal image above to begin AI-powered screening and get personalized care guidance.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            <div>Built with <span class="accent">Streamlit</span> · <span class="accent">TensorFlow</span> · <span class="accent">EfficientNetB4</span> · <span class="accent">FedProx</span></div>
            <div style="margin-top:6px;opacity:0.7;">© RetinaScan AI — Research prototype developed by Gajendra Sahani.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
