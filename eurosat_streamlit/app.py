import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EuroSAT Land Classifier",
    page_icon="🛰️",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]
CLASS_EMOJIS = ["🌾", "🌲", "🌿", "🛣️", "🏭", "🐄", "🍇", "🏘️", "🏞️", "🌊"]
CLASS_DESCS = [
    "Seasonal agricultural fields", "Dense tree cover",
    "Grasslands and shrubs", "Roads and highways",
    "Factories and warehouses", "Open grazing land",
    "Orchards and vineyards", "Urban housing areas",
    "Water bodies – rivers", "Sea and lake water bodies",
]

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Model (cached so it loads only once) ──────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 10),
    )
    model_path = os.path.join(os.path.dirname(__file__), "models", "resnet18_optical_best.pth")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0).numpy()
    idx = int(np.argmax(probs))
    return idx, probs

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛰️ EuroSAT Land Classifier")
st.caption("ResNet-18 · Optical RGB · 91.00% accuracy · B.Tech AIML CSET301 2025")

st.markdown("---")

# Load model (spinner shown only first time)
with st.spinner("Loading ResNet-18 model..."):
    model = load_model()

st.success("✅ Model loaded — ResNet-18 Optical (11.2M parameters)", icon="🧠")

st.markdown("---")

# Upload
uploaded = st.file_uploader(
    "Upload a satellite image patch",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    help="64×64 EuroSAT-style Sentinel-2 RGB patch recommended",
)

if uploaded:
    col1, col2 = st.columns([1, 2])

    pil_img = Image.open(uploaded)

    with col1:
        st.image(pil_img, caption=f"{uploaded.name}", use_container_width=True)
        st.caption(f"Size: {pil_img.size[0]}×{pil_img.size[1]}px · {uploaded.size/1024:.1f} KB")

    with col2:
        with st.spinner("Running inference..."):
            idx, probs = predict(model, pil_img)

        pred_class = CLASS_NAMES[idx]
        pred_emoji = CLASS_EMOJIS[idx]
        pred_desc  = CLASS_DESCS[idx]
        confidence = float(probs[idx]) * 100

        st.markdown(f"### {pred_emoji} {pred_class}")
        st.caption(pred_desc)

        # Confidence metric
        st.metric("Confidence", f"{confidence:.2f}%")

        # Colour the bar green if confident, orange if not
        bar_color = "green" if confidence >= 70 else "orange"
        st.progress(float(probs[idx]))

    st.markdown("---")
    st.subheader("All class probabilities")

    # Sort by probability descending
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        cols = st.columns([3, 6, 1])
        label = f"{CLASS_EMOJIS[i]} {CLASS_NAMES[i]}"
        pct   = float(probs[i]) * 100
        cols[0].write(f"**{label}**" if i == idx else label)
        cols[1].progress(float(probs[i]))
        cols[2].write(f"**{pct:.1f}%**" if i == idx else f"{pct:.1f}%")

    st.markdown("---")
    st.markdown(
        f"**Model:** ResNet-18 (Optical only) &nbsp;|&nbsp; "
        f"**Input:** Sentinel-2 Bands B4, B3, B2 (RGB) &nbsp;|&nbsp; "
        f"**Test accuracy:** 91.00% &nbsp;|&nbsp; "
        f"**F1-score:** 91.17%"
    )

else:
    # Show class reference when no image uploaded
    st.info("👆 Upload a satellite image above to classify it", icon="🛰️")
    st.markdown("#### Supported land-use classes")
    cols = st.columns(5)
    for i, (name, emoji, desc) in enumerate(zip(CLASS_NAMES, CLASS_EMOJIS, CLASS_DESCS)):
        with cols[i % 5]:
            st.markdown(f"**{emoji} {name}**")
            st.caption(desc)
