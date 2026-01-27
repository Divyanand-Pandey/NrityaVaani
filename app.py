import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
MODEL_PATH = "nrityavaani_mobilenet.pth"
DATA_DIR = "final_dataset/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.60
# ----------------------------

st.set_page_config(
    page_title="NrityaVaani – Mudra Recognition",
    layout="centered"
)

st.title("NrityaVaani – Live Mudra Recognition")
st.write("Upload an image or use camera input to recognize Bharatanatyam mudras.")

# ---------- LOAD CLASSES ----------
try:
    classes = sorted(os.listdir(DATA_DIR))
except Exception as e:
    st.error("Dataset folder not found.")
    st.stop()

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- UI ----------------
option = st.radio(
    "Choose input method:",
    ("📷 Use Camera", "🖼 Upload Image")
)

image = None

if option == "📷 Use Camera":
    image = st.camera_input("Capture a mudra")

elif option == "🖼 Upload Image":
    image = st.file_uploader(
        "Upload a mudra image",
        type=["jpg", "jpeg", "png"]
    )

# ---------------- PREDICTION ----------------
if image is not None:
    img = Image.open(image).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]

    # Top-3 predictions
    top_probs, top_idxs = torch.topk(prob, 3)

    st.markdown("### 🔍 Prediction")

    main_idx = top_idxs[0].item()
    main_conf = top_probs[0].item()

    if main_conf < CONFIDENCE_THRESHOLD:
        st.warning("⚠ Low confidence prediction. Mudra may be unclear.")
    else:
        st.success(f"**Mudra:** {classes[main_idx]}")

    st.info(f"**Confidence:** {main_conf*100:.2f}%")

    # ---------- BAR CHART ----------
    fig, ax = plt.subplots()
    labels = [classes[i] for i in top_idxs]
    values = [p.item() * 100 for p in top_probs]

    ax.barh(labels, values)
    ax.set_xlabel("Confidence (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

# ---------- MUDRA INFO ----------
MUDRA_INFO = {
    "Alapadma": "Fully bloomed lotus; beauty and fullness.",
    "Ardhapataka": "Half flag; leaves, knives, or separation.",
    "Chandrakala": "Crescent moon; moon, ornament, tenderness.",
    "Kartarimukha": "Scissors face; separation, opposition.",
    "Mayura": "Peacock; purity, bird, delicate actions.",
    "Pataka": "Flag; cloud, forest, denial.",
    "Shikhara": "Peak; bow, determination, valor.",
    "Simhamukha": "Lion face; courage, strength.",
    "Suchi": "Needle; indication, number one.",
    "Tripataka": "Three-part flag; crown, tree, arrow."
}

st.markdown("## 🧠 Bharatanatyam Mudras")
st.markdown("Below are the mudras recognized by **NrityaVaani**.")

cols = st.columns(3)

for i, (mudra, meaning) in enumerate(MUDRA_INFO.items()):
    with cols[i % 3]:
        try:
            st.image(f"assets/mudras/{mudra}.jpg", use_column_width=True)
        except:
            st.write("🖐 Image not available")
        st.markdown(f"### {mudra}")
        st.caption(meaning)
        st.write("---")
        st.markdown("---")
        st.markdown("Developed by **NrityaVaani Team**. © 2024")
