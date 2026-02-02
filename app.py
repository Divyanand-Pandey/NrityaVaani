import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ---------- CONFIG ----------
MODEL_PATH = "nrityavaani_mobilenet.pth"
DATA_DIR = "final_dataset/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

st.set_page_config(
    page_title="NrityaVaani – Mudra Recognition",
    layout="centered"
)

st.title("NrityaVaani – Live Mudra Recognition")
st.write("Upload an image or use camera input to recognize Bharatanatyam mudras.")

# Load class names
classes = sorted(os.listdir(DATA_DIR))

# Load model (cached)
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

# Transform (optimized for MobileNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    
])

# ---------------- UI ----------------
option = st.radio(
    "Choose input method:",
    ("🖼 Upload Image", "📷 Use Camera")
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
    st.image(img, caption="Input Image", width=300)

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        idx = prob.argmax().item()
        confidence = prob[0][idx].item()

    st.markdown("### 🔍 Prediction")
    st.success(f"**Mudra:** {classes[idx]}")
    st.info(f"**Confidence:** {confidence*100:.2f}%")

# ---------------- MUDRA INFO ----------------
MUDRA_INFO = {
    "Alapadma": "Fully bloomed lotus; beauty, purity, fullness.",
    "Araala": "Bent finger; drinking poison, wind, unsteadiness.",
    "Ardhachandra": "Half moon; moon, female, spear.",
    "Ardhapataka": "Half flag; leaves, knives, separation.",
    "Chandrakala": "Crescent moon; moon, ornament, tenderness.",
    "Kartarimukha": "Scissors face; separation, opposition, lightning.",
    "Mayura": "Peacock; purity, bird, applying kajal.",
    "Mrigashirsha": "Deer head; woman's cheek, searching, calling.",
    "Mushti": "Fist; strength, holding, anger.",
    "Padmakosha": "Lotus bud; fruits, balls, breast.",
    "Pataka": "Flag; cloud, forest, denial, wind.",
    "Sarpashirsha": "Serpent head; snake, water offering.",
    "Shikhara": "Peak; bow, determination, valor.",
    "Simhamukha": "Lion face; courage, strength, ferocity.",
    "Suchi": "Needle; indication, number one, pointing.",
    "Tripataka": "Three-part flag; crown, tree, arrow.",
    "Trishula": "Trident; trinity, weapon of Shiva."
}


st.markdown("## 🧠 Bharatanatyam Mudras")
st.markdown("Below are the mudras recognized by **NrityaVaani**.")

cols = st.columns(3)

for i, (mudra, meaning) in enumerate(MUDRA_INFO.items()):
    with cols[i % 3]:
        st.image(f"assets/mudras/{mudra}.jpg", width=220)
        st.markdown(f"### {mudra}")
        st.caption(meaning)
