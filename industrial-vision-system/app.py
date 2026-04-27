import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from core.pipeline import run_pipeline
from simulation.conveyor import ConveyorItem, encoder_move, check_trigger
from core.yolo import load_yolo
from core.patchcore import load_patchcore

st.set_page_config(layout="wide")
st.title("🏭 Industrial Vision System")

# -------------------
# SAFE LOAD MODELS
# -------------------
@st.cache_resource
def load_models():
    yolo = load_yolo()
    memory_bank, threshold = load_patchcore()
    backbone = None
    return yolo, backbone, memory_bank, threshold

yolo_model, backbone, memory_bank, threshold = load_models()

# -------------------
# IMAGES
# -------------------
BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "data" / "test_images"

images = sorted(list(IMG_DIR.glob("*.jpg")))

if len(images) == 0:
    st.error(f"No images found in: {IMG_DIR}")
    st.stop()

# -------------------
# INIT STATE (FIXED NAME)
# -------------------
if "conveyor_items" not in st.session_state:
    st.session_state.conveyor_items = [
        ConveyorItem(i, Image.open(img).convert("RGB"))
        for i, img in enumerate(images)
    ]

# -------------------
# CONTROLS
# -------------------
speed = st.sidebar.slider("Speed", 1, 20, 5)
zone = st.sidebar.slider("Zone", 100, 400, 250)

# -------------------
# KPI SAFE FIX (CRITICAL FIXED)
# -------------------
items = st.session_state.conveyor_items

total = len(items)

processed = sum(
    1 for i in items if getattr(i, "result", None)
)

unknown = sum(
    len(i.result["unknown"])
    for i in items
    if getattr(i, "result", None)
)

passed = sum(
    1 for i in items
    if getattr(i, "result", None) and len(i.result["unknown"]) == 0
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", total)
col2.metric("Unknown", unknown)
col3.metric("Passed", passed)
col4.metric("Processed", processed)

st.divider()

# -------------------
# CONVEYOR LOOP
# -------------------
fig, ax = plt.subplots(figsize=(10, 3))

ax.set_xlim(0, 600)
ax.set_ylim(0, 100)

ax.hlines(50, 0, 600, linewidth=6)
ax.axvline(zone, color="red", linestyle="--")

encoder_move(items, speed)

for item in items:

    if check_trigger(item, zone) and item.result is None:

        item.result = run_pipeline(
            item.image,
            yolo_model,
            backbone,
            memory_bank
        )

    color = "blue"

    if item.result:
        color = "red" if len(item.result["unknown"]) > 0 else "green"

    ax.scatter(item.x, 50, s=150, color=color)

st.pyplot(fig)

# -------------------
# LOG
# -------------------
st.divider()
st.subheader("Inspection Log")

log = []

for i in items:
    if getattr(i, "result", None):

        log.append({
            "ID": i.idx,
            "YOLO": len(i.result["yolo"]),
            "Anomaly": len(i.result["anomaly"]),
            "Unknown": len(i.result["unknown"]),
            "Score": i.result.get("score", 0)
        })

st.dataframe(log)
