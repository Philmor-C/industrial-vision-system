import streamlit as st
import glob
from PIL import Image
import matplotlib.pyplot as plt

from core.pipeline import run_pipeline
from simulation.conveyor import ConveyorItem, encoder_move, check_trigger
from core.yolo import load_yolo
from core.patchcore import load_patchcore

st.set_page_config(layout="wide")
st.title("🏭 Industrial Vision System")

# -------------------
# LOAD MODELS
# -------------------
@st.cache_resource
def load_models():
    yolo = load_yolo()
    memory_bank, threshold = load_patchcore()
    return yolo, memory_bank, threshold

yolo_model, memory_bank, threshold = load_models()

# -------------------
# LOAD IMAGES
# -------------------
images = glob.glob("data/test_images/*.jpg")

# -------------------
# SAFE SESSION STATE (FIXED)
# -------------------
if "conveyor_items" not in st.session_state:

    st.session_state.conveyor_items = [
        ConveyorItem(i, Image.open(img).convert("RGB"))
        for i, img in enumerate(images)
    ]

# -------------------
# SIDEBAR
# -------------------
speed = st.sidebar.slider("Speed", 1, 20, 5)
zone = st.sidebar.slider("Zone", 100, 400, 250)

# -------------------
# KPI (FIXED SAFE VERSION)
# -------------------
items = st.session_state.conveyor_items   # 🔥 FIX: no more .items()

total = len(items)

processed = sum(1 for i in items if i.result)

unknown = sum(
    len(i.result["unknown"])
    for i in items
    if i.result
)

passed = sum(
    1 for i in items
    if i.result and len(i.result["unknown"]) == 0
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", total)
col2.metric("Unknown", unknown)
col3.metric("Passed", passed)
col4.metric("Processed", processed)

st.divider()

# -------------------
# CONVEYOR VISUAL
# -------------------
left, center, right = st.columns([1, 3, 1])

with center:

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
                None,
                memory_bank
            )

        color = "blue"

        if item.result:
            if len(item.result["unknown"]) > 0:
                color = "red"
            else:
                color = "green"

        ax.scatter(item.x, 50, s=150, color=color)

    st.pyplot(fig)

# -------------------
# LOG TABLE
# -------------------
st.divider()
st.subheader("Inspection Log")

log = []

for i in items:
    if i.result:
        log.append({
            "ID": i.idx,
            "YOLO": len(i.result["yolo"]),
            "Anomaly": len(i.result["anomaly"]),
            "Unknown": len(i.result["unknown"])
        })

st.dataframe(log)
