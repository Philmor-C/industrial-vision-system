import streamlit as st
import glob
from PIL import Image
import matplotlib.pyplot as plt

from core.pipeline import run_pipeline
from simulation.conveyor import ConveyorItem, encoder_move, check_trigger
from core.yolo import load_yolo
from core.patchcore import load_patchcore
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

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

if "items" not in st.session_state:
    st.session_state.items = [
        ConveyorItem(i, Image.open(img).convert("RGB"))
        for i, img in enumerate(images)
    ]

# -------------------
# SIDEBAR
# -------------------
speed = st.sidebar.slider("Speed", 1, 20, 5)
zone = st.sidebar.slider("Zone", 100, 400, 250)

# -------------------
# KPI (SAFE)
# -------------------
total = len(st.session_state.items)

processed = sum(
    1 for i in st.session_state.items
    if isinstance(i.result, dict)
)

unknown = sum(
    len(i.result.get("unknown", []))
    for i in st.session_state.items
    if isinstance(i.result, dict)
)

passed = sum(
    1 for i in st.session_state.items
    if isinstance(i.result, dict) and len(i.result.get("unknown", [])) == 0
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", total)
col2.metric("Unknown", unknown)
col3.metric("Passed", passed)
col4.metric("Processed", processed)

st.divider()

# -------------------
# CONVEYOR
# -------------------
left, center, right = st.columns([1,3,1])

with center:

    fig, ax = plt.subplots(figsize=(10,3))

    ax.set_xlim(0,600)
    ax.set_ylim(0,100)

    ax.hlines(50,0,600,linewidth=6)
    ax.axvline(zone,color="red",linestyle="--")

    encoder_move(st.session_state.items, speed)

    for item in st.session_state.items:

        if check_trigger(item, zone) and item.result is None:
            item.result = run_pipeline(
                item.image,
                yolo_model,
                memory_bank,
                threshold
            )

        color = "blue"

        if isinstance(item.result, dict):
            if len(item.result.get("unknown", [])) > 0:
                color = "red"
            else:
                color = "green"

        ax.scatter(item.x,50,s=150,color=color)

    st.pyplot(fig)

# -------------------
# LOG
# -------------------
st.divider()
st.subheader("Inspection Log")

log = []

for i in st.session_state.items:
    if isinstance(i.result, dict):
        log.append({
            "ID": i.idx,
            "YOLO": len(i.result["yolo"]),
            "Anomaly": len(i.result["anomaly"]),
            "Unknown": len(i.result["unknown"])
        })

st.dataframe(log)
