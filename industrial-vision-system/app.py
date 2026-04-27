import streamlit as st
import glob
import matplotlib.pyplot as plt
from PIL import Image

from core.pipeline import run_pipeline
from simulation.conveyor import ConveyorItem, encoder_move, check_trigger

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🏭 Industrial Vision System Dashboard")

# =========================
# LOAD DATA (SAFE - NO cv2)
# =========================
images = glob.glob("data/test_images/*.jpg")

# =========================
# INIT STATE
# =========================
if "items" not in st.session_state:
    st.session_state.items = [
        ConveyorItem(i, Image.open(img).convert("RGB"))
        for i, img in enumerate(images)
    ]

# =========================
# SIDEBAR CONTROL PANEL
# =========================
st.sidebar.header("⚙️ Control Panel")

mode = st.sidebar.selectbox(
    "Operation Mode",
    ["Inspection", "Batch", "Conveyor Live"]
)

speed = st.sidebar.slider("Conveyor Speed", 1, 20, 5)
zone = st.sidebar.slider("Inspection Zone", 100, 400, 250)
threshold_ui = st.sidebar.slider("Anomaly Sensitivity", 0.3, 0.9, 0.6)

# =========================
# KPI CALCULATION (SAFE)
# =========================
total = len(st.session_state.items)

processed = sum(
    1 for i in st.session_state.items if i.result is not None
)

unknown = sum(
    len(i.result["unknown"])
    for i in st.session_state.items
    if i.result
)

passed = sum(
    1 for i in st.session_state.items
    if i.result and len(i.result["unknown"]) == 0
)

# =========================
# KPI BAR
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📦 Total Items", total)
col2.metric("⚠️ Unknown Defects", unknown)
col3.metric("✅ Passed Items", passed)
col4.metric("⚙️ Processed", processed)

st.divider()

# =========================
# MAIN LAYOUT
# =========================
left, center, right = st.columns([1, 3, 1])

# =========================
# LEFT PANEL
# =========================
with left:
    st.subheader("📊 System Status")

    st.write("Mode:", mode)
    st.write("Speed:", speed)
    st.write("Zone:", zone)

    st.write("---")
    st.subheader("📡 Live Signals")

    st.success("Encoder: ACTIVE")
    st.success("YOLO: RUNNING")
    st.success("PatchCore: RUNNING")

# =========================
# CENTER PANEL (CONVEYOR)
# =========================
with center:

    st.subheader("🚂 Conveyor Line")

    placeholder = st.empty()

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.set_xlim(0, 600)
    ax.set_ylim(0, 100)

    ax.hlines(50, 0, 600, linewidth=6)
    ax.axvline(zone, linestyle="--", color="red")

    # move conveyor
    encoder_move(st.session_state.items, speed=speed)

    # process items safely
    for item in st.session_state.items:

        if check_trigger(item, zone) and item.result is None:
            try:
                item.result = run_pipeline(item.image)
            except Exception as e:
                item.result = {
                    "yolo": [],
                    "anomaly": [],
                    "unknown": [],
                    "error": str(e)
                }

        # visualization color logic
        color = "blue"

        if item.result:
            if len(item.result.get("unknown", [])) > 0:
                color = "red"
            else:
                color = "green"

        ax.scatter(item.x, 50, s=180, color=color)
        ax.text(item.x, 60, str(item.idx), fontsize=8)

    with placeholder:
        st.pyplot(fig)

# =========================
# RIGHT PANEL
# =========================
with right:

    st.subheader("🔍 Latest Inspection")

    latest = None

    for i in reversed(st.session_state.items):
        if i.result:
            latest = i.result
            break

    if latest:

        st.write("YOLO detections:", len(latest.get("yolo", [])))
        st.write("Anomalies:", len(latest.get("anomaly", [])))
        st.write("Unknown defects:", len(latest.get("unknown", [])))

        if "error" in latest:
            st.error(latest["error"])

    else:
        st.info("No inspection yet")

# =========================
# LOG TABLE
# =========================
st.divider()
st.subheader("📋 Inspection Log")

log_data = []

for item in st.session_state.items:

    if item.result:

        log_data.append({
            "ID": item.idx,
            "YOLO": len(item.result.get("yolo", [])),
            "Anomaly": len(item.result.get("anomaly", [])),
            "Unknown": len(item.result.get("unknown", [])),
            "X Position": item.x
        })

st.dataframe(log_data)
