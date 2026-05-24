import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Diabetic Foot Ulcer Detction",
    page_icon="🦶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; }
.stApp { background: #0b0e14; color: #e2e8f0; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

.dfu-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 24px; background: #131720;
    border: 1px solid #1e2535; border-radius: 14px; margin-bottom: 28px;
}
.dfu-logo {
    width: 46px; height: 46px;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 22px; flex-shrink: 0;
}
.dfu-title { font-size: 20px; font-weight: 700; color: #e2e8f0; margin: 0; }
.dfu-sub   { font-size: 12px; color: #64748b; margin: 2px 0 0; }

.section-card {
    background: #131720; border: 1px solid #1e2535;
    border-radius: 14px; padding: 22px 24px; margin-bottom: 18px;
}
.section-title {
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: #64748b; margin-bottom: 14px;
}

.result-healthy {
    display: flex; align-items: center; gap: 16px;
    background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.3);
    border-radius: 12px; padding: 20px 22px;
}
.result-healthy .rh-icon  { font-size: 32px; }
.result-healthy .rh-title { font-size: 18px; font-weight: 700; color: #22c55e; margin: 0; }
.result-healthy .rh-sub   { font-size: 13px; color: #64748b; margin: 4px 0 0; }

.result-ulcer {
    display: flex; align-items: center; gap: 16px;
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px; padding: 20px 22px; margin-bottom: 18px;
}
.result-ulcer .ru-icon  { font-size: 32px; }
.result-ulcer .ru-title { font-size: 18px; font-weight: 700; color: #ef4444; margin: 0; }
.result-ulcer .ru-sub   { font-size: 13px; color: #64748b; margin: 4px 0 0; }

.grade-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-top: 6px;
}
.grade-card {
    border: 1px solid #1e2535; border-radius: 10px;
    padding: 16px 10px; text-align: center; position: relative;
}
.grade-card .gc-label { font-size: 10px; color: #64748b; margin-bottom: 2px; }
.grade-card .gc-num   { font-size: 26px; font-weight: 700; color: #e2e8f0; }
.grade-card .gc-prob  { font-size: 12px; color: #64748b; margin-top: 4px; }
.grade-card .gc-desc  { font-size: 10px; color: #475569; margin-top: 6px; line-height: 1.4; }
.grade-card .gc-badge {
    position: absolute; top: -10px; left: 50%; transform: translateX(-50%);
    font-size: 9px; font-weight: 700; letter-spacing: .5px;
    padding: 2px 9px; border-radius: 20px;
    background: #3b82f6; color: #fff; white-space: nowrap;
}
.grade-1-active { border-color: #22c55e !important; background: rgba(34,197,94,0.07); }
.grade-1-active .gc-num { color: #22c55e !important; }
.grade-2-active { border-color: #f59e0b !important; background: rgba(245,158,11,0.07); }
.grade-2-active .gc-num { color: #f59e0b !important; }
.grade-3-active { border-color: #f97316 !important; background: rgba(249,115,22,0.07); }
.grade-3-active .gc-num { color: #f97316 !important; }
.grade-4-active { border-color: #ef4444 !important; background: rgba(239,68,68,0.07); }
.grade-4-active .gc-num { color: #ef4444 !important; }

.report-block {
    background: #0b0e14; border: 1px solid #1e2535; border-radius: 10px;
    padding: 16px 18px; font-size: 13px; color: #94a3b8;
    line-height: 1.8; white-space: pre-wrap;
}

div[data-testid="stFileUploadDropzone"] {
    background: #131720 !important; border: 2px dashed #1e2535 !important;
    border-radius: 12px !important;
}
div[data-testid="stFileUploadDropzone"]:hover { border-color: #3b82f6 !important; }
.stSlider [data-testid="stTickBar"] { display: none; }

div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    color: white !important; font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important; font-size: 15px !important;
    padding: 14px !important; border: none !important;
    border-radius: 12px !important; letter-spacing: 0.3px;
}
div[data-testid="stButton"] > button:hover { opacity: 0.87 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading DFU pipeline…")
def load_pipeline():
    from src.model_integration import DFUPipeline
    # Signature: DFUPipeline(stage1_path: str, stage2_path: str)
    return DFUPipeline(
        stage1_path="models/model_1.h5",
        stage2_path="models/model_2.pth",
    )


st.markdown("""
<div class="dfu-header">
  <div class="dfu-logo">🦶</div>
  <div>
    <p class="dfu-title">Diabetic Foot Ulcer Detection</p>
    <p class="dfu-sub">Diabetic Foot Ulcer · Two-Stage Wagner Grade Classification</p>
  </div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Stage 1 threshold",
        min_value=0.50, max_value=0.95, value=0.50, step=0.05,
        help="Raise to reduce false positives on dry/cracked skin."
    )
    st.caption("Default is 0.50 (pipeline internal). Raise if you see false positives.")


st.markdown('<div class="section-card"><div class="section-title">Upload Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a foot image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True, caption=uploaded.name)

    run = st.button("🔍  Analyse Image")

    if run:
        pipeline = load_pipeline()

        with st.spinner("Running pipeline…"):
            result = pipeline.predict(image)

        if result["has_ulcer"] and threshold > 0.50:
            raw_prob = pipeline.stage1.predict(image)["raw_prob"]
            if raw_prob < threshold:
                result["has_ulcer"]         = False
                result["stage1_confidence"] = round((1.0 - raw_prob) * 100, 2)

        st.divider()

        if not result["has_ulcer"]:
            conf = result.get("stage1_confidence", 0)
            st.markdown(f"""
            <div class="result-healthy">
              <div class="rh-icon">✅</div>
              <div>
                <p class="rh-title">No Ulcer Detected</p>
                <p class="rh-sub">Healthy confidence: {conf:.1f}%</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            s1_conf = result.get("stage1_confidence", 0)
            st.markdown(f"""
            <div class="result-ulcer">
              <div class="ru-icon">⚠️</div>
              <div>
                <p class="ru-title">Ulcer Detected</p>
                <p class="ru-sub">Stage 1 confidence: {s1_conf:.1f}%</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(s1_conf / 100, 1.0),
                text=f"Ulcer detection confidence: {s1_conf:.1f}%"
            )

            grade     = result.get("wagner_grade")           
            all_probs = result.get("all_grade_probs") or {}  
            colors    = ["grade-1-active", "grade-2-active", "grade-3-active", "grade-4-active"]
            descs     = [
                "Superficial ulcer",
                "Deep ulcer to tendon / capsule",
                "Deep ulcer with abscess / osteitis",
                "Partial foot gangrene",
            ]

            cards = []
            for i in range(1, 5):
                active_cls = colors[i - 1] if grade == i else ""
                badge      = '<div class="gc-badge">PREDICTED</div>' if grade == i else ""
                p          = all_probs.get("Grade " + str(i))
                prob_str   = ("%.1f%%" % p) if p is not None else "&mdash;"
                desc       = descs[i - 1].replace("/", "&#47;")
                cards.append(
                    '<div class="grade-card ' + active_cls + '">'
                    + badge
                    + '<div class="gc-label">Grade</div>'
                    + '<div class="gc-num">' + str(i) + '</div>'
                    + '<div class="gc-prob">' + prob_str + '</div>'
                    + '<div class="gc-desc">' + desc + '</div>'
                    + '</div>'
                )
            grade_html = (
                '<div class="section-card">'
                '<div class="section-title">WAGNER CLASSIFICATION</div>'
                '<div class="grade-grid">'
                + "".join(cards)
                + '</div></div>'
            )
            st.markdown(grade_html, unsafe_allow_html=True)

        summary = result.get("summary")
        if summary:
            st.markdown(f"""
            <div class="section-card">
              <div class="section-title">Clinical Summary</div>
              <div class="report-block">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Upload a foot image above to begin analysis.", icon="🖼️")
