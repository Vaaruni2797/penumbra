"""
streamlit_app.py — Penumbra demo UI.
"""

import json
import sys
import os
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from inference.predict import get_predictor

st.set_page_config(
    page_title="Penumbra",
    page_icon="🌔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Tighten top padding */
    div[data-testid="stAppViewContainer"] > section > div.block-container {
        padding-top: 0.5rem !important;
    }

    .penumbra-title {
        font-size: 7rem !important;
        font-weight: 900 !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        color: #ffffff !important;
        letter-spacing: -1px;
        margin-bottom: 0;
        margin-top: 0;
        line-height: 1;
    }
    .penumbra-sub {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        color: #ffffff;
        margin-top: 4px;
        margin-bottom: 0.2rem;
    }
    .penumbra-tagline {
        display: block;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        font-style: italic;
        color: #c4b5fd;
        margin-top: 4px;
        margin-bottom: 0.8rem;
    }

    /* Confidence cards */
    .claim-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid;
    }

    /* Panels */
    .base-panel {
        background: #000000;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #374151;
        color: #ffffff;
        min-height: 300px;
    }
    .penumbra-panel {
        background: #faf5ff;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #e9d5ff;
        min-height: 300px;
    }

    /* Epistemic summary — tight margin */
    .epistemic-box {
        background: #f5f0ff;
        border-left: 3px solid #7c3aed;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #4c1d95;
        margin: 4px 0 8px 0;
    }

    .wandb-badge {
        background: #fef3c7;
        color: #92400e;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-decoration: none;
        border: 1px solid #fde68a;
    }
</style>
""", unsafe_allow_html=True)


def conf_to_color(c: float) -> str:
    """Interpolate: dark purple #3b0764 → purple #7c3aed → yellow #eab308"""
    if c >= 0.7:
        t = (c - 0.7) / 0.3
        r = int(124 + t * (234 - 124))
        g = int(58  + t * (179 - 58))
        b = int(237 + t * (8   - 237))
    else:
        t = c / 0.7
        r = int(59  + t * (124 - 59))
        g = int(7   + t * (58  - 7))
        b = int(100 + t * (237 - 100))
    return f"rgb({r},{g},{b})"


def confidence_to_css(confidence: float):
    if confidence >= 0.7:
        return "high-conf"
    elif confidence >= 0.4:
        return "med-conf"
    else:
        return "low-conf"


def render_uncertainty_map(uncertainty_map: dict):
    if "error" in uncertainty_map:
        st.error(f"Parse error: {uncertainty_map.get('raw', uncertainty_map['error'])[:300]}")
        return

    claims       = uncertainty_map.get("claims", [])
    overall_conf = uncertainty_map.get("overall_confidence", 0.5)

    with st.container(border=True):
        # ── 1. Gauge ──
        st.caption("📊 Overall Confidence")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_conf * 100,
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b21a8"},
                "bar": {"color": conf_to_color(overall_conf)},
                "steps": [
                    {"range": [0,  40], "color": "#ede9fe"},
                    {"range": [40, 70], "color": "#f5f0ff"},
                    {"range": [70,100], "color": "#fefce8"}
                ],
                "bordercolor": "#e9d5ff"
            },
            number={"suffix": "%", "font": {"size": 22, "color": "#1a1a2e"}}
        ))
        fig.update_layout(height=150, margin=dict(t=10, b=0, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # ── 2. Epistemic summary ──
        if uncertainty_map.get("epistemic_summary"):
            st.markdown(
                f'<div class="epistemic-box">📊 {uncertainty_map["epistemic_summary"]}</div>',
                unsafe_allow_html=True
            )

        # ── 3. Answer ──
        st.markdown("**Answer:**")
        st.write(uncertainty_map.get("answer", ""))

        # ── 4. Weakest claim ──
        if uncertainty_map.get("least_certain_claim"):
            st.markdown(
                f'<div style="background:#ede9fe;border:1px solid #a78bfa;border-radius:6px;'
                f'padding:6px 12px;font-size:0.85rem;color:#3b0764;margin:6px 0;">'
                f'⚠️ <strong>Weakest claim:</strong> {uncertainty_map["least_certain_claim"]}</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # ── 5. Horizontal bar chart ──
        if len(claims) > 1:
            names      = [c["claim"][:45] + "…" if len(c["claim"]) > 45 else c["claim"] for c in claims]
            confs      = [c["confidence"] for c in claims]
            bar_colors = [conf_to_color(c) for c in confs]

            fig2 = go.Figure(go.Bar(
                x=confs, y=names, orientation="h",
                marker_color=bar_colors,
                marker_line_width=0
            ))
            fig2.update_layout(
                xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#f3f4f6"),
                yaxis=dict(tickfont=dict(size=11)),
                height=max(180, len(claims) * 38),
                margin=dict(t=10, b=10, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── 6. Claim cards ──
        st.markdown("**Claim-by-claim breakdown:**")
        if not claims:
            st.warning("No claims extracted.")
            return

        for claim in claims:
            conf     = claim.get("confidence", 0.5)
            color    = conf_to_color(conf)
            conf_pct = int(conf * 100)
            alt      = claim.get("alternative_views")

            # Derive rgba background from the confidence color (low opacity tint)
            rgb_parts = color[4:-1]  # strip "rgb(" and ")"
            bg_color  = f"rgba({rgb_parts}, 0.12)"

            st.markdown(f"""
            <div class="claim-card" style="background:{bg_color};border-color:{color}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <strong style="color:#f3f4f6">{claim['claim']}</strong>
                    <span style="font-size:1.1rem;font-weight:700;color:#f3f4f6">{conf_pct}%</span>
                </div>
                <div style="margin-top:5px;font-size:0.85rem;color:#d1d5db">{claim.get('basis','')}</div>
                {f'<div style="margin-top:4px;font-size:0.8rem;color:#c4b5fd;font-style:italic">⚖️ {alt}</div>' if alt else ''}
            </div>
            """, unsafe_allow_html=True)





def main():
    # Header
    st.markdown('<p class="penumbra-title">🌔 Penumbra</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="penumbra-sub">Between knowing and guessing, there\'s a shadow.</p>'
        '<span class="penumbra-tagline">Epistemic transparency for language models.</span>',
        unsafe_allow_html=True
    )

    st.divider()

    demo_questions = [
        "What caused the 2008 financial crisis?",
        "Is coffee good or bad for your health?",
        "What will AI look like in 2030?",
        "Did the Roman Empire fall due to barbarian invasions?",
        "Is nuclear energy safe?",
        "What is the speed of light?",
        "What causes depression?",
        "Was the moon landing faked?",
    ]

    # Initialise session state for the question box
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    col1, col2 = st.columns([3, 1])

    # Render selectbox first so we can update session_state before the
    # text_input widget below reads it on this same script run.
    with col2:
        demo_q = st.selectbox(
            "Demo", ["Try a demo…"] + demo_questions,
            label_visibility="collapsed"
        )
        if demo_q != "Try a demo…":
            st.session_state["question_input"] = demo_q

    with col1:
        question = st.text_input(
            "Question", placeholder="Ask anything…",
            label_visibility="collapsed",
            key="question_input"
        )

    # Sidebar
    st.sidebar.markdown("### ⚙️ Settings")
    use_local = st.sidebar.checkbox("Use local model", value=True)

    wandb_url = os.getenv("WANDB_RUN_URL", "https://wandb.ai")
    st.sidebar.divider()
    st.sidebar.markdown(
        f'<a href="{wandb_url}" target="_blank" class="wandb-badge">📊 W&B Training Run</a>',
        unsafe_allow_html=True
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🧪 Test Your Own Data")
    st.sidebar.caption("Upload a .txt or .jsonl file — one question per line")
    uploaded_file = st.sidebar.file_uploader(
        "Upload", type=["txt", "jsonl"], label_visibility="collapsed"
    )

    if uploaded_file:
        lines = uploaded_file.read().decode().strip().split("\n")
        questions_from_file = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                questions_from_file.append(obj.get("question", line))
            except json.JSONDecodeError:
                questions_from_file.append(line)

        st.sidebar.success(f"Loaded {len(questions_from_file)} questions")

        if st.sidebar.button("🔍 Run Batch Analysis", type="primary"):
            predictor = get_predictor(use_local=use_local)
            st.subheader("📊 Batch Uncertainty Analysis")
            for i, q in enumerate(questions_from_file[:10]):
                with st.expander(f"Q{i+1}: {q[:70]}"):
                    result = predictor.predict(q)
                    if "claims" in result:
                        st.write(f"**Overall confidence:** {result.get('overall_confidence', 0):.0%}")
                        for claim in result["claims"]:
                            conf  = claim["confidence"]
                            color = conf_to_color(conf)
                            st.markdown(
                                f'<span style="color:{color};font-weight:700">●</span> '
                                f'{claim["claim"]} — <strong>{conf:.0%}</strong>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.error("Failed to parse uncertainty map")

    # Landing
    if not question:
        st.markdown("### Ask a question above to see Penumbra in action")
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**Without Penumbra:**
> "The 2008 financial crisis was caused by subprime mortgage lending, deregulation, and CDOs..."

*No signal about what to trust.*
""")
        with col2:
            st.markdown("""
**With Penumbra:**
- Subprime lending: 🟡 95%
- Deregulation as cause: 🟣 61% *(contested)*
- CDOs as mechanism: 🟣 73%

*You know exactly what to verify.*
""")
        with col3:
            st.markdown("""
**Why it matters:**
- Catch hallucinations early
- Know which claims need checking
- Calibrated trust in AI outputs
- Built on Ministral 8B + QLoRA
""")
        return

    # Main — Penumbra LEFT, Base RIGHT
    if st.button("🌔 Generate Uncertainty Map", type="primary"):
        predictor = get_predictor(use_local=use_local)

        col_penumbra, col_base = st.columns(2)

        with col_penumbra:
            st.markdown("### 🌔 Penumbra")
            st.caption("Uncertainty-aware — maps confidence claim by claim")
            with st.spinner("Mapping uncertainty…"):
                uncertainty_map = predictor.predict(question)
            render_uncertainty_map(uncertainty_map)

        with col_base:
            st.markdown("### 🤖 Base Ministral 8B")
            st.caption("Standard — no uncertainty awareness")
            with st.spinner("Generating…"):
                base_response = predictor.get_base_response(question)
            st.markdown(
                f'<div class="base-panel">{base_response}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div style="margin-top:8px;font-size:0.85rem;color:#9ca3af;">'
                '⚠️ Confident tone regardless of actual certainty</div>',
                unsafe_allow_html=True
            )

        # Stats strip
        if "claims" in uncertainty_map:
            st.divider()
            claims = uncertainty_map["claims"]
            confs  = [c["confidence"] for c in claims]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Claims Analyzed", len(claims))
            c2.metric("Avg Confidence", f"{sum(confs)/len(confs):.0%}")
            c3.metric("Uncertain Claims", sum(1 for c in confs if c < 0.6))
            c4.metric("Overall Confidence", f"{uncertainty_map.get('overall_confidence', 0):.0%}")


if __name__ == "__main__":
    main()