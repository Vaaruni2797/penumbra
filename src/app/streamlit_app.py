"""
streamlit_app.py — Penumbra demo UI.
Clean/minimal design with purple gradient confidence colors.
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
    page_icon="🌒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Clean base */
    .main { background: #ffffff; }
    
    .penumbra-title {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .penumbra-sub {
        font-size: 1.05rem;
        color: #ffffff;
        margin-top: 4px;
        margin-bottom: 1.5rem;
    }
    .penumbra-tagline {
        font-style: italic;
        color: #c4b5fd;
        font-size: 0.95rem;
    }

    /* Confidence cards — purple gradient */
    .claim-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    .high-conf {
        background: #f5f0ff;
        border-color: #7c3aed;
    }
    .med-conf {
        background: #faf5ff;
        border-color: #a855f7;
    }
    .low-conf {
        background: #fdf4ff;
        border-color: #d8b4fe;
    }

    /* Side by side panels */
    .base-panel {
        background: #000000;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #374151;
        color: #ffffff;
    }
    .penumbra-panel {
        background: #faf5ff;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #e9d5ff;
    }

    /* W&B badge */
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

    /* Metric strip */
    .metric-strip {
        background: #faf5ff;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9d5ff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def confidence_to_style(confidence: float):
    """Map confidence to CSS class and emoji — purple gradient."""
    if confidence >= 0.7:
        return "high-conf", "🟣"
    elif confidence >= 0.4:
        return "med-conf", "🔵"
    else:
        return "low-conf", "⚪"


def render_uncertainty_map(uncertainty_map: dict):
    if "error" in uncertainty_map:
        st.error(f"Parse error: {uncertainty_map.get('raw', uncertainty_map['error'])[:300]}")
        return

    # Answer
    st.markdown("**Answer:**")
    st.write(uncertainty_map.get("answer", ""))
    st.divider()

    # Overall confidence gauge — purple
    overall_conf = uncertainty_map.get("overall_confidence", 0.5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_conf * 100,
        title={"text": "Overall Confidence", "font": {"size": 13, "color": "#6b21a8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b21a8"},
            "bar": {"color": "#7c3aed" if overall_conf > 0.7 else "#a855f7" if overall_conf > 0.4 else "#d8b4fe"},
            "steps": [
                {"range": [0, 40],  "color": "#fdf4ff"},
                {"range": [40, 70], "color": "#faf5ff"},
                {"range": [70, 100],"color": "#f5f0ff"}
            ],
            "bordercolor": "#e9d5ff"
        },
        number={"suffix": "%", "font": {"size": 22, "color": "#1a1a2e"}}
    ))
    fig.update_layout(height=175, margin=dict(t=30, b=0, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Epistemic summary
    if uncertainty_map.get("epistemic_summary"):
        st.markdown(
            f'<div style="background:#f5f0ff;border-left:3px solid #7c3aed;'
            f'padding:10px 14px;border-radius:6px;font-size:0.9rem;color:#4c1d95;">'
            f'📊 {uncertainty_map["epistemic_summary"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

    # Claims
    st.markdown("**Claim-by-claim breakdown:**")
    claims = uncertainty_map.get("claims", [])
    if not claims:
        st.warning("No claims extracted.")
        return

    for claim in claims:
        conf = claim.get("confidence", 0.5)
        css_class, emoji = confidence_to_style(conf)
        conf_pct = int(conf * 100)
        alt = claim.get("alternative_views")

        st.markdown(f"""
        <div class="claim-card {css_class}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <strong style="color:#1a1a2e">{emoji} {claim['claim']}</strong>
                <span style="font-size:1.15rem;font-weight:700;color:#6b21a8">{conf_pct}%</span>
            </div>
            <div style="margin-top:5px;font-size:0.85rem;color:#555">{claim.get('basis','')}</div>
            {f'<div style="margin-top:4px;font-size:0.8rem;color:#7c3aed;font-style:italic">⚖️ {alt}</div>' if alt else ''}
        </div>
        """, unsafe_allow_html=True)

    # Horizontal bar chart — purple
    if len(claims) > 1:
        names = [c["claim"][:45] + "…" if len(c["claim"]) > 45 else c["claim"] for c in claims]
        confs = [c["confidence"] for c in claims]
        bar_colors = ["#7c3aed" if c >= 0.7 else "#a855f7" if c >= 0.4 else "#d8b4fe" for c in confs]

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

    if uncertainty_map.get("least_certain_claim"):
        st.markdown(
            f'<div style="background:#fdf4ff;border:1px solid #d8b4fe;border-radius:6px;'
            f'padding:8px 14px;font-size:0.85rem;color:#6b21a8;">'
            f'⚠️ <strong>Weakest claim:</strong> {uncertainty_map["least_certain_claim"]}</div>',
            unsafe_allow_html=True
        )


def main():
    # Header
    st.markdown('<p class="penumbra-title">🌒 Penumbra</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="penumbra-sub">Epistemic transparency for language models &nbsp;·&nbsp; '
        '<span class="penumbra-tagline">Between knowing and guessing, there\'s a shadow.</span></p>',
        unsafe_allow_html=True
    )

    wandb_url = os.getenv("WANDB_RUN_URL", "https://wandb.ai")
    st.markdown(
        f'<a href="{wandb_url}" target="_blank" class="wandb-badge">📊 W&B Training Run</a>',
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

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Question", placeholder="Ask anything…",
            label_visibility="collapsed"
        )
    with col2:
        demo_q = st.selectbox(
            "Demo", ["Try a demo…"] + demo_questions,
            label_visibility="collapsed"
        )
    if demo_q != "Try a demo…":
        question = demo_q

    # Sidebar
    st.sidebar.markdown("### ⚙️ Settings")
    use_local = st.sidebar.checkbox("Use local model", value=False)
    use_hub   = st.sidebar.checkbox("Load from HuggingFace Hub", value=True)

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
            predictor = get_predictor(use_local=use_local, use_hub=use_hub)
            st.subheader("📊 Batch Uncertainty Analysis")
            for i, q in enumerate(questions_from_file[:10]):
                with st.expander(f"Q{i+1}: {q[:70]}"):
                    result = predictor.predict(q)
                    if "claims" in result:
                        st.write(f"**Overall confidence:** {result.get('overall_confidence', 0):.0%}")
                        for claim in result["claims"]:
                            conf = claim["confidence"]
                            emoji = "🟣" if conf >= 0.7 else "🔵" if conf >= 0.4 else "⚪"
                            st.write(f"{emoji} {claim['claim']} — **{conf:.0%}**")
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
- Subprime lending: 🟣 95%
- Deregulation as cause: 🔵 61% *(contested)*
- CDOs as mechanism: 🔵 73%

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

    # Main generation
    if st.button("🌒 Generate Uncertainty Map", type="primary"):
        predictor = get_predictor(use_local=use_local, use_hub=use_hub)

        col_base, col_penumbra = st.columns(2)

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

        with col_penumbra:
            st.markdown("### 🌒 Penumbra")
            st.caption("Uncertainty-aware — maps confidence claim by claim")
            with st.spinner("Mapping uncertainty…"):
                uncertainty_map = predictor.predict(question)
            render_uncertainty_map(uncertainty_map)

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