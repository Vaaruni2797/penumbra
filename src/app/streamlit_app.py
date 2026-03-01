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
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stAppViewContainer"] > section > div.block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0 !important;
    }

    /* ── Wider sidebar ── */
    div[data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 320px !important;
    }
    div[data-testid="stSidebar"] > div:first-child {
        min-width: 320px !important;
    }

    /* ── Title — centered, clickable ── */
    .penumbra-title-link {
        display: block;
        font-size: 7rem !important;
        font-weight: 900 !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        color: #ffffff !important;
        letter-spacing: -1px;
        margin-bottom: 0;
        margin-top: 0;
        line-height: 1;
        text-align: center;
        text-decoration: none !important;
        cursor: pointer;
        transition: opacity 0.15s;
    }
    .penumbra-title-link:hover { opacity: 0.8; }

    .penumbra-sub {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        color: #ffffff;
        margin-top: 6px;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    .penumbra-tagline {
        display: block;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        font-style: italic;
        color: #c4b5fd;
        margin-top: 2px;
        margin-bottom: 1.2rem;
        text-align: center;
    }

    /* ── Question bubble ── */
    .question-bubble {
        background: rgba(124, 58, 237, 0.18);
        border: 1px solid rgba(124, 58, 237, 0.45);
        border-radius: 14px;
        padding: 14px 22px;
        margin: 0 auto 20px auto;
        max-width: 800px;
        font-size: 1.05rem;
        font-weight: 500;
        color: #e9d5ff;
        text-align: center;
        font-family: "Helvetica Neue", sans-serif;
    }
    .question-bubble-label {
        font-size: 0.72rem;
        color: #7c3aed;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 4px;
    }

    /* ── Confidence cards ── */
    .claim-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid;
    }

    /* ── Base panel ── */
    .base-panel {
        background: #0d0d0d;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #374151;
        color: #e5e7eb;
        min-height: 300px;
        line-height: 1.7;
    }

    /* ── Epistemic summary ── */
    .epistemic-box {
        background: rgba(124, 58, 237, 0.15);
        border-left: 3px solid #7c3aed;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #c4b5fd;
        margin: 4px 0 8px 0;
    }

    /* ── W&B badge ── */
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

    /* ── Sidebar buttons ── */
    div[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: none;
        color: #d1d5db;
        text-align: left;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: background 0.15s;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(124, 58, 237, 0.2);
        color: #f3f4f6;
    }

    /* ── Nav active ── */
    .nav-btn-active > button {
        background: rgba(124, 58, 237, 0.28) !important;
        color: #e9d5ff !important;
        font-weight: 700 !important;
        border-left: 3px solid #7c3aed !important;
    }

    /* ── Ask a Question button ── */
    .ask-btn > button {
        background: rgba(124, 58, 237, 0.18) !important;
        border: 1px solid rgba(124, 58, 237, 0.4) !important;
        color: #e9d5ff !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        margin-bottom: 8px;
    }
    .ask-btn > button:hover {
        background: rgba(124, 58, 237, 0.35) !important;
    }

    /* ── About page ── */
    .about-section {
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 10px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .about-section h3 {
        color: #c4b5fd;
        font-family: "Helvetica Neue", sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Confidence display ── */
    .conf-display {
        text-align: center;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        margin: -8px 0 4px 0;
    }
    .conf-label {
        font-size: 0.75rem;
        color: #a78bfa;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .conf-value {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .conf-qualifier {
        font-size: 0.88rem;
        color: #9ca3af;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def conf_to_color(c: float) -> str:
    """Interpolate: dark purple → purple → violet → amber → yellow"""
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


def conf_to_qualifier(c: float) -> str:
    if c >= 0.85: return "Very High"
    if c >= 0.70: return "High"
    if c >= 0.50: return "Moderate"
    if c >= 0.30: return "Low"
    return "Very Low"


# ─────────────────────────────────────────────────────────────
# Gauge — smooth gradient arc, needle, labels outside chart
# ─────────────────────────────────────────────────────────────

def render_gauge(overall_conf: float):
    color     = conf_to_color(overall_conf)
    qualifier = conf_to_qualifier(overall_conf)
    pct       = int(overall_conf * 100)

    # 40-step smooth gradient: dark purple → purple → violet → amber → yellow
    smooth_steps = [
        {"range": [0.0,   2.5],  "color": "#3c0868"},
        {"range": [2.5,   5.0],  "color": "#3f0a70"},
        {"range": [5.0,   7.5],  "color": "#430d78"},
        {"range": [7.5,  10.0],  "color": "#461080"},
        {"range": [10.0, 12.5],  "color": "#491288"},
        {"range": [12.5, 15.0],  "color": "#4c1591"},
        {"range": [15.0, 17.5],  "color": "#4f1799"},
        {"range": [17.5, 20.0],  "color": "#531aa1"},
        {"range": [20.0, 22.5],  "color": "#561da9"},
        {"range": [22.5, 25.0],  "color": "#591fb1"},
        {"range": [25.0, 27.5],  "color": "#5d22ba"},
        {"range": [27.5, 30.0],  "color": "#6226c2"},
        {"range": [30.0, 32.5],  "color": "#672aca"},
        {"range": [32.5, 35.0],  "color": "#6b2dd2"},
        {"range": [35.0, 37.5],  "color": "#7031da"},
        {"range": [37.5, 40.0],  "color": "#7535e2"},
        {"range": [40.0, 42.5],  "color": "#7a38ea"},
        {"range": [42.5, 45.0],  "color": "#7f39ec"},
        {"range": [45.0, 47.5],  "color": "#8337ec"},
        {"range": [47.5, 50.0],  "color": "#8736eb"},
        {"range": [50.0, 52.5],  "color": "#8c35ea"},
        {"range": [52.5, 55.0],  "color": "#9033ea"},
        {"range": [55.0, 57.5],  "color": "#9537eb"},
        {"range": [57.5, 60.0],  "color": "#9a3fee"},
        {"range": [60.0, 62.5],  "color": "#a048f2"},
        {"range": [62.5, 65.0],  "color": "#a550f5"},
        {"range": [65.0, 67.5],  "color": "#ab5af7"},
        {"range": [67.5, 70.0],  "color": "#b166f8"},
        {"range": [70.0, 72.5],  "color": "#b772fa"},
        {"range": [72.5, 75.0],  "color": "#bd7efb"},
        {"range": [75.0, 77.5],  "color": "#c381d5"},
        {"range": [77.5, 80.0],  "color": "#cb7d88"},
        {"range": [80.0, 82.5],  "color": "#d3793b"},
        {"range": [82.5, 85.0],  "color": "#dc7b06"},
        {"range": [85.0, 87.5],  "color": "#e68908"},
        {"range": [87.5, 90.0],  "color": "#f0970a"},
        {"range": [90.0, 92.5],  "color": "#f3a00a"},
        {"range": [92.5, 95.0],  "color": "#f0a509"},
        {"range": [95.0, 97.5],  "color": "#eeab09"},
        {"range": [97.5, 100.0], "color": "#ebb008"},
    ]

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=overall_conf * 100,
        gauge={
            "bar": {"color": "rgba(0,0,0,0)", "thickness": 0.02},
            "threshold": {
                "line": {"color": "#ffffff", "width": 4},
                "thickness": 0.85,
                "value": overall_conf * 100,
            },
            "axis": {
                "range": [0, 100],
                # Place labels away from the arc: Very Low/High at edges, others spread
                "tickvals": [5, 25, 50, 75, 95],
                "ticktext": ["Very\nLow", "Low", "Moderate", "High", "Very\nHigh"],
                "tickcolor": "rgba(0,0,0,0)",
                "tickfont": {"size": 11, "color": "#c4b5fd"},
                "tickwidth": 0,
            },
            "steps": smooth_steps,
            "bordercolor": "rgba(0,0,0,0)",
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(t=20, b=0, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # % + qualifier below the chart
    st.markdown(
        f'<div class="conf-display">'
        f'  <div class="conf-label">Overall Confidence</div>'
        f'  <div class="conf-value" style="color:{color}">{pct}%</div>'
        f'  <div class="conf-qualifier">{qualifier}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Uncertainty map renderer
# ─────────────────────────────────────────────────────────────

def render_uncertainty_map(uncertainty_map: dict):
    if "error" in uncertainty_map:
        st.error(f"Parse error: {uncertainty_map.get('raw', uncertainty_map['error'])[:300]}")
        return

    claims       = uncertainty_map.get("claims", [])
    overall_conf = uncertainty_map.get("overall_confidence", 0.5)

    with st.container(border=True):
        # 1. Gauge
        render_gauge(overall_conf)

        # 2. Answer — right below speedometer
        st.markdown(
            '<p style="font-size:1.05rem;font-weight:700;color:#e9d5ff;margin:10px 0 4px 0;">Answer</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:1.05rem;color:#f3f4f6;line-height:1.7;">{uncertainty_map.get("answer", "")}</div>',
            unsafe_allow_html=True,
        )

        # 3. Reasoning / epistemic summary
        if uncertainty_map.get("epistemic_summary"):
            st.markdown(
                f'<div class="epistemic-box" style="font-size:0.95rem;">📊 {uncertainty_map["epistemic_summary"]}</div>',
                unsafe_allow_html=True,
            )

        # 4. Weakest claim
        if uncertainty_map.get("least_certain_claim"):
            st.markdown(
                f'<div style="background:rgba(59,7,100,0.35);border:1px solid #7c3aed;'
                f'border-radius:6px;padding:8px 14px;font-size:0.95rem;color:#e9d5ff;margin:8px 0;">'
                f'⚠️ <strong>Weakest claim:</strong> {uncertainty_map["least_certain_claim"]}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # 5. Claim cards — sorted low → high (before bar chart)
        st.markdown(
            '<p style="font-size:1.05rem;font-weight:700;color:#e9d5ff;margin:0 0 6px 0;">Claim-by-claim breakdown</p>',
            unsafe_allow_html=True,
        )
        if not claims:
            st.warning("No claims extracted.")
            return

        for claim in sorted(claims, key=lambda c: c.get("confidence", 0), reverse= True):
            conf      = claim.get("confidence", 0.5)
            color     = conf_to_color(conf)
            conf_pct  = int(conf * 100)
            alt       = claim.get("alternative_views")
            rgb_parts = color[4:-1]
            bg_color  = f"rgba({rgb_parts}, 0.15)"

            st.markdown(f"""
            <div class="claim-card" style="background:{bg_color};border-color:{color}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <strong style="color:#f3f4f6;font-size:1rem;">{claim['claim']}</strong>
                    <span style="font-size:1.2rem;font-weight:700;color:#f3f4f6">{conf_pct}%</span>
                </div>
                <div style="margin-top:6px;font-size:0.95rem;color:#d1d5db">{claim.get('basis','')}</div>
                {f'<div style="margin-top:5px;font-size:0.9rem;color:#c4b5fd;font-style:italic">⚖️ {alt}</div>' if alt else ''}
            </div>
            """, unsafe_allow_html=True)

        # 6. Bar chart — after cards
        if len(claims) > 1:
            st.markdown(
                '<p style="font-size:1.0rem;font-weight:600;color:#a78bfa;margin:14px 0 4px 0;">Confidence overview</p>',
                unsafe_allow_html=True,
            )
            sorted_claims = sorted(claims, key=lambda c: c.get("confidence", 0))
            names      = [c["claim"][:45] + "…" if len(c["claim"]) > 45 else c["claim"] for c in sorted_claims]
            confs      = [c["confidence"] for c in sorted_claims]
            bar_colors = [conf_to_color(c) for c in confs]

            fig2 = go.Figure(go.Bar(
                x=confs, y=names, orientation="h",
                marker_color=bar_colors,
                marker_line_width=0,
            ))
            fig2.update_layout(
                xaxis=dict(
                    range=[0, 1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                    tickfont=dict(size=12, color="#1a1a2e"),
                    gridcolor="#d1d5db",
                    showgrid=True,
                ),
                yaxis=dict(tickfont=dict(size=12, color="#f3f4f6")),
                height=max(180, len(sorted_claims) * 40),
                margin=dict(t=10, b=30, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.92)",
            )
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():

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

    # Session state
    if "question"         not in st.session_state: st.session_state.question         = ""
    if "results_for"      not in st.session_state: st.session_state.results_for      = ""
    if "uncertainty_map"  not in st.session_state: st.session_state.uncertainty_map  = None
    if "base_response"    not in st.session_state: st.session_state.base_response    = ""
    if "penumbra_ready"   not in st.session_state: st.session_state.penumbra_ready   = False
    if "base_ready"       not in st.session_state: st.session_state.base_ready       = False
    if "page"             not in st.session_state: st.session_state.page             = "chat"

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<p style="font-size:1.3rem;font-weight:900;color:#c4b5fd;'
            'font-family:\'Helvetica Neue\',sans-serif;margin:0 0 4px 0;">🌔 Penumbra</p>',
            unsafe_allow_html=True,
        )
        st.caption("Uncertainty-aware language model")
        st.divider()

        # Page navigation
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            chat_cls = "nav-btn-active" if st.session_state.page == "chat" else ""
            st.markdown(f'<div class="{chat_cls}">', unsafe_allow_html=True)
            if st.button("💬  Chat", key="nav_chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with nav_col2:
            about_cls = "nav-btn-active" if st.session_state.page == "about" else ""
            st.markdown(f'<div class="{about_cls}">', unsafe_allow_html=True)
            if st.button("ℹ️  About", key="nav_about", use_container_width=True):
                st.session_state.page = "about"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        if st.session_state.page == "chat":
            # Ask a Question — clears and goes home
            st.markdown('<div class="ask-btn">', unsafe_allow_html=True)
            if st.button("❓  Ask a Question", key="ask_a_question", use_container_width=True):
                st.session_state.question       = ""
                st.session_state.results_for    = ""
                st.session_state.penumbra_ready = False
                st.session_state.base_ready     = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                '<p style="font-size:0.72rem;font-weight:600;color:#6b7280;'
                'text-transform:uppercase;letter-spacing:0.06em;margin:2px 0 4px 0;">Try a demo</p>',
                unsafe_allow_html=True,
            )
            for i, q in enumerate(demo_questions):
                label = ("💬  " + q[:36] + "…") if len(q) > 36 else ("💬  " + q)
                if st.button(label, key=f"demo_{i}", use_container_width=True):
                    st.session_state.question       = q
                    st.session_state.results_for    = ""
                    st.session_state.penumbra_ready = False
                    st.session_state.base_ready     = False
                    st.session_state.page           = "chat"
                    st.rerun()

        st.divider()
        st.markdown("### ⚙️ Settings")
        use_local = st.checkbox("Use local model", value=True)

        wandb_url = os.getenv("WANDB_RUN_URL", "https://wandb.ai")
        st.divider()
        st.markdown(
            f'<a href="{wandb_url}" target="_blank" class="wandb-badge">📊 W&B Training Run</a>',
            unsafe_allow_html=True,
        )

        if st.session_state.page == "chat":
            st.divider()
            st.markdown("### 🧪 Batch Analysis")
            st.caption("Upload a .txt or .jsonl file — one question per line")
            uploaded_file = st.file_uploader(
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
                st.success(f"Loaded {len(questions_from_file)} questions")
                if st.button("🔍 Run Batch Analysis", type="primary"):
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
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.error("Failed to parse uncertainty map")

    # ── Header — title is a link that resets to home ──────────
    st.markdown(
        '<div style="text-align:center;">'
        '  <a class="penumbra-title-link" href="/" style="display:inline-block;">🌔 Penumbra</a>'
        '  <p class="penumbra-sub">Between knowing and guessing, there\'s a shadow.</p>'
        '  <span class="penumbra-tagline">Know what to trust. Know what to question.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ══════════════════════════════════════════════════════════
    # ABOUT PAGE
    # ══════════════════════════════════════════════════════════
    if st.session_state.page == "about":

        def about_block(icon_title, body_html):
            st.markdown(
                f'''<div class="about-section">
                    <h3>{icon_title}</h3>
                    <div style="font-size:0.97rem;color:#d1d5db;line-height:1.75;">{body_html}</div>
                </div>''',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="text-align:center;margin-bottom:28px;"><p style="font-size:2.2rem;font-weight:900;color:#ffffff;font-family:Helvetica Neue,sans-serif;margin:0;">ℹ️ About Penumbra</p><p style="font-size:1rem;color:#a78bfa;margin-top:6px;font-style:italic;">Built for the Mistral Hackathon 2026</p></div>',
            unsafe_allow_html=True,
        )

        about_block("🎯 The Problem", """
Every AI model — GPT-4, Claude, Gemini, Ministral — outputs answers in the same confident tone,
whether it is certain or completely guessing. A hallucination sounds identical to a well-established fact.
There is no signal to tell you which claims to trust, which to question, and which to verify before acting on them.
<br><br>
This is especially dangerous in high-stakes domains: medical decisions, financial analysis, legal research, and enterprise workflows
where a single confident-sounding but wrong claim can cause real damage.
Penumbra exists to fix that.
""")

        about_block("🧠 How It Works", """
<strong style="color:#c4b5fd;">1. Collect</strong> — Pulled TruthfulQA, TriviaQA, and FEVER: three benchmark datasets covering factual claims
with known ground-truth labels, uncertainty levels, and contested areas.<br><br>
<strong style="color:#c4b5fd;">2. Annotate</strong> — Used Mistral Large 3 to decompose each answer into individual claims and assign a structured
uncertainty map: confidence score, basis for that score, and alternative views where the evidence is contested.<br><br>
<strong style="color:#c4b5fd;">3. Generate</strong> — Synthesised additional QA pairs across high / medium / low certainty domains
to balance the training distribution and expose the model to diverse uncertainty patterns.<br><br>
<strong style="color:#c4b5fd;">4. Finetune</strong> — Trained Ministral 8B via QLoRA to always produce a structured JSON uncertainty map
alongside every answer: claim-level confidence scores, reasoning basis, and flagged alternatives.<br><br>
<strong style="color:#c4b5fd;">5. Demo</strong> — Side-by-side UI comparing the base model's confident prose against Penumbra's mapped output,
so the before/after delta is immediately visible.
""")

        about_block("📊 Training Details", """
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;">
  <div><span style="color:#7c3aed;font-weight:600;">Base model</span><br>mistralai/Ministral-8B-Instruct-2410</div>
  <div><span style="color:#7c3aed;font-weight:600;">Method</span><br>QLoRA (r=4, alpha=8)</div>
  <div><span style="color:#7c3aed;font-weight:600;">Target modules</span><br>q_proj, v_proj</div>
  <div><span style="color:#7c3aed;font-weight:600;">Training data</span><br>TruthfulQA + TriviaQA + FEVER + Synthetic</div>
  <div><span style="color:#7c3aed;font-weight:600;">Total examples</span><br>545</div>
  <div><span style="color:#7c3aed;font-weight:600;">Epochs</span><br>1</div>
  <div><span style="color:#7c3aed;font-weight:600;">Batch size</span><br>1 (grad accum 16 → effective 16)</div>
  <div><span style="color:#7c3aed;font-weight:600;">Max sequence length</span><br>512 tokens</div>
  <div><span style="color:#7c3aed;font-weight:600;">Hardware</span><br>RTX 3070 8GB</div>
  <div><span style="color:#7c3aed;font-weight:600;">Training time</span><br>~1 hour</div>
  <div><span style="color:#7c3aed;font-weight:600;">Final loss</span><br>0.87</div>
  <div><span style="color:#7c3aed;font-weight:600;">Token accuracy</span><br>82.8%</div>
</div>
<br>
<div style="background:rgba(234,179,8,0.1);border-left:3px solid #eab308;border-radius:6px;padding:10px 14px;margin-top:4px;font-size:0.92rem;color:#fde68a;">
  ⚡ <strong>This is a proof of concept trained on just 545 examples in under an hour on a consumer GPU.</strong>
  The model already surfaces meaningful claim-level uncertainty. Imagine what a dataset 10× or 100× larger,
  with more diverse domains and a longer training run, could achieve — calibrated epistemic transparency at scale.
</div>
""")

        about_block("📈 Results &amp; Evaluation", """
The core evaluation is a before/after comparison. Ask the same question to base Ministral 8B and Penumbra.
The base model responds with uniformly confident prose — no signal about what it knows versus what it is guessing.
Penumbra responds with a structured uncertainty map: each claim broken out with a confidence score,
the reasoning basis behind it, and flagged alternative views where the evidence is contested.
<br><br>
At 82.8% token accuracy with a final loss of 0.87, the model learned to reliably produce well-formed
uncertainty maps. The most meaningful result is qualitative: claims the model should be uncertain about
(contested causal chains, future predictions, incomplete evidence) consistently receive lower confidence
scores than well-established facts — even with minimal training data.
""")

        about_block("💡 Why It Matters", """
<div style="display:flex;flex-direction:column;gap:10px;">
  <div style="display:flex;gap:12px;"><span style="color:#7c3aed;font-size:1.1rem;">→</span><div><strong style="color:#e9d5ff;">For users:</strong> Know what to trust vs verify in any AI response before acting on it.</div></div>
  <div style="display:flex;gap:12px;"><span style="color:#7c3aed;font-size:1.1rem;">→</span><div><strong style="color:#e9d5ff;">For enterprises:</strong> Catch hallucinations before they cause damage in legal, medical, or financial workflows.</div></div>
  <div style="display:flex;gap:12px;"><span style="color:#7c3aed;font-size:1.1rem;">→</span><div><strong style="color:#e9d5ff;">For researchers:</strong> A ground-truth signal for epistemic calibration — does the model's stated confidence match its actual accuracy?</div></div>
  <div style="display:flex;gap:12px;"><span style="color:#7c3aed;font-size:1.1rem;">→</span><div><strong style="color:#e9d5ff;">For AI safety:</strong> Transparency about model uncertainty is a prerequisite for trust. You cannot build reliable human-AI collaboration on a foundation of uniform confidence.</div></div>
</div>
""")

        about_block("🛠️ Tech Stack &amp; Links", """
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;margin-bottom:14px;">
  <div><span style="color:#7c3aed;font-weight:600;">Annotator</span><br>Mistral Large 3 (mistral-large-latest)</div>
  <div><span style="color:#7c3aed;font-weight:600;">Finetuned model</span><br>Ministral 8B + QLoRA</div>
  <div><span style="color:#7c3aed;font-weight:600;">Comparison model</span><br>Ministral 8B base</div>
  <div><span style="color:#7c3aed;font-weight:600;">Experiment tracking</span><br>Weights &amp; Biases</div>
  <div><span style="color:#7c3aed;font-weight:600;">UI</span><br>Streamlit</div>
  <div><span style="color:#7c3aed;font-weight:600;">Visualisation</span><br>Plotly</div>
</div>
<a href="https://huggingface.co/Vaaruni2797/penumbra-ministral-8b" target="_blank"
   style="display:inline-block;background:rgba(124,58,237,0.2);border:1px solid #7c3aed;border-radius:8px;
          padding:6px 14px;color:#c4b5fd;text-decoration:none;font-size:0.88rem;margin-right:8px;">
  🤗 Model on HuggingFace
</a>
<a href="https://huggingface.co/spaces/Vaaruni2797/penumbra" target="_blank"
   style="display:inline-block;background:rgba(124,58,237,0.2);border:1px solid #7c3aed;border-radius:8px;
          padding:6px 14px;color:#c4b5fd;text-decoration:none;font-size:0.88rem;margin-right:8px;">
  🚀 Live Demo
</a>
<a href="https://github.com/Vaaruni2797/penumbra" target="_blank"
   style="display:inline-block;background:rgba(124,58,237,0.2);border:1px solid #7c3aed;border-radius:8px;
          padding:6px 14px;color:#c4b5fd;text-decoration:none;font-size:0.88rem;">
  💻 GitHub
</a>
""")

        return

    # ══════════════════════════════════════════════════════════
    # CHAT PAGE
    # ══════════════════════════════════════════════════════════
    question = st.session_state.question

    # ── Landing ───────────────────────────────────────────────
    if not question:
        st.markdown(
            '<p style="text-align:center;color:#6b7280;font-size:1rem;margin-top:1.5rem;">'
            'Select a demo from the sidebar, click ❓ Ask a Question, or type below.</p>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div style="background:rgba(55,65,81,0.35);border:1px solid #374151;border-radius:12px;padding:24px 20px;height:100%;">'
                '  <p style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6b7280;margin-bottom:14px;">Without Penumbra</p>'
                '  <p style="font-size:1rem;color:#9ca3af;font-style:italic;line-height:1.6;margin-bottom:16px;">'
                '    &ldquo;The 2008 financial crisis was caused by subprime mortgage lending, deregulation, and CDOs...&rdquo;'
                '  </p>'
                '  <p style="font-size:0.88rem;color:#6b7280;margin:0;">Everything sounds equally certain. Nothing tells you what to trust.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div style="background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.4);border-radius:12px;padding:24px 20px;height:100%;">'
                '  <p style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#7c3aed;margin-bottom:14px;">With Penumbra</p>'
                '  <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:16px;">'
                '    <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(234,179,8,0.1);border-radius:6px;padding:6px 10px;">'
                '      <span style="font-size:0.88rem;color:#e9d5ff;">Subprime lending</span>'
                '      <span style="font-size:0.95rem;font-weight:700;color:#eab308;">95%</span>'
                '    </div>'
                '    <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(124,58,237,0.15);border-radius:6px;padding:6px 10px;">'
                '      <span style="font-size:0.88rem;color:#e9d5ff;">Deregulation as cause <em style=\"color:#a78bfa\">(contested)</em></span>'
                '      <span style="font-size:0.95rem;font-weight:700;color:#a855f7;">61%</span>'
                '    </div>'
                '    <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(124,58,237,0.15);border-radius:6px;padding:6px 10px;">'
                '      <span style="font-size:0.88rem;color:#e9d5ff;">CDOs as mechanism</span>'
                '      <span style="font-size:0.95rem;font-weight:700;color:#c084fc;">73%</span>'
                '    </div>'
                '  </div>'
                '  <p style="font-size:0.88rem;color:#a78bfa;font-style:italic;margin:0;">You know exactly what to verify.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div style="background:rgba(55,65,81,0.35);border:1px solid #374151;border-radius:12px;padding:24px 20px;height:100%;">'
                '  <p style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6b7280;margin-bottom:14px;">Why it matters</p>'
                '  <div style="display:flex;flex-direction:column;gap:10px;">'
                '    <div style="display:flex;align-items:flex-start;gap:10px;">'
                '      <span style="color:#7c3aed;font-size:1rem;margin-top:1px;">→</span>'
                '      <span style="font-size:0.92rem;color:#d1d5db;">Catch hallucinations before they cause damage</span>'
                '    </div>'
                '    <div style="display:flex;align-items:flex-start;gap:10px;">'
                '      <span style="color:#7c3aed;font-size:1rem;margin-top:1px;">→</span>'
                '      <span style="font-size:0.92rem;color:#d1d5db;">Know exactly which claims need checking</span>'
                '    </div>'
                '    <div style="display:flex;align-items:flex-start;gap:10px;">'
                '      <span style="color:#7c3aed;font-size:1rem;margin-top:1px;">→</span>'
                '      <span style="font-size:0.92rem;color:#d1d5db;">Calibrated trust in every AI response</span>'
                '    </div>'
                '    <div style="display:flex;align-items:flex-start;gap:10px;">'
                '      <span style="color:#7c3aed;font-size:1rem;margin-top:1px;">→</span>'
                '      <span style="font-size:0.92rem;color:#d1d5db;">Built on Ministral 8B + QLoRA</span>'
                '    </div>'
                '  </div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Results ───────────────────────────────────────────────
    else:
        # Question bubble — always shown when a question is active
        st.markdown(
            f'<div style="text-align:center;margin-bottom:6px;">'
            f'<div class="question-bubble-label">Your question</div>'
            f'<div class="question-bubble">{question}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        predictor = get_predictor(use_local=use_local)

        col_penumbra, col_base = st.columns(2)

        # ── Stage 1: Fetch Penumbra ──
        if not st.session_state.penumbra_ready:
            with col_penumbra:
                st.markdown("### 🌔 Penumbra")
                st.caption("Uncertainty-aware — maps confidence claim by claim")
                with st.spinner("Mapping uncertainty…"):
                    uncertainty_map = predictor.predict(question)
                st.session_state.uncertainty_map = uncertainty_map
                st.session_state.penumbra_ready  = True
            with col_base:
                st.markdown("### 🤖 Base Ministral 8B")
                st.caption("Standard — no uncertainty awareness")
                st.info("⏳ Loading base response…")
            st.rerun()

        # ── Stage 2: Fetch Base ──
        elif not st.session_state.base_ready:
            with col_penumbra:
                st.markdown("### 🌔 Penumbra")
                st.caption("Uncertainty-aware — maps confidence claim by claim")
                render_uncertainty_map(st.session_state.uncertainty_map)
            with col_base:
                st.markdown("### 🤖 Base Ministral 8B")
                st.caption("Standard — no uncertainty awareness")
                with st.spinner("Generating base response…"):
                    base_response = predictor.get_base_response(question)
                st.session_state.base_response = base_response
                st.session_state.base_ready    = True
            st.rerun()

        # ── Stage 3: Show both ──
        else:
            with col_penumbra:
                st.markdown("### 🌔 Penumbra")
                st.caption("Uncertainty-aware — maps confidence claim by claim")
                render_uncertainty_map(st.session_state.uncertainty_map)

            with col_base:
                st.markdown("### 🤖 Base Ministral 8B")
                st.caption("Standard — no uncertainty awareness")
                st.markdown(
                    f'<div class="base-panel">{st.session_state.base_response}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div style="margin-top:8px;font-size:0.85rem;color:#9ca3af;">'
                    '⚠️ Confident tone regardless of actual certainty</div>',
                    unsafe_allow_html=True,
                )

            # Stats strip
            uncertainty_map = st.session_state.uncertainty_map
            if uncertainty_map and "claims" in uncertainty_map:
                st.divider()
                claims = uncertainty_map["claims"]
                confs  = [c["confidence"] for c in claims]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Claims Analyzed",   len(claims))
                c2.metric("Avg Confidence",    f"{sum(confs)/len(confs):.0%}")
                c3.metric("Uncertain Claims",   sum(1 for c in confs if c < 0.6))
                c4.metric("Overall Confidence", f"{uncertainty_map.get('overall_confidence', 0):.0%}")

    # ── Chat input — pinned at bottom ──
    if user_input := st.chat_input("Ask a Question…"):
        st.session_state.question       = user_input
        st.session_state.results_for    = ""
        st.session_state.penumbra_ready = False
        st.session_state.base_ready     = False
        st.session_state.page           = "chat"
        st.rerun()


if __name__ == "__main__":
    main()