"""
Route Resilience Analyzer — ISRO Bharatiya Antariksh Hackathon 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import networkx as nx

from pipeline import GraphBuilder, GraphHealer, GraphAnalyzer
# RoadSegmenter imported lazily inside the pipeline button (needs torch + smp)
from utils.colors import centrality_to_hex, resilience_color
from demo_data import load_demo


# 
# Page config
# 
st.set_page_config(
    page_title="Route Resilience Analyzer | ISRO BAH 2026",
    page_icon="satellite",
    layout="wide",
    initial_sidebar_state="expanded",
)

ISRO_CSS = """
<style>
/*  Base  */
html, body, .stApp { background-color: #030712 !important; color: #f1f5f9; }

/*  Sidebar  */
[data-testid="stSidebar"] {
    background: #0c1220 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/*  Hero banner  */
.hero {
    background: linear-gradient(135deg, #0c1a3a 0%, #0f2460 50%, #0c1a3a 100%);
    border: 1px solid #1e3a8a;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(249,115,22,0.08) 0%, transparent 60%);
}
.hero-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg, #f97316, #fb923c, #fdba74);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero-sub { font-size: 0.95rem; color: #94a3b8; margin: 0; }
.badge {
    display: inline-block;
    background: rgba(249,115,22,0.15); color: #f97316;
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.75rem; font-weight: 600;
    margin-top: 10px;
}

/*  Pipeline stepper  */
.stepper {
    display: flex; gap: 0;
    background: #0c1220; border: 1px solid #1e293b;
    border-radius: 10px; padding: 14px 24px;
    margin-bottom: 24px; align-items: center;
}
.step { display: flex; align-items: center; flex: 1; }
.step-icon {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 700; flex-shrink: 0;
}
.step-icon.done    { background: #16a34a; color: white; }
.step-icon.active  { background: #f97316; color: white; box-shadow: 0 0 12px rgba(249,115,22,0.5); }
.step-icon.pending { background: #1e293b; color: #64748b; }
.step-label { font-size: 0.78rem; margin-left: 8px; }
.step-label .name  { font-weight: 600; color: #e2e8f0; }
.step-label .desc  { color: #64748b; font-size: 0.7rem; }
.step-divider { height: 1px; flex: 0.5; background: #1e293b; margin: 0 8px; }

/*  Metric cards  */
.metric-card {
    background: #0c1220;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #334155; }
.metric-value { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.metric-label { font-size: 0.75rem; color: #64748b; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-icon  { font-size: 1.1rem; margin-bottom: 4px; }

/*  Section headers  */
.section-header {
    font-size: 1rem; font-weight: 700; color: #f97316;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px; margin-bottom: 16px;
    letter-spacing: 0.03em; text-transform: uppercase;
}

/*  Status pill  */
.pill-ok  { display:inline-block; background:#14532d; color:#4ade80; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.pill-err { display:inline-block; background:#7f1d1d; color:#f87171; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.pill-run { display:inline-block; background:#7c2d12; color:#fb923c; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }

/*  Buttons  */
div.stButton > button {
    background: linear-gradient(135deg, #c2410c, #ea580c) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    padding: 10px 24px !important; font-size: 0.9rem !important;
    transition: all 0.2s !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #ea580c, #f97316) !important;
    box-shadow: 0 0 20px rgba(249,115,22,0.4) !important;
    transform: translateY(-1px) !important;
}

/*  Dataframe  */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/*  Tabs  */
[data-testid="stTabs"] { background: transparent; }
button[data-baseweb="tab"] { font-weight: 600 !important; color: #94a3b8 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #f97316 !important; border-bottom-color: #f97316 !important; }

/*  Info/warning boxes  */
.stAlert { border-radius: 8px !important; }

/*  Scrollbar  */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #030712; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
"""
st.markdown(ISRO_CSS, unsafe_allow_html=True)


# 
# Helper functions
# 

def _overlay_skeleton(img: np.ndarray, skel: np.ndarray | None) -> np.ndarray:
    if skel is None:
        return img
    out = img.copy()
    out[skel > 127] = [0, 230, 120]
    return out


def _stepper_html(stage: int) -> str:
    steps = [
        ("1", "Upload", "Satellite image"),
        ("2", "Extract", "Road mask + skeleton"),
        ("3", "Analyze", "Graph + centrality"),
        ("4", "Simulate", "Resilience score"),
    ]
    html = '<div class="stepper">'
    for i, (num, name, desc) in enumerate(steps):
        if i < stage:
            cls = "done"
            icon = ""
        elif i == stage:
            cls = "active"
            icon = num
        else:
            cls = "pending"
            icon = num

        html += f"""
        <div class="step">
            <div class="step-icon {cls}">{icon}</div>
            <div class="step-label">
                <div class="name">{name}</div>
                <div class="desc">{desc}</div>
            </div>
        </div>"""
        if i < len(steps) - 1:
            html += '<div class="step-divider"></div>'

    html += "</div>"
    return html


def _metric_card(value, label, color):
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def _build_graph_figure(
    G: nx.Graph,
    highlight_nodes: list | None = None,
    highlight_path: list | None = None,
) -> go.Figure:
    if len(G) == 0:
        return go.Figure()

    nodes      = list(G.nodes())
    failed_set = set(highlight_nodes or [])
    path_set   = set(highlight_path  or [])

    path_edges: set = set()
    if highlight_path and len(highlight_path) > 1:
        for i in range(len(highlight_path) - 1):
            path_edges.add((highlight_path[i], highlight_path[i + 1]))
            path_edges.add((highlight_path[i + 1], highlight_path[i]))

    edge_x, edge_y = [], []
    path_ex, path_ey = [], []

    for u, v in G.edges():
        x0, y0 = G.nodes[u].get("x", u[1]), -G.nodes[u].get("y", u[0])
        x1, y1 = G.nodes[v].get("x", v[1]), -G.nodes[v].get("y", v[0])
        if (u, v) in path_edges:
            path_ex.extend([x0, x1, None])
            path_ey.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    node_x     = [G.nodes[n].get("x", n[1]) for n in nodes]
    node_y     = [-G.nodes[n].get("y", n[0]) for n in nodes]
    node_colors = []
    node_sizes  = []
    for n in nodes:
        bet = G.nodes[n].get("bottleneck", 0)
        if n in failed_set:
            node_colors.append("#ef4444")
            node_sizes.append(14)
        elif n in path_set:
            node_colors.append("#22c55e")
            node_sizes.append(12)
        else:
            node_colors.append(centrality_to_hex(bet))
            node_sizes.append(6 + bet * 12)

    node_text = [
        f"<b>Node {n}</b><br>"
        f"Betweenness: {G.nodes[n].get('betweenness', 0):.4f}<br>"
        f"Closeness: {G.nodes[n].get('closeness', 0):.4f}<br>"
        f"Degree: {G.degree(n)}"
        for n in nodes
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#1e3a5f", width=1.5),
        hoverinfo="none", name="Road Segment",
    ))
    if path_ex:
        fig.add_trace(go.Scatter(
            x=path_ex, y=path_ey, mode="lines",
            line=dict(color="#22c55e", width=4),
            hoverinfo="none", name="Emergency Route",
        ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(
            size=node_sizes, color=node_colors,
            line=dict(color="#030712", width=0.8),
        ),
        text=node_text, hoverinfo="text",
        hovertemplate="%{text}<extra></extra>",
        name="Intersection",
    ))
    fig.update_layout(
        paper_bgcolor="#030712", plot_bgcolor="#0c1220",
        font=dict(color="#94a3b8", family="monospace"),
        showlegend=True,
        legend=dict(bgcolor="rgba(12,18,32,0.8)", bordercolor="#1e293b",
                    borderwidth=1, font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=500,
        hoverlabel=dict(bgcolor="#0c1220", bordercolor="#1e293b",
                        font=dict(color="#f1f5f9", size=12)),
    )
    return fig


def _resilience_gauge(score: float) -> go.Figure:
    color = resilience_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 100, "valueformat": ".1f"},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
            "bar":       {"color": color, "thickness": 0.3},
            "bgcolor":   "#0c1220",
            "bordercolor": "#1e293b",
            "steps": [
                {"range": [0, 50],   "color": "rgba(239,68,68,0.15)"},
                {"range": [50, 75],  "color": "rgba(251,146,60,0.15)"},
                {"range": [75, 100], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 2},
                "thickness": 0.75, "value": 50,
            },
        },
        title={"text": "Network Resilience Score", "font": {"color": "#94a3b8", "size": 13}},
    ))
    fig.update_layout(
        paper_bgcolor="#030712", font=dict(color="#94a3b8"),
        margin=dict(l=20, r=20, t=30, b=10), height=240,
    )
    return fig


def _plot_resilience_curve(G: nx.Graph, analyzer: GraphAnalyzer):
    top_nodes = [n for n, _ in analyzer.top_bottlenecks(10)]
    scores = []
    for k in range(1, len(top_nodes) + 1):
        r = analyzer.simulate_failure(top_nodes[:k], find_alternates=False)
        scores.append(round(r.resilience_score, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(scores) + 1)), y=scores,
        mode="lines+markers", name="Resilience (%)",
        line=dict(color="#f97316", width=2.5),
        marker=dict(size=7, color=scores,
                    colorscale=[[0,"#ef4444"],[0.5,"#f97316"],[1,"#22c55e"]],
                    showscale=False),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
        hovertemplate="Nodes failed: %{x}<br>Resilience: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=50, line_dash="dash", line_color="#ef4444", line_width=1.5,
        annotation_text="Critical Threshold",
        annotation_font_color="#ef4444", annotation_font_size=11,
    )
    fig.update_layout(
        xaxis_title="Critical nodes removed (most → least important)",
        yaxis_title="Resilience Score (%)",
        paper_bgcolor="#030712", plot_bgcolor="#0c1220",
        font=dict(color="#94a3b8"),
        yaxis=dict(range=[0, 105], gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b", dtick=1),
        height=300, margin=dict(l=50, r=20, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_folium_map(G, base_layer, center, bbox, image_shape, failed_nodes):
    from pipeline.graph_builder import pixel_to_latlon

    tile_cfg = {
        "OpenStreetMap":           ("OpenStreetMap",       "© OpenStreetMap contributors"),
        "ISRO Bhuvan (Satellite)": ("CartoDB dark_matter", "ISRO NRSC Bhuvan"),
        "CartoDB Dark":            ("CartoDB dark_matter", "© CartoDB"),
    }
    tiles, attr = tile_cfg.get(base_layer, ("OpenStreetMap", "OSM"))
    fmap = folium.Map(location=center, zoom_start=13, tiles=tiles, attr=attr)

    failed_set = set(failed_nodes)
    for u, v in G.edges():
        lat1, lon1 = pixel_to_latlon(G.nodes[u]["y"], G.nodes[u]["x"], image_shape, bbox)
        lat2, lon2 = pixel_to_latlon(G.nodes[v]["y"], G.nodes[v]["x"], image_shape, bbox)
        col = "#ef4444" if (u in failed_set or v in failed_set) else "#3b82f6"
        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color=col, weight=2.5, opacity=0.85).add_to(fmap)

    for n in G.nodes():
        lat, lon = pixel_to_latlon(G.nodes[n]["y"], G.nodes[n]["x"], image_shape, bbox)
        bet = G.nodes[n].get("betweenness", 0)
        col = "#ef4444" if n in failed_set else centrality_to_hex(bet)
        folium.CircleMarker(
            location=(lat, lon), radius=max(4, 4 + bet * 14),
            color=col, fill=True, fill_color=col, fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>Node {n}</b><br>"
                f"Betweenness: {bet:.4f}<br>"
                f"Degree: {G.degree(n)}<br>"
                f"{'[FAILED]' if n in failed_set else '[Active]'}",
                max_width=180,
            ),
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap


def _export_csv(G, analyzer):
    rows = [{
        "node":             str(n),
        "y_pixel":          G.nodes[n].get("y"),
        "x_pixel":          G.nodes[n].get("x"),
        "degree":           G.degree(n),
        "betweenness":      round(G.nodes[n].get("betweenness", 0), 6),
        "closeness":        round(G.nodes[n].get("closeness", 0), 6),
        "degree_centrality": round(G.nodes[n].get("degree_c", 0), 6),
        "bottleneck_index": round(G.nodes[n].get("bottleneck", 0), 6),
    } for n in G.nodes()]
    return pd.DataFrame(rows).sort_values("bottleneck_index", ascending=False).to_csv(index=False).encode()


def _generate_pdf(G, analyzer, cfi, report) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib import colors as rc
        from reportlab.lib.units import cm
    except ImportError:
        return b"reportlab not installed - run: pip install reportlab"

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    orange = rc.HexColor("#f97316")
    navy   = rc.HexColor("#0f2460")
    story  = []

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  textColor=orange, fontSize=20, spaceAfter=4)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  textColor=rc.HexColor("#64748b"), fontSize=10)

    story.append(Paragraph("ROUTE RESILIENCE ANALYZER", title_style))
    story.append(Paragraph("ISRO Bharatiya Antariksh Hackathon 2026 — Disaster Preparedness Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=orange, spaceAfter=12))

    story.append(Paragraph("Network Summary", styles["Heading2"]))
    cfi_val = cfi["cfi"] if cfi else "N/A"
    risk    = "HIGH" if isinstance(cfi_val, float) and cfi_val > 60 else \
              "MEDIUM" if isinstance(cfi_val, float) and cfi_val > 30 else "LOW"

    tdata = [
        ["Metric", "Value"],
        ["Total Intersections (Nodes)", str(G.number_of_nodes())],
        ["Total Road Segments (Edges)", str(G.number_of_edges())],
        ["Connected Zones", str(nx.number_connected_components(G))],
        ["Cascading Failure Index (CFI)", f"{cfi_val}  — Risk: {risk}"],
        ["Resilience Score (post-simulation)",
         f"{report.resilience_score:.1f}%" if report else "Not simulated"],
    ]
    t = Table(tdata, colWidths=[9*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), navy),
        ("TEXTCOLOR",   (0, 0), (-1, 0), rc.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",        (0, 0), (-1, -1), 0.5, rc.HexColor("#1e293b")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rc.white, rc.HexColor("#f8fafc")]),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("PADDING",     (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Top 5 Critical Bottleneck Nodes", styles["Heading2"]))
    btn  = [["Rank", "Node ID", "Bottleneck Index", "Betweenness", "Degree"]]
    for i, (n, score) in enumerate(analyzer.top_bottlenecks(5)):
        btn.append([
            str(i + 1), str(n), f"{score:.4f}",
            f"{G.nodes[n].get('betweenness', 0):.4f}",
            str(G.degree(n)),
        ])
    t2 = Table(btn, colWidths=[1.5*cm, 4.5*cm, 4*cm, 4*cm, 2*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR",  (0, 0), (-1, 0), rc.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, rc.HexColor("#1e293b")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rc.white, rc.HexColor("#fff7ed")]),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Risk Assessment", styles["Heading2"]))
    if cfi:
        story.append(Paragraph(
            f"The Cascading Failure Index of <b>{cfi['cfi']}</b> indicates a <b>{risk} risk</b> network. "
            f"Removing the 5 most critical intersections would create {cfi['fragmentation']} additional "
            f"isolated zone(s) and strand {cfi['isolated_nodes']} intersection(s) entirely.",
            styles["Normal"],
        ))

    doc.build(story)
    return buf.getvalue()


# 
# Sidebar
# 
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 8px;">
        <div style="font-weight:800; font-size:1rem; color:#f97316;">Route Resilience</div>
        <div style="font-size:0.7rem; color:#475569;">ISRO BAH 2026 · PS-4</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader(
        "Upload Satellite Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Any RGB satellite or aerial image works.",
    )

    st.markdown("### Parameters")
    seg_threshold = st.slider("Road Detection Level (Sadak Pahchaan Sthar)",  0.20, 0.80, 0.45, 0.05,
                              help="Kitni clearly sadak detect ho — low = zyada sadkein, high = sirf pakki sadkein")
    max_gap       = st.slider("Broken Road Fix - px (Tuti Sadak Jod)",        5,    60,   25,   5,
                              help="Kitne pixel ki tuti sadak ko joda jaye — jaise tree ya building se dhaki sadak")
    top_k         = st.slider("Key Choke Points (Mukhya Avrodh Bindu)",       3,    20,   10,   1,
                              help="Kitne sabse important/busy crossings highlight karne hain")

    st.markdown("### Map Layer")
    base_layer = st.selectbox("Tile Source",
        ["OpenStreetMap", "ISRO Bhuvan (Satellite)", "CartoDB Dark"])

    st.markdown("### Model Weights")
    use_weights  = st.checkbox("Use fine-tuned weights (optional)")
    weights_path = ""
    if use_weights:
        weights_path = st.text_input("Path to .pth file", "models/road_seg.pth")

    st.divider()
    st.markdown(
        '<div style="text-align:center; font-size:0.7rem; color:#334155;">'
        'DeepLabV3+ · NetworkX · Streamlit<br>'
        'Built for ISRO BAH 2026</div>',
        unsafe_allow_html=True,
    )


# 
# Session state
# 
for _k in ["image", "mask", "skeleton", "graph", "analyzer", "cfi", "report", "stage", "seg_mode", "_uploaded_name"]:
    if _k not in st.session_state:
        st.session_state[_k] = None
if st.session_state["stage"] is None:
    st.session_state["stage"] = 0


# 
# Hero banner
# 
st.markdown("""
<div class="hero">
    <p class="hero-title">Route Resilience Analyzer</p>
    <p class="hero-sub">
        AI-powered road network extraction · Bottleneck detection · Disaster resilience simulation
    </p>
    <span class="badge">ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 4</span>
</div>
""", unsafe_allow_html=True)

# Pipeline stepper
st.markdown(_stepper_html(st.session_state["stage"]), unsafe_allow_html=True)


# 
# Demo mode launch
# 
if uploaded is None and st.session_state["graph"] is None:
    st.markdown("---")
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            '<div style="text-align:center; color:#64748b; margin-bottom:16px;">'
            '<div style="font-size:1rem; font-weight:600; color:#94a3b8; margin-bottom:4px;">'
            'No image uploaded yet</div>'
            '<div style="font-size:0.85rem;">Upload a satellite image from the sidebar,<br>'
            'or try the built-in Indian city demo.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Launch Demo — Indian City Grid", use_container_width=True):
            with st.spinner("Building demo dataset…"):
                demo = load_demo()
            st.session_state.update(demo)
            st.session_state["stage"] = 3
            st.rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.7rem; font-weight:700; color:#f97316; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Road Extraction</div>
            <div style="font-size:0.85rem; color:#94a3b8;">DeepLabV3+ with ResNet50 extracts roads even under cloud, shadow, and tree occlusion.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.7rem; font-weight:700; color:#f97316; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Bottleneck Detection</div>
            <div style="font-size:0.85rem; color:#94a3b8;">Betweenness centrality pinpoints the intersections whose failure causes maximum damage.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.7rem; font-weight:700; color:#f97316; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Resilience Simulation</div>
            <div style="font-size:0.85rem; color:#94a3b8;">Remove nodes and watch the score drop live — with emergency alternate routes highlighted.</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


#
# Load real image (if uploaded or changed)
#
if uploaded is not None:
    if st.session_state["_uploaded_name"] != uploaded.name:
        # New file — reset all pipeline state
        for _k in ["image", "mask", "skeleton", "graph", "analyzer", "cfi", "report", "seg_mode"]:
            st.session_state[_k] = None
        st.session_state["stage"] = 0
        st.session_state["_uploaded_name"] = uploaded.name

        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["image"] = image_rgb
        st.session_state["stage"] = 1
        st.rerun()


# 
# Tabs
# 
tab_extract, tab_graph, tab_sim, tab_map, tab_report = st.tabs([
    "Road Extraction",
    "Network Analysis",
    "Disruption Simulation",
    "Live Map",
    "Report",
])


# 
# TAB 1 — Road Extraction
# 
with tab_extract:
    st.markdown('<div class="section-header">Road Extraction Pipeline</div>',
                unsafe_allow_html=True)

    img_col, ctl_col = st.columns([3, 1])
    with img_col:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original Image**")
            if st.session_state["image"] is not None:
                st.image(st.session_state["image"], use_container_width=True)

        with c2:
            st.markdown("**Road Mask**")
            if st.session_state["mask"] is not None:
                st.image(st.session_state["mask"], use_container_width=True, clamp=True)
            else:
                st.markdown(
                    '<div style="height:200px; background:#0c1220; border:1px dashed #1e293b; '
                    'border-radius:8px; display:flex; align-items:center; justify-content:center; '
                    'color:#334155; font-size:0.8rem;">Run pipeline →</div>',
                    unsafe_allow_html=True,
                )

        with c3:
            st.markdown("**Skeleton Overlay**")
            if st.session_state["skeleton"] is not None:
                skel_vis = _overlay_skeleton(st.session_state["image"], st.session_state["skeleton"])
                st.image(skel_vis, use_container_width=True)
            else:
                st.markdown(
                    '<div style="height:200px; background:#0c1220; border:1px dashed #1e293b; '
                    'border-radius:8px; display:flex; align-items:center; justify-content:center; '
                    'color:#334155; font-size:0.8rem;">Run pipeline →</div>',
                    unsafe_allow_html=True,
                )

    with ctl_col:
        st.markdown("**Pipeline Control**")
        st.caption("Runs segmentation → skeletonize → graph → centrality")

        if st.session_state["graph"] is not None:
            G = st.session_state["graph"]
            st.markdown('<span class="pill-ok">Pipeline complete</span>', unsafe_allow_html=True)
            st.markdown(f"**{G.number_of_nodes()}** intersections")
            st.markdown(f"**{G.number_of_edges()}** road segments")
            st.markdown(f"**{nx.number_connected_components(G)}** zones")
            if st.session_state.get("seg_mode"):
                st.caption(f"Mode: {st.session_state['seg_mode']}")

        st.markdown("")
        if st.button(" Run Full Pipeline", use_container_width=True):
            if st.session_state["image"] is None:
                st.warning("Upload an image first.")
            else:
                with st.spinner("Extracting roads..."):
                    from pipeline.segmentation import RoadSegmenter
                    seg  = RoadSegmenter(weights_path=weights_path if use_weights else None)
                    if seg._has_dl and seg.trained:
                        mode = "DeepLabV3+ (fine-tuned weights)"
                    elif seg._has_dl:
                        mode = "Heuristic (DL ready — train model to activate)"
                    else:
                        mode = "Heuristic (install torch for DL mode)"
                    mask = seg.segment(st.session_state["image"], threshold=seg_threshold)
                    st.session_state["mask"] = mask
                    st.session_state["seg_mode"] = mode

                with st.spinner("Skeletonising…"):
                    builder = GraphBuilder()
                    skel, G = builder.build(mask)
                    st.session_state["skeleton"] = skel

                with st.spinner("Healing gaps…"):
                    G = GraphHealer(max_gap_px=max_gap).heal(G)

                with st.spinner("Computing centrality…"):
                    analyzer = GraphAnalyzer(G)
                    analyzer.compute_centrality()
                    cfi = analyzer.cascading_failure_index(top_k=5)
                    st.session_state.update({"graph": G, "analyzer": analyzer, "cfi": cfi})
                    st.session_state["stage"] = 2

                st.rerun()

    #  Network summary metrics 
    if st.session_state["graph"] is not None:
        G   = st.session_state["graph"]
        cfi = st.session_state["cfi"]
        st.markdown("---")
        st.markdown('<div class="section-header">Network Summary</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(_metric_card(G.number_of_nodes(), "Intersections", "#38bdf8"),
                        unsafe_allow_html=True)
        with m2:
            st.markdown(_metric_card(G.number_of_edges(), "Road Segments", "#38bdf8"),
                        unsafe_allow_html=True)
        with m3:
            comps = nx.number_connected_components(G)
            col   = "#22c55e" if comps == 1 else "#f97316"
            st.markdown(_metric_card(comps, "Connected Zones", col), unsafe_allow_html=True)
        with m4:
            if cfi:
                c = "#ef4444" if cfi["cfi"] > 60 else "#f97316" if cfi["cfi"] > 30 else "#22c55e"
                st.markdown(_metric_card(cfi["cfi"], "Cascading Failure Index", c),
                            unsafe_allow_html=True)


# 
# TAB 2 — Network Analysis
# 
with tab_graph:
    if st.session_state["graph"] is None:
        st.warning("Run the pipeline first (Road Extraction tab).")
    else:
        G        = st.session_state["graph"]
        analyzer = st.session_state["analyzer"]

        left, right = st.columns([2, 1])

        with left:
            st.markdown('<div class="section-header">Interactive Road Network Graph</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "**Node colour** → red = high bottleneck centrality, green = low centrality &nbsp;|&nbsp; "
                "**Node size** → proportional to criticality"
            )
            fig = _build_graph_figure(G)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        with right:
            st.markdown('<div class="section-header">Top Critical Nodes</div>',
                        unsafe_allow_html=True)
            top = analyzer.top_bottlenecks(k=min(top_k, 15))
            for i, (n, score) in enumerate(top[:8]):
                pct = int(score * 100)
                bar_color = "#ef4444" if pct > 60 else "#f97316" if pct > 30 else "#22c55e"
                st.markdown(f"""
                <div style="margin-bottom:8px; padding:8px 10px; background:#0c1220;
                            border-radius:6px; border-left:3px solid {bar_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:0.78rem; color:#94a3b8;">
                            <b style="color:#e2e8f0;">#{i+1}</b> &nbsp; Node {n}
                        </div>
                        <div style="font-size:0.78rem; font-weight:700; color:{bar_color};">{score:.4f}</div>
                    </div>
                    <div style="height:4px; background:#1e293b; border-radius:2px; margin-top:5px;">
                        <div style="width:{pct}%; height:100%; background:{bar_color}; border-radius:2px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # Full table
        st.markdown('<div class="section-header">Full Bottleneck Table</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame([{
            "Rank": i + 1,
            "Node":  str(n),
            "Bottleneck Index": round(score, 4),
            "Betweenness":      round(G.nodes[n]["betweenness"], 4),
            "Closeness":        round(G.nodes[n]["closeness"], 4),
            "Degree":           G.degree(n),
        } for i, (n, score) in enumerate(top)])
        st.dataframe(df, use_container_width=True, hide_index=True)


# 
# TAB 3 — Disruption Simulation
# 
with tab_sim:
    if st.session_state["graph"] is None:
        st.warning("Run the pipeline first.")
    else:
        G        = st.session_state["graph"]
        analyzer = st.session_state["analyzer"]

        st.markdown('<div class="section-header">Infrastructure Failure Simulation</div>',
                    unsafe_allow_html=True)
        st.markdown("Select one or more intersections to fail — simulating flood, landslide, or accident. "
                    "The network recomputes instantly.")

        top_nodes   = [n for n, _ in analyzer.top_bottlenecks(20)]
        node_labels = [f"Node {n}  [rank #{i+1}]" for i, n in enumerate(top_nodes)]

        fc1, fc2 = st.columns([3, 1])
        with fc1:
            sel_labels = st.multiselect(
                "Failed nodes:", node_labels, default=node_labels[:1],
                help="Top-ranked = most critical. Removing these causes maximum damage."
            )
        with fc2:
            n_cascade = st.number_input("Auto-cascade top-N", 1, 10, 1, 1)
            if st.button("Cascade Failure", use_container_width=True):
                sel_labels = node_labels[:n_cascade]
                st.session_state["stage"] = 3

        failed_nodes = [top_nodes[node_labels.index(l)] for l in sel_labels if l in node_labels]

        if failed_nodes:
            with st.spinner("Simulating…"):
                sim = analyzer.simulate_failure(failed_nodes, find_alternates=True)
                st.session_state["report"] = sim
                st.session_state["stage"]  = 3

            score    = sim.resilience_score
            eff_drop = (1 - sim.post_failure_efficiency / max(sim.baseline_efficiency, 1e-9)) * 100
            frag     = sim.n_components_after - sim.n_components_before

            # Top row: gauge + 3 metrics
            g1, g2 = st.columns([1, 2])
            with g1:
                st.plotly_chart(_resilience_gauge(score), use_container_width=True)
            with g2:
                mm1, mm2, mm3 = st.columns(3)
                with mm1:
                    st.markdown(_metric_card(
                        f"{eff_drop:.1f}%", "Efficiency Drop", "#ef4444"),
                        unsafe_allow_html=True)
                with mm2:
                    c = "#ef4444" if frag > 2 else "#f97316" if frag > 0 else "#22c55e"
                    st.markdown(_metric_card(f"+{frag}", "New Isolated Zones", c),
                                unsafe_allow_html=True)
                with mm3:
                    st.markdown(_metric_card(
                        sim.n_isolated_nodes, "Stranded Nodes", "#ef4444"),
                        unsafe_allow_html=True)

                # Plain-language verdict
                if score >= 75:
                    st.success(f" Network is resilient ({score:.0f}%). Disruption manageable.")
                elif score >= 50:
                    st.warning(f" Moderate damage ({score:.0f}%). Some routes severely compromised.")
                else:
                    st.error(f" Critical failure ({score:.0f}%). Major connectivity lost — immediate action required.")

            st.markdown("---")
            net_col, route_col = st.columns([3, 2])

            with net_col:
                st.markdown("**Network after failure** — red nodes = failed")
                fig_fail = _build_graph_figure(G, highlight_nodes=failed_nodes)
                st.plotly_chart(fig_fail, use_container_width=True)

            with route_col:
                st.markdown("**Emergency Alternate Routes**")
                if sim.alternate_routes:
                    for (src, tgt), path in sim.alternate_routes.items():
                        st.markdown(
                            f'<div style="background:#0c1220; border:1px solid #1e293b; '
                            f'border-left:3px solid #22c55e; border-radius:6px; padding:10px 14px; margin-bottom:8px;">'
                            f'<div style="font-size:0.8rem; color:#22c55e; font-weight:700;">Alternate route found</div>'
                            f'<div style="font-size:0.75rem; color:#94a3b8; margin-top:3px;">'
                            f'{src} → {tgt} &nbsp;·&nbsp; {len(path)} hops</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    # Show first alternate on graph
                    first_path = next(iter(sim.alternate_routes.values()))
                    st.markdown("**Best route (highlighted green):**")
                    fig_route = _build_graph_figure(
                        G, highlight_nodes=failed_nodes, highlight_path=first_path
                    )
                    st.plotly_chart(fig_route, use_container_width=True)
                else:
                    st.error(" No viable alternate routes exist. Network critically fragmented.")

            st.markdown("---")
            st.markdown('<div class="section-header">Resilience Degradation Curve</div>',
                        unsafe_allow_html=True)
            st.caption("Shows how resilience drops as more top-ranked nodes are removed sequentially.")
            _plot_resilience_curve(G, analyzer)


# 
# TAB 4 — Live Map
# 
with tab_map:
    if st.session_state["graph"] is None:
        st.warning("Run the pipeline first.")
    else:
        G = st.session_state["graph"]

        st.markdown('<div class="section-header">Geospatial Network Map</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Nodes are colour-coded by centrality. Red = failed (from simulation). "
            "Click any node for details."
        )

        map_left, map_right = st.columns([4, 1])
        with map_right:
            lat_c    = st.text_input("Centre Lat", "20.5937")
            lon_c    = st.text_input("Centre Lon", "78.9629")
            use_bbox = st.checkbox("Custom bounding box")
            bbox     = None
            if use_bbox:
                st.caption("Pixel 0,0 = top-left corner of your image")
                lat_min = st.number_input("Lat min", value=20.0)
                lat_max = st.number_input("Lat max", value=21.0)
                lon_min = st.number_input("Lon min", value=78.0)
                lon_max = st.number_input("Lon max", value=80.0)
                bbox    = (lat_min, lat_max, lon_min, lon_max)

            st.markdown("**Legend**")
            st.markdown(
                '<div style="font-size:0.75rem; line-height:1.8;">'
                '<span style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:50%;margin-right:5px;"></span> Failed / High centrality<br>'
                '<span style="display:inline-block;width:10px;height:10px;background:#f97316;border-radius:50%;margin-right:5px;"></span> Medium centrality<br>'
                '<span style="display:inline-block;width:10px;height:10px;background:#22c55e;border-radius:50%;margin-right:5px;"></span> Low centrality<br>'
                '<span style="display:inline-block;width:30px;height:3px;background:#3b82f6;margin-right:5px;vertical-align:middle;"></span> Active road<br>'
                '<span style="display:inline-block;width:30px;height:3px;background:#ef4444;margin-right:5px;vertical-align:middle;"></span> Affected road</div>',
                unsafe_allow_html=True,
            )

        with map_left:
            failed = st.session_state["report"].failed_nodes if st.session_state["report"] else []
            fmap   = _build_folium_map(
                G, base_layer=base_layer,
                center=(float(lat_c), float(lon_c)),
                bbox=bbox,
                image_shape=st.session_state["image"].shape,
                failed_nodes=failed,
            )
            st_folium(fmap, width=700, height=560)


# 
# TAB 5 — Report
# 
with tab_report:
    if st.session_state["graph"] is None:
        st.warning("Run the pipeline first.")
    else:
        G        = st.session_state["graph"]
        analyzer = st.session_state["analyzer"]
        cfi      = st.session_state["cfi"]
        report   = st.session_state["report"]

        st.markdown('<div class="section-header">Export Disaster Preparedness Report</div>',
                    unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Network statistics**")
            st.json({
                "nodes":  G.number_of_nodes(),
                "edges":  G.number_of_edges(),
                "connected_components": nx.number_connected_components(G),
                "cascading_failure_index": cfi,
                "resilience_score": round(report.resilience_score, 2) if report else None,
                "new_isolated_zones": (report.n_components_after - report.n_components_before)
                                       if report else 0,
                "top_5_bottlenecks": [str(n) for n, _ in analyzer.top_bottlenecks(5)],
            })
        with r2:
            st.markdown("**Download options**")
            st.markdown("")

            csv_data  = _export_csv(G, analyzer)
            pdf_bytes = _generate_pdf(G, analyzer, cfi, report)

            st.download_button(
                "  Node Analysis (CSV)",
                data=csv_data,
                file_name="road_network_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("")
            st.download_button(
                "  Disaster Preparedness Report (PDF)",
                data=pdf_bytes,
                file_name="route_resilience_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.markdown("")
            st.info(
                " Run the Disruption Simulation tab first so the PDF includes "
                "resilience scores and alternate route data."
            )
