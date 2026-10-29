import sqlite3
import io
import struct
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

# ReportLab imports for direct in-memory PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect, String

st.set_page_config(page_title="Geological Petrography QC & Analysis", layout="wide")

# ---------------------------------------------------------
# CUSTOM LABORATORY STYLING (CSS) - LIGHT GREY & BLACK TEXT
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card styling - Light Grey background with Black text */
    .metric-card {
        background-color: #f0f2f6 !important;
        border: 1px solid #d0d7de !important;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #111111 !important;
    }
    .metric-card b, .metric-card span, .metric-card p {
        color: #111111 !important;
    }

    /* Native Streamlit metric boxes styling */
    [data-testid="stMetric"] {
        background-color: #f0f2f6 !important;
        border: 1px solid #d0d7de !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #222222 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] div {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .pill-green { background-color: #d1e7dd; color: #0f5132; }
    .pill-amber { background-color: #fff3cd; color: #664d03; }
    .pill-red { background-color: #f8d7da; color: #842029; }
    .pill-blue { background-color: #cfe2ff; color: #084298; }
    
    /* Top Context Ribbon */
    .context-ribbon {
        background: #1b365d;
        color: #ffffff;
        padding: 10px 18px;
        border-radius: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Geological Petrography Analysis Platform")

SCALE_UM_PER_PX = 1.82  # 1.82 microns per pixel
DB_FILE = "petrography_data.db"

# ---------------------------------------------------------
# 0. DATABASE PERSISTENCE & AUTO-MIGRATION MODULE
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY,
            job_number TEXT,
            sample_order TEXT,
            sample_type TEXT,
            operator TEXT,
            total_points INTEGER,
            valid_points INTEGER,
            bounding_area_mm2 REAL,
            linear_traverse_mm REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    try:
        cursor.execute("ALTER TABLE samples ADD COLUMN lithology_code TEXT;")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS modal_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT,
            phase_name TEXT,
            point_count INTEGER,
            modal_percent REAL,
            std_error REAL,
            abundance TEXT,
            en933_class TEXT,
            carbonation TEXT,
            liberation TEXT,
            FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
        )
    """)
    
    try:
        cursor.execute("ALTER TABLE modal_results ADD COLUMN density_gcm3 REAL;")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE modal_results ADD COLUMN wt_percent REAL;")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

init_db()

def _clean_numeric_val(val, to_float=False):
    """Safely converts potential byte-blobs or malformed values from SQLite to clean numbers."""
    if pd.isna(val) or val is None:
        return 0.0 if to_float else 0
    if isinstance(val, (bytes, bytearray)):
        try:
            fmt = '<d' if to_float else '<q'
            return struct.unpack(fmt, val[:8])[0]
        except Exception:
            try:
                fmt = '<f' if to_float else '<i'
                return struct.unpack(fmt, val[:4])[0]
            except Exception:
                return 0.0 if to_float else 0
    try:
        return float(val) if to_float else int(val)
    except (ValueError, TypeError):
        return 0.0 if to_float else 0

def save_sample_to_db(sample_meta, df_summary):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO samples 
        (sample_id, job_number, sample_order, sample_type, operator, total_points, valid_points, bounding_area_mm2, linear_traverse_mm, lithology_code)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(sample_meta["sample_id"]),
        str(sample_meta["job_number"]),
        str(sample_meta["sample_order"]),
        str(sample_meta["sample_type"]),
        str(sample_meta["operator"]),
        int(_clean_numeric_val(sample_meta["total_points"], to_float=False)),
        int(_clean_numeric_val(sample_meta["valid_points"], to_float=False)),
        float(_clean_numeric_val(sample_meta["area_mm2"], to_float=True)),
        float(_clean_numeric_val(sample_meta["linear_traverse_mm"], to_float=True)),
        str(sample_meta.get("lithology_code", ""))
    ))
    
    cursor.execute("DELETE FROM modal_results WHERE sample_id = ?", (str(sample_meta["sample_id"]),))
    
    for _, row in df_summary.iterrows():
        p_name = str(row.get("Standardized_Phase", row.get("phase_name", "")))
        p_count = int(_clean_numeric_val(row.get("Point_Count", row.get("point_count", 0)), to_float=False))
        m_pct = float(_clean_numeric_val(row.get("Modal_Percent (%)", row.get("modal_percent", 0.0)), to_float=True))
        s_err = float(_clean_numeric_val(row.get("Std_Error (%)", row.get("std_error", 0.0)), to_float=True))
        abun = row.get("Abundance", row.get("abundance", None))
        
        raw_dens = row.get("Density (g/cm³)", row.get("density_gcm3", None))
        d_val = float(_clean_numeric_val(raw_dens, to_float=True)) if pd.notnull(raw_dens) else None
        
        raw_wt = row.get("Weight_Percent (%)", row.get("wt_percent", None))
        wt_val = float(_clean_numeric_val(raw_wt, to_float=True)) if pd.notnull(raw_wt) else None

        cursor.execute("""
            INSERT INTO modal_results 
            (sample_id, phase_name, point_count, modal_percent, std_error, abundance, density_gcm3, wt_percent, en933_class, carbonation, liberation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(sample_meta["sample_id"]),
            p_name,
            p_count,
            m_pct,
            s_err,
            str(abun) if abun is not None else None,
            d_val,
            wt_val,
            str(row.get("EN933_Class", row.get("en933_class", ""))) or None,
            str(row.get("Carbonation", row.get("carbonation", ""))) or None,
            str(row.get("Liberation", row.get("liberation", ""))) or None
        ))
        
    conn.commit()
    conn.close()

def load_all_samples():
    conn = sqlite3.connect(DB_FILE)
    df_samples = pd.read_sql_query("SELECT * FROM samples ORDER BY created_at DESC", conn)
    conn.close()
    
    if not df_samples.empty:
        df_samples["total_points"] = df_samples["total_points"].apply(lambda v: _clean_numeric_val(v, to_float=False))
        df_samples["valid_points"] = df_samples["valid_points"].apply(lambda v: _clean_numeric_val(v, to_float=False))
        df_samples["bounding_area_mm2"] = df_samples["bounding_area_mm2"].apply(lambda v: _clean_numeric_val(v, to_float=True))
        df_samples["linear_traverse_mm"] = df_samples["linear_traverse_mm"].apply(lambda v: _clean_numeric_val(v, to_float=True))
        
        for col in ["sample_id", "job_number", "sample_order", "sample_type", "operator", "lithology_code", "created_at"]:
            if col in df_samples.columns:
                df_samples[col] = df_samples[col].astype(str)
                
    return df_samples

def load_sample_details(sample_id):
    conn = sqlite3.connect(DB_FILE)
    df_results = pd.read_sql_query("SELECT * FROM modal_results WHERE sample_id = ?", conn, params=(str(sample_id),))
    conn.close()
    
    if not df_results.empty:
        df_results["point_count"] = df_results["point_count"].apply(lambda v: _clean_numeric_val(v, to_float=False))
        df_results["modal_percent"] = df_results["modal_percent"].apply(lambda v: _clean_numeric_val(v, to_float=True))
        df_results["std_error"] = df_results["std_error"].apply(lambda v: _clean_numeric_val(v, to_float=True))
        df_results["density_gcm3"] = df_results["density_gcm3"].apply(lambda v: _clean_numeric_val(v, to_float=True) if pd.notnull(v) else np.nan)
        df_results["wt_percent"] = df_results["wt_percent"].apply(lambda v: _clean_numeric_val(v, to_float=True) if pd.notnull(v) else np.nan)
        
        for col in ["phase_name", "abundance", "en933_class", "carbonation", "liberation"]:
            if col in df_results.columns:
                df_results[col] = df_results[col].astype(str).replace({"None": "", "nan": ""})
                
    return df_results

# ---------------------------------------------------------
# 1. SAMPLE TYPES, TAXONOMY & CODES
# ---------------------------------------------------------
SAMPLE_TYPES = [
    "Aggregates",
    "Aggregate - Recycled",
    "Concrete - General",
    "Concrete - I.S.465",
    "Rock",
    "Concrete - Block/Precast",
    "Render/Mortar"
]

IS465_LITHOLOGY_CODES = [
    "MCM - Muscovite Mica",
    "MCP - Phlogopite Mica",
    "QTZ - Quartzite / Quartz",
    "CAL - Calcite / Limestone",
    "PIR - Pyrite",
    "PYR - Pyrrhotite",
    "SIL - Siltstone",
    "SND - Sandstone",
    "GRE - Greywacke",
    "SCH - Schist",
    "DOL - Dolomite",
    "FEL - Feldspar Matrix"
]

EN933_11_CLASSES = [
    "Rc - Concrete & Mortar",
    "Ru - Unbound / Natural Stone",
    "Rb - Clay Masonry & Bricks",
    "Ra - Bituminous / Asphalt",
    "FL - Floating Material (Wood/Plastic)",
    "X - Other Contaminants (Glass/Metal/Soil)"
]

CARBONATION_STATES = ["Uncarbonated", "Carbonated", "Non-cementitious"]
LIBERATION_INDEXES = ["I - Liberated", "II - Partly locked", "III - Fully locked", "IV - Paste only"]

DEFAULT_LOOKUPS = {
    "Aggregates": {
        "Phase1": "Quartz", "Phase2": "Plagioclase", "Phase3": "K-Feldspar", 
        "Phase4": "Feldspathoid", "Good_Agg": "Good Aggregate", "Bad_Agg": "Bad Aggregate",
        "Binder": "Cement Matrix", "NA": "NA"
    },
    "Aggregate - Recycled": {
        "Phase1": "Concrete Particle", "Phase2": "Brick / Masonry", "Phase3": "Natural Quartz Aggregate", 
        "Phase4": "Asphalt Fragment", "Phase5": "Hardened Cement Paste", "Good_Agg": "Clean Stone",
        "Bad_Agg": "Contaminated Aggregate", "Binder": "Residual Paste", "NA": "NA"
    },
    "Concrete - I.S.465": {
        "Phase1": "Quartz", "Phase2": "Cement Paste", "Phase3": "Mica / Phlogopite", 
        "Good_Agg": "Good Aggregate", "Bad_Agg": "Bad Aggregate", "Fracture": "Microcrack",
        "NA": "NA"
    },
    "Rock": {
        "Phase1": "Quartz", "Phase2": "Plagioclase", "Phase3": "K-Feldspar", 
        "Phase4": "Feldspathoid", "Phase5": "Biotite", "Phase6": "Amphibole", "NA": "NA"
    }
}

DEFAULT_DENSITIES = {
    "Quartz": 2.65, "Plagioclase": 2.68, "K-Feldspar": 2.56, "Feldspathoid": 2.45,
    "Biotite": 3.09, "Muscovite": 2.82, "Calcite": 2.71, "Pyrite": 5.01,
    "Pyrrhotite": 4.61, "Cement Paste": 2.10, "Good Aggregate": 2.65, "Bad Aggregate": 2.50
}

MAIN_BINDER_TYPES = {
    "CEM I - Ordinary Portland Cement (OPC)": 3.15,
    "CEM II/A-LL - Portland-Limestone Cement": 3.08,
    "CEM II/B-V - Portland-Fly Ash Cement": 2.95,
    "CEM III/A - Blastfurnace Slag Cement": 3.02,
    "CEM IV - Pozzolanic Cement": 2.88,
    "White Portland Cement": 3.15,
    "Sulfate Resisting Portland Cement (SRPC)": 3.18
}

SCM_DENSITIES = {
    "GGBS": 2.90,
    "PFA": 2.30,
    "Microsilica": 2.20,
    "Metakaolin": 2.55
}

def classify_abundance(pct):
    if pct > 50.0:
        return "Major+"
    elif 10.0 <= pct <= 50.0:
        return "Major"
    elif 2.0 <= pct < 10.0:
        return "Minor"
    else:
        return "Trace"

def evaluate_asr_risk(df_summary):
    high_react = ["opal", "chalcedony", "chert", "flint", "volcanic glass", "strained quartz"]
    norm_react = ["sandstone", "greywacke", "siltstone", "microcrystalline quartz"]
    
    high_pct = 0.0
    norm_pct = 0.0
    
    for _, row in df_summary.iterrows():
        p_name = str(row.get("Standardized_Phase", row.get("phase_name", ""))).lower()
        pct = float(_clean_numeric_val(row.get("Modal_Percent (%)", row.get("modal_percent", 0.0)), to_float=True))
        if any(h in p_name for h in high_react):
            high_pct += pct
        elif any(n in p_name for n in norm_react):
            norm_pct += pct
            
    if high_pct > 5.0:
        return "HIGH RISK", "🔴 Contains > 5% highly reactive silica constituents (BS 8500-2 Annex C)."
    elif norm_pct > 10.0 or high_pct > 0.0:
        return "MEDIUM / NORMAL RISK", "🟡 Contains normal reactivity aggregates. Apply standard concrete alkali limits."
    else:
        return "LOW RISK", "🟢 Predominantly low-reactivity minerals (Quartz/Feldspar/Carbonates)."

def calculate_spatial_index(df_points):
    if len(df_points) < 5:
        return None, "Insufficient points"
    
    coords = df_points[["X", "Y"]].values
    dist_mat = cdist(coords, coords, metric='euclidean')
    np.fill_diagonal(dist_mat, np.inf)
    
    nearest_dists = np.min(dist_mat, axis=1)
    mean_observed_dist = np.mean(nearest_dists)
    
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    area = (x_max - x_min) * (y_max - y_min)
    
    if area == 0:
        return None, "Zero Area"
    
    density = len(coords) / area
    mean_expected_dist = 0.5 / np.sqrt(density)
    
    R = mean_observed_dist / mean_expected_dist
    
    if R < 0.95:
        desc = "Clustered / Segregated"
    elif 0.95 <= R <= 1.05:
        desc = "Randomly Distributed"
    else:
        desc = "Uniformly / Regularly Distributed"
        
    return R, desc

# ---------------------------------------------------------
# PDF REPORT GENERATOR: TAB 1 PETROGRAPHIC QC REPORT
# ---------------------------------------------------------
def generate_petrography_pdf(meta_dict, df_summary, specific_metrics):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#1b365d')
    )
    h2_style = ParagraphStyle(
        'Heading2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#1b365d')
    )
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12)
    bold_style = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')
    
    story = []
    
    story.append(Paragraph("Petrographic Modal Analysis & QC Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 12))
    
    # 1. Sample Reference Data
    story.append(Paragraph("1. Sample Reference & Metadata", h2_style))
    meta_table_data = [
        [
            Paragraph("<b>Job Number:</b>", body_style), Paragraph(str(meta_dict.get("job_number", "N/A")), body_style),
            Paragraph("<b>Sample ID:</b>", body_style), Paragraph(str(meta_dict.get("sample_id", "N/A")), body_style)
        ],
        [
            Paragraph("<b>Sample Order:</b>", body_style), Paragraph(str(meta_dict.get("sample_order", "1")), body_style),
            Paragraph("<b>Operator:</b>", body_style), Paragraph(str(meta_dict.get("operator", "N/A")), body_style)
        ],
        [
            Paragraph("<b>Material Type:</b>", body_style), Paragraph(str(meta_dict.get("sample_type", "General")), body_style),
            Paragraph("<b>Lithology Code:</b>", body_style), Paragraph(str(meta_dict.get("lithology_code", "N/A")), body_style)
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[110, 150, 110, 150])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f2f6')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#d0d7de')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))

    # 2. Key Petrographic Metrics (Traverse & Material-Specific)
    story.append(Paragraph("2. Point Count & Material Metrics", h2_style))
    metrics_rows = [
        [Paragraph(f"<b>{k}:</b>", body_style), Paragraph(str(v), body_style)]
        for k, v in specific_metrics.items()
    ]
    t_metrics = Table(metrics_rows, colWidths=[200, 320])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e1e4e8')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 14))

    # 3. Modal Results Table
    story.append(Paragraph("3. Standardized Modal Composition Table", h2_style))
    col_names = ["Phase / Constituent", "Point Count", "Modal % (vol%)", "2-Sigma Err (±%)", "Abundance"]
    table_data = [[Paragraph(f"<b>{h}</b>", bold_style) for h in col_names]]
    
    for _, r in df_summary.iterrows():
        table_data.append([
            Paragraph(str(r.get("Standardized_Phase", r.get("phase_name", ""))), body_style),
            Paragraph(str(r.get("Point_Count", r.get("point_count", ""))), body_style),
            Paragraph(f"{float(r.get('Modal_Percent (%)', r.get('modal_percent', 0))):.2f}%", body_style),
            Paragraph(f"± {float(r.get('Std_Error (%)', r.get('std_error', 0))):.2f}%", body_style),
            Paragraph(str(r.get("Abundance", r.get("abundance", "-"))), body_style)
        ])
        
    t_modal = Table(table_data, colWidths=[160, 85, 95, 95, 85])
    t_modal.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1b365d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_modal)
    story.append(Spacer(1, 16))

    # 4. Graphical Modal Stacked Bar Chart
    story.append(Paragraph("4. Modal Distribution Graphic", h2_style))
    draw_w, draw_h = 520, 48
    d = Drawing(draw_w, draw_h)
    
    palette = [
        colors.HexColor('#3366cc'), colors.HexColor('#22aa99'), colors.HexColor('#ff9900'),
        colors.HexColor('#dc3912'), colors.HexColor('#990099'), colors.HexColor('#3b3eac'),
        colors.HexColor('#0099c6'), colors.HexColor('#dd4477')
    ]
    
    cur_x = 0
    bar_y = 20
    bar_h = 24
    
    legend_items = []
    for i, (_, r) in enumerate(df_summary.iterrows()):
        p_name = str(r.get("Standardized_Phase", r.get("phase_name", "")))
        pct = float(r.get("Modal_Percent (%)", r.get("modal_percent", 0.0)))
        col = palette[i % len(palette)]
        
        seg_w = (pct / 100.0) * draw_w
        if seg_w > 0:
            d.add(Rect(cur_x, bar_y, seg_w, bar_h, fillColor=col, strokeColor=colors.white, strokeWidth=1))
            if seg_w > 32:
                d.add(String(cur_x + 3, bar_y + 8, f"{pct:.1f}%", fontName="Helvetica-Bold", fontSize=7, fillColor=colors.white))
            cur_x += seg_w
            legend_items.append((p_name, col))

    leg_x = 0
    for p_name, col in legend_items[:5]:
        d.add(Rect(leg_x, 4, 8, 8, fillColor=col, strokeColor=None))
        d.add(String(leg_x + 12, 5, p_name[:15], fontName="Helvetica", fontSize=7))
        leg_x += 100
        
    story.append(d)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------
# PDF REPORT GENERATOR: TAB 3 CONCRETE MIX DESIGN REPORT
# ---------------------------------------------------------
def generate_mix_design_pdf(meta_dict, df_results, strength_dict, slump_dict, mass_pcts):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#1b365d')
    )
    h2_style = ParagraphStyle(
        'Heading2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#1b365d')
    )
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12)
    bold_style = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')
    
    story = []
    
    story.append(Paragraph("Concrete Mix Reverse-Engineering & Petrographic Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 12))
    
    # 1. Sample Metadata
    story.append(Paragraph("1. Sample Reference & Metadata", h2_style))
    meta_table_data = [
        [
            Paragraph("<b>Job Number:</b>", body_style), Paragraph(str(meta_dict.get("job_number", "N/A")), body_style),
            Paragraph("<b>Sample ID:</b>", body_style), Paragraph(str(meta_dict.get("sample_id", "N/A")), body_style)
        ],
        [
            Paragraph("<b>Sample Order:</b>", body_style), Paragraph(str(meta_dict.get("sample_order", "1")), body_style),
            Paragraph("<b>Operator:</b>", body_style), Paragraph(str(meta_dict.get("operator", "N/A")), body_style)
        ],
        [
            Paragraph("<b>Calculation Approach:</b>", body_style), Paragraph(str(meta_dict.get("approach", "Approach B")), body_style),
            Paragraph("<b>Est. w/b Ratio:</b>", body_style), Paragraph(f"{meta_dict.get('wb_ratio', 0.48):.2f}", body_style)
        ],
        [
            Paragraph("<b>Composite Binder:</b>", body_style), Paragraph(str(meta_dict.get("binder_desc", "CEM I")), body_style),
            Paragraph("<b>Binder Density:</b>", body_style), Paragraph(f"{meta_dict.get('rho_cement', 3.15):.2f} g/cm³", body_style)
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[110, 150, 110, 150])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f2f6')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#d0d7de')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))

    # 2. Mix Proportions Table
    story.append(Paragraph("2. Reconstructed Concrete Mix Design (per 1 m³)", h2_style))
    col_names = ["Constituent", "Mass (kg/m³)", "Mass % (wt%)", "Vol (L/m³)", "Vol %", "Proportion (1:X)"]
    table_data = [[Paragraph(f"<b>{h}</b>", bold_style) for h in col_names]]
    
    for _, r in df_results.iterrows():
        table_data.append([
            Paragraph(str(r["Constituent"]), body_style),
            Paragraph(str(r["Batch Weight (kg/m³)"]), body_style),
            Paragraph(str(r["Determined Mass % (wt%)"]), body_style),
            Paragraph(str(r["Batch Volume (L/m³)"]), body_style),
            Paragraph(str(r["Determined Vol %"]), body_style),
            Paragraph(str(r["Mix Proportion (1 : X)"]), body_style),
        ])
        
    t_mix = Table(table_data, colWidths=[130, 75, 75, 75, 70, 95])
    t_mix.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1b365d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_mix)
    story.append(Spacer(1, 14))

    # 3. 100% Mass Distribution Graphic
    story.append(Paragraph("3. Concrete Mix Composition (100% Mass Distribution Graphic)", h2_style))
    draw_w, draw_h = 520, 48
    d = Drawing(draw_w, draw_h)
    
    c_map = [
        ("Binder", mass_pcts.get("Binder", 0), colors.HexColor('#3366cc')),
        ("Water", mass_pcts.get("Water", 0), colors.HexColor('#22aa99')),
        ("Fine Agg", mass_pcts.get("Fine Agg", 0), colors.HexColor('#ff9900')),
        ("Coarse Agg", mass_pcts.get("Coarse Agg", 0), colors.HexColor('#dc3912'))
    ]
    
    cur_x = 0
    bar_y = 20
    bar_h = 24
    
    for label, pct, col in c_map:
        seg_w = (pct / 100.0) * draw_w
        if seg_w > 0:
            d.add(Rect(cur_x, bar_y, seg_w, bar_h, fillColor=col, strokeColor=colors.white, strokeWidth=1))
            if seg_w > 38:
                d.add(String(cur_x + 4, bar_y + 8, f"{label}: {pct:.1f}%", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.white))
            cur_x += seg_w
            
    d.add(Rect(0, 4, 10, 10, fillColor=colors.HexColor('#3366cc'), strokeColor=None))
    d.add(String(14, 6, "Binder", fontName="Helvetica", fontSize=8))
    d.add(Rect(70, 4, 10, 10, fillColor=colors.HexColor('#22aa99'), strokeColor=None))
    d.add(String(84, 6, "Water", fontName="Helvetica", fontSize=8))
    d.add(Rect(140, 4, 10, 10, fillColor=colors.HexColor('#ff9900'), strokeColor=None))
    d.add(String(154, 6, "Fine Aggregate", fontName="Helvetica", fontSize=8))
    d.add(Rect(235, 4, 10, 10, fillColor=colors.HexColor('#dc3912'), strokeColor=None))
    d.add(String(249, 6, "Coarse Aggregate", fontName="Helvetica", fontSize=8))
    
    story.append(d)
    story.append(Spacer(1, 16))

    # 4. Performance
    story.append(Paragraph("4. Estimated Engineering Performance", h2_style))
    perf_data = [
        [
            Paragraph("<b>Estimated 28-Day Cube Strength (fcu,28):</b>", bold_style),
            Paragraph(f"<b>{strength_dict.get('strength', 0.0):.1f} MPa</b> (95% CI: {strength_dict.get('ci_low', 0.0):.1f} – {strength_dict.get('ci_high', 0.0):.1f} MPa)", body_style)
        ],
        [
            Paragraph("<b>Air Void Strength Penalty:</b>", body_style),
            Paragraph(f"{strength_dict.get('air_penalty', 0.0):.1f}% reduction (Air Void content: {strength_dict.get('air_pct', 0.0):.1f}%)", body_style)
        ],
        [
            Paragraph("<b>Estimated Fresh Slump:</b>", bold_style),
            Paragraph(f"<b>{slump_dict.get('slump', 0.0):.0f} mm</b> ({slump_dict.get('class', 'N/A')})", body_style)
        ],
        [
            Paragraph("<b>Total Fresh Density:</b>", body_style),
            Paragraph(f"{meta_dict.get('fresh_density', 0.0):.0f} kg/m³", body_style)
        ]
    ]
    t_perf = Table(perf_data, colWidths=[200, 320])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#d0d7de')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e1e4e8')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_perf)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------
# PERSISTENT LABORATORY CONTEXT BAR
# ---------------------------------------------------------
active_id = st.session_state.get("active_sample_id", "No Active Sample")
active_job = st.session_state.get("active_job_number", "None")
active_type = st.session_state.get("active_sample_type", "General")
active_status = "Committed to DB" if st.session_state.get("is_db_committed", False) else "Session Draft"

st.markdown(f"""
<div class="context-ribbon">
    <div>
        <b>🔬 Active Sample:</b> <span style="font-size:1.05rem; color:#85bbf0;">{active_id}</span> &nbsp;|&nbsp; 
        <b>Job:</b> {active_job} &nbsp;|&nbsp; 
        <b>Material:</b> {active_type}
    </div>
    <div>
        <span class="status-pill {'pill-green' if active_status == 'Committed to DB' else 'pill-amber'}">{active_status}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN TAB NAVIGATION
# ---------------------------------------------------------
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6, main_tab7 = st.tabs([
    "🔬 Sample Point Count Analysis", 
    "⚖️ Aggregate Density & DB Editor",
    "🏗️ Concrete Mix Design (DB)", 
    "📊 Multi-Sample Comparison", 
    "🌐 Geological Drive Resources",
    "📁 Local Database Browser",
    "ℹ️ Application Guide & DB Schema"
])

# =========================================================
# TAB 1: SPLIT-PANE POINT COUNT ANALYSIS & INGESTION
# =========================================================
with main_tab1:
    st.sidebar.header("1. Sample Source & Metadata")
    
    input_mode = st.sidebar.radio(
        "Ingestion Mode:",
        ["Upload New File (CSV/TSV)", "Load Existing Sample from DB"],
        help="Upload raw counter coordinates or load an already saved record to alter mappings and recommit."
    )

    df_raw = None
    existing_sample_data = None
    
    if input_mode == "Upload New File (CSV/TSV)":
        uploaded_file = st.sidebar.file_uploader("Upload Point Count File (CSV/TSV)", type=["txt", "tsv", "csv"])
        job_number = st.sidebar.text_input("Job Number", value="JOB-2026-001")
        sample_order = st.sidebar.text_input("Sample Order", value="1")
        sample_type = st.sidebar.selectbox("Sample Type", SAMPLE_TYPES)
        operator = st.sidebar.text_input("Operator Name", value="Analyst")
        
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file, sep="\t", header=None)
                if df_raw.shape[1] == 1:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, sep=",", header=None)
                df_raw.columns = ["Image_File", "Point_ID", "X", "Y", "Raw_Term", "Code_ID", "Timestamp"]
            except Exception as e:
                st.error(f"Error reading file structure: {e}")
                st.stop()
    else:
        df_samples_db = load_all_samples()
        if not df_samples_db.empty:
            sel_sample_id = st.sidebar.selectbox("Select Sample from Database:", df_samples_db["sample_id"].unique())
            existing_sample_data = df_samples_db[df_samples_db["sample_id"] == sel_sample_id].iloc[0]
            
            job_number = st.sidebar.text_input("Job Number", value=str(existing_sample_data["job_number"]))
            sample_order = st.sidebar.text_input("Sample Order", value=str(existing_sample_data["sample_order"]))
            
            def_idx = SAMPLE_TYPES.index(existing_sample_data["sample_type"]) if existing_sample_data["sample_type"] in SAMPLE_TYPES else 0
            sample_type = st.sidebar.selectbox("Sample Type", SAMPLE_TYPES, index=def_idx)
            operator = st.sidebar.text_input("Operator Name", value=str(existing_sample_data["operator"]))
            
            df_modal_loaded = load_sample_details(sel_sample_id)
        else:
            st.sidebar.warning("Database has no records yet. Upload a file first.")
            st.info("👈 No database records found. Please switch to 'Upload New File (CSV/TSV)' in the sidebar.")

    # ------------------ SPLIT PANE WORKFLOW ------------------
    if input_mode == "Upload New File (CSV/TSV)" and df_raw is not None:
        sample_id = str(df_raw["Image_File"].iloc[0]).split(".")[0]

        # Geometry calculations
        df_raw["X_um"] = df_raw["X"] * SCALE_UM_PER_PX
        df_raw["Y_um"] = df_raw["Y"] * SCALE_UM_PER_PX
        x_min_um, x_max_um = df_raw["X_um"].min(), df_raw["X_um"].max()
        y_min_um, y_max_um = df_raw["Y_um"].min(), df_raw["Y_um"].max()
        width_mm = (x_max_um - x_min_um) / 1000.0
        height_mm = (y_max_um - y_min_um) / 1000.0
        area_mm2 = width_mm * height_mm

        unique_y = np.sort(df_raw["Y_um"].unique())
        unique_x = np.sort(df_raw["X_um"].unique())
        num_traverses = len(unique_y)
        num_transects = len(unique_x)
        total_linear_traverse_mm = num_traverses * width_mm

        left_pane, right_pane = st.columns([4, 6])
        
        with left_pane:
            st.subheader("📋 Ingestion & Geometry")
            sample_id = st.text_input("Sample ID", value=sample_id)
            
            litho_tag = ""
            if sample_type == "Concrete - I.S.465":
                litho_tag = st.selectbox("I.S. 465 Main Lithology Tag", IS465_LITHOLOGY_CODES)

            st.markdown(f"""
            <div class="metric-card">
                <b>Total Gross Points:</b> {len(df_raw)} &nbsp;|&nbsp; <b>Bounding Area:</b> {area_mm2:.2f} mm²<br>
                <b>Linear Traverse:</b> {total_linear_traverse_mm:.1f} mm &nbsp;|&nbsp; <b>Traverses/Transects:</b> {num_traverses} × {num_transects}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🛠️ Term Standardization")
            
            unique_terms = df_raw["Raw_Term"].unique().tolist()
            lookup = DEFAULT_LOOKUPS.get(sample_type, {})
            mapping_dict = {}
            
            for term in unique_terms:
                default_val = lookup.get(term, term)
                mapping_dict[term] = st.text_input(f"Raw: **{term}**", value=default_val, key=f"map_{term}")

            df_mapped = df_raw.copy()
            df_mapped["Standardized_Phase"] = df_mapped["Raw_Term"].map(mapping_dict)
            df_valid = df_mapped[df_mapped["Standardized_Phase"].str.upper() != "NA"].copy()
            valid_points_count = max(1, len(df_valid))
            na_count = len(df_mapped) - len(df_valid)

            st.caption(f"ℹ️ {na_count} 'NA' points excluded. Valid point count: **{valid_points_count}**.")

            # Compute summary early so both left_pane and right_pane can safely access it
            summary = df_valid.groupby("Standardized_Phase").size().reset_index(name="Point_Count")
            summary["Modal_Percent (%)"] = (summary["Point_Count"] / valid_points_count) * 100.0
            p = summary["Modal_Percent (%)"] / 100.0
            summary["Std_Error (%)"] = 2.0 * np.sqrt((p * (1.0 - p)) / valid_points_count) * 100.0

            if sample_type in ["Aggregates", "Aggregate - Recycled", "Rock"]:
                summary["Abundance"] = summary["Modal_Percent (%)"].apply(classify_abundance)

            summary["Modal_Percent (%)"] = summary["Modal_Percent (%)"].round(2)
            summary["Std_Error (%)"] = summary["Std_Error (%)"].round(2)

            tagging_data = {}
            if sample_type == "Aggregate - Recycled":
                st.write("##### EN 933-11 Classification Tags")
                active_phases = df_valid["Standardized_Phase"].unique().tolist()
                for phase in active_phases:
                    t_col1, t_col2 = st.columns(2)
                    en_class = t_col1.selectbox(f"EN933 ({phase})", EN933_11_CLASSES, key=f"en_{phase}")
                    carb_state = t_col2.selectbox(f"Carb ({phase})", CARBONATION_STATES, key=f"carb_{phase}")
                    lib_idx = st.selectbox(f"Liberation ({phase})", LIBERATION_INDEXES, key=f"lib_{phase}")
                    tagging_data[phase] = {"EN933_Class": en_class, "Carbonation": carb_state, "Liberation": lib_idx}

                df_valid["EN933_Class"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["EN933_Class"])
                df_valid["Carbonation"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["Carbonation"])
                df_valid["Liberation"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["Liberation"])

            # Save & PDF export
            st.markdown("---")
            c_btn1, c_btn2 = st.columns(2)
            
            with c_btn1:
                if st.button("💾 Confirm QC & Save to DB", type="primary"):
                    meta_payload = {
                        "sample_id": sample_id,
                        "job_number": job_number,
                        "sample_order": sample_order,
                        "sample_type": sample_type,
                        "operator": operator,
                        "total_points": len(df_raw),
                        "valid_points": valid_points_count,
                        "area_mm2": area_mm2,
                        "linear_traverse_mm": total_linear_traverse_mm,
                        "lithology_code": litho_tag
                    }
                    if sample_type == "Aggregate - Recycled" and tagging_data:
                        tag_df = pd.DataFrame.from_dict(tagging_data, orient='index').reset_index().rename(columns={'index': 'Standardized_Phase'})
                        summary_save = summary.merge(tag_df, on="Standardized_Phase", how="left")
                    else:
                        summary_save = summary
                        
                    save_sample_to_db(meta_payload, summary_save)
                    st.session_state["is_db_committed"] = True
                    st.success(f"Sample '{sample_id}' committed to database!")

        with right_pane:
            st.subheader("📊 Live Analytical Viewport")

            st.session_state["active_sample_id"] = sample_id
            st.session_state["active_summary"] = summary.copy()
            st.session_state["active_job_number"] = job_number
            st.session_state["active_sample_order"] = sample_order
            st.session_state["active_operator"] = operator
            st.session_state["active_sample_type"] = sample_type

            tab_plot1, tab_plot2, tab_plot3 = st.tabs(["📍 Spatial Distribution Map", "📊 Modal Percentages", "🧱 I.S. 465 / Standards"])

            R_val, R_desc = calculate_spatial_index(df_valid)

            with tab_plot1:
                fig_scatter = px.scatter(
                    df_valid, x="X_um", y="Y_um", color="Standardized_Phase",
                    title="Thin Section Coordinate Distribution (μm)"
                )
                fig_scatter.update_yaxes(autorange="reversed")
                fig_scatter.update_layout(height=420)
                st.plotly_chart(fig_scatter, width="stretch")

                if R_val is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <b>Clark-Evans Dispersion Ratio (R):</b> {R_val:.3f} &nbsp;|&nbsp; 
                        <b>Pattern:</b> <span class="status-pill pill-blue">{R_desc}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with tab_plot2:
                fig_bar = px.bar(
                    summary, x="Standardized_Phase", y="Modal_Percent (%)", 
                    text="Modal_Percent (%)", color="Standardized_Phase",
                    title="Phase Modal Composition (%)"
                )
                fig_bar.update_layout(height=340)
                st.plotly_chart(fig_bar, width="stretch")
                st.dataframe(summary, width="stretch")

            specific_metrics = {
                "Total Points Counted": len(df_raw),
                "Valid Phase Points": valid_points_count,
                "Slide Bounding Area": f"{area_mm2:.2f} mm²",
                "Total Traverse Length": f"{total_linear_traverse_mm:.1f} mm",
                "Clark-Evans Ratio (R)": f"{R_val:.3f} ({R_desc})" if R_val else "N/A"
            }

            with tab_plot3:
                if sample_type == "Concrete - I.S.465":
                    good_agg_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Good Aggregate", case=False), "Point_Count"].sum()
                    bad_agg_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Bad Aggregate", case=False), "Point_Count"].sum()
                    total_agg_cnt = good_agg_cnt + bad_agg_cnt

                    if total_agg_cnt > 0:
                        total_agg_pct = (total_agg_cnt / valid_points_count) * 100.0
                        pct_good_of_agg = (good_agg_cnt / total_agg_cnt) * 100.0
                        pct_bad_of_agg = (bad_agg_cnt / total_agg_cnt) * 100.0

                        st.metric("Total Aggregate Fraction", f"{total_agg_pct:.2f} %")
                        c_iso1, c_iso2 = st.columns(2)
                        c_iso1.metric("Good Aggregate Ratio", f"{pct_good_of_agg:.1f} %")
                        c_iso2.metric("Bad / Deleterious Ratio", f"{pct_bad_of_agg:.1f} %")
                        
                        specific_metrics["Total Aggregate %"] = f"{total_agg_pct:.2f}%"
                        specific_metrics["Good Aggregate % of Total"] = f"{pct_good_of_agg:.1f}%"
                        specific_metrics["Bad Aggregate % of Total"] = f"{pct_bad_of_agg:.1f}%"
                elif sample_type in ["Rock", "Aggregates"]:
                    q_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Quartz", case=False), "Point_Count"].sum()
                    a_cnt = summary.loc[summary["Standardized_Phase"].str.contains("K-Feldspar|Alkali", case=False), "Point_Count"].sum()
                    p_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Plagioclase", case=False), "Point_Count"].sum()
                    qap_sum = q_cnt + a_cnt + p_cnt
                    if qap_sum > 0:
                        norm_q = (q_cnt / qap_sum) * 100.0
                        norm_a = (a_cnt / qap_sum) * 100.0
                        norm_p = (p_cnt / qap_sum) * 100.0

                        fig_ternary = go.Figure(go.Scatterternary({
                            'mode': 'markers', 'a': [norm_q], 'b': [norm_a], 'c': [norm_p],
                            'text': [f"Sample: {sample_id}"], 'marker': {'symbol': 'circle', 'size': 12, 'color': 'red'}
                        }))
                        fig_ternary.update_layout({
                            'ternary': {'sum': 100, 'aaxis': {'title': 'Q'}, 'baxis': {'title': 'A'}, 'caxis': {'title': 'P'}},
                            'title': 'Q-A-P Ternary Classification'
                        })
                        st.plotly_chart(fig_ternary, width="stretch")
                        specific_metrics["QAPF Normalized Q / A / P"] = f"{norm_q:.1f}% / {norm_a:.1f}% / {norm_p:.1f}%"
                else:
                    st.info("No specific standard derived parameters for this material type.")

            # Tab 1 PDF Download Button in left pane
            with c_btn2:
                try:
                    pdf_tab1_meta = {
                        "sample_id": sample_id,
                        "job_number": job_number,
                        "sample_order": sample_order,
                        "sample_type": sample_type,
                        "operator": operator,
                        "lithology_code": litho_tag or "N/A"
                    }
                    pdf_tab1_bytes = generate_petrography_pdf(pdf_tab1_meta, summary, specific_metrics)
                    st.download_button(
                        label="📥 Download Petrographic Report (PDF)",
                        data=pdf_tab1_bytes,
                        file_name=f"{job_number}_{sample_id}_petrographic_report.pdf",
                        mime="application/pdf",
                        type="secondary"
                    )
                except Exception as e:
                    st.error(f"Error generating Tab 1 PDF: {e}")

    elif input_mode == "Load Existing Sample from DB" and existing_sample_data is not None:
        st.subheader(f"📋 Record Inspection: {existing_sample_data['sample_id']}")
        
        left_e, right_e = st.columns([4, 6])
        with left_e:
            sample_id = st.text_input("Sample ID", value=str(existing_sample_data["sample_id"]), disabled=True)
            cur_litho = str(existing_sample_data.get("lithology_code", ""))
            def_lith_idx = IS465_LITHOLOGY_CODES.index(cur_litho) if cur_litho in IS465_LITHOLOGY_CODES else 0
            
            litho_tag = ""
            if sample_type == "Concrete - I.S.465":
                litho_tag = st.selectbox("I.S. 465 Main Lithology Tag", IS465_LITHOLOGY_CODES, index=def_lith_idx)

            df_modal_edit = df_modal_loaded.copy()
            st.write("Modify phase names below:")
            new_names = []
            for idx, row in df_modal_edit.iterrows():
                n_name = st.text_input(f"Phase #{idx+1}", value=row["phase_name"], key=f"re_p_{idx}")
                new_names.append(n_name)
            df_modal_edit["phase_name"] = new_names
            
            valid_tot = max(1, df_modal_edit["point_count"].sum())
            df_modal_edit["modal_percent"] = (df_modal_edit["point_count"] / valid_tot * 100.0).round(2)
            p = df_modal_edit["modal_percent"] / 100.0
            df_modal_edit["std_error"] = (2.0 * np.sqrt((p * (1.0 - p)) / valid_tot) * 100.0).round(2)

            st.session_state["active_sample_id"] = sample_id
            st.session_state["active_summary"] = df_modal_edit.rename(columns={"phase_name": "Standardized_Phase", "modal_percent": "Modal_Percent (%)"}).copy()
            st.session_state["active_job_number"] = job_number
            st.session_state["active_sample_order"] = sample_order
            st.session_state["active_operator"] = operator
            st.session_state["active_sample_type"] = sample_type

            st.markdown("---")
            c_re1, c_re2 = st.columns(2)
            with c_re1:
                if st.button("💾 Overwrite & Recommit to DB", type="primary"):
                    meta_recommit = {
                        "sample_id": sample_id,
                        "job_number": job_number,
                        "sample_order": sample_order,
                        "sample_type": sample_type,
                        "operator": operator,
                        "total_points": existing_sample_data["total_points"],
                        "valid_points": valid_tot,
                        "area_mm2": existing_sample_data["bounding_area_mm2"],
                        "linear_traverse_mm": existing_sample_data["linear_traverse_mm"],
                        "lithology_code": litho_tag
                    }
                    save_sample_to_db(meta_recommit, df_modal_edit)
                    st.session_state["is_db_committed"] = True
                    st.success(f"Sample '{sample_id}' successfully updated and recommitted!")

            with c_re2:
                try:
                    db_pdf_meta = {
                        "sample_id": sample_id, "job_number": job_number, "sample_order": sample_order,
                        "sample_type": sample_type, "operator": operator, "lithology_code": litho_tag or "N/A"
                    }
                    db_metrics = {
                        "Total Points Counted": existing_sample_data["total_points"],
                        "Valid Phase Points": valid_tot,
                        "Slide Bounding Area": f"{float(existing_sample_data['bounding_area_mm2']):.2f} mm²",
                        "Linear Traverse": f"{float(existing_sample_data['linear_traverse_mm']):.1f} mm"
                    }
                    pdf_recommit_bytes = generate_petrography_pdf(db_pdf_meta, df_modal_edit, db_metrics)
                    st.download_button(
                        label="📥 Download Petrographic Report (PDF)",
                        data=pdf_recommit_bytes,
                        file_name=f"{job_number}_{sample_id}_petrographic_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error generating Tab 1 PDF: {e}")

        with right_e:
            st.write("##### Stored Modal Composition")
            fig_edit = px.bar(df_modal_edit, x="phase_name", y="modal_percent", color="phase_name")
            st.plotly_chart(fig_edit, width="stretch")
            st.dataframe(df_modal_edit[["phase_name", "point_count", "modal_percent", "std_error", "abundance"]], width="stretch")
    else:
        st.info("👈 Please upload a point count file or choose a database sample from the left sidebar to begin.")

# =========================================================
# TAB 2: INLINE DATA GRID DENSITY & WT% EDITOR
# =========================================================
with main_tab2:
    st.header("⚖️ Database Aggregate Density & wt% DataGrid Editor")
    st.info("Directly double-click cells in the interactive spreadsheet below to reassign specific densities (g/cm³). Weight percentages recalculate automatically.")

    df_db_samples = load_all_samples()
    if not df_db_samples.empty:
        selected_db_id = st.selectbox("Select Sample to Edit:", df_db_samples["sample_id"].unique(), key="db_edit_sel")
        
        if selected_db_id:
            sample_row = df_db_samples[df_db_samples["sample_id"] == selected_db_id].iloc[0]
            df_details = load_sample_details(selected_db_id)
            
            df_details["density_gcm3"] = df_details.apply(
                lambda r: float(r["density_gcm3"]) if pd.notnull(r["density_gcm3"]) else DEFAULT_DENSITIES.get(r["phase_name"], 2.65),
                axis=1
            )

            grid_df = df_details[["phase_name", "point_count", "modal_percent", "density_gcm3"]].copy()
            
            edited_grid = st.data_editor(
                grid_df,
                disabled=["phase_name", "point_count", "modal_percent"],
                column_config={
                    "phase_name": st.column_config.TextColumn("Phase / Mineral", help="Standardized phase name"),
                    "point_count": st.column_config.NumberColumn("Points", help="Stereological raw count"),
                    "modal_percent": st.column_config.NumberColumn("Modal Vol %", format="%.2f %%"),
                    "density_gcm3": st.column_config.NumberColumn("Density (g/cm³)", min_value=1.0, max_value=8.0, step=0.01, format="%.2f")
                },
                use_container_width=True,
                key=f"editor_{selected_db_id}"
            )

            mass_factors = edited_grid["modal_percent"] * edited_grid["density_gcm3"]
            total_mass = mass_factors.sum()
            edited_grid["wt_percent"] = ((mass_factors / total_mass) * 100.0).round(2) if total_mass > 0 else 0.0

            st.write("##### Resulting Mass Balance & ASR Threat Assessment")
            res_col1, res_col2 = st.columns([1.5, 1])
            with res_col1:
                st.dataframe(edited_grid[["phase_name", "modal_percent", "density_gcm3", "wt_percent"]], width="stretch")

            with res_col2:
                df_eval = edited_grid.rename(columns={"phase_name": "Standardized_Phase", "modal_percent": "Modal_Percent (%)"})
                re_asr_risk, re_asr_desc = evaluate_asr_risk(df_eval)
                
                risk_pill = "pill-green" if "LOW" in re_asr_risk else ("pill-amber" if "MEDIUM" in re_asr_risk else "pill-red")
                st.markdown(f"""
                <div class="metric-card">
                    <b>ASR Risk Rating:</b> <span class="status-pill {risk_pill}">{re_asr_risk}</span>
                    <p style="font-size:0.85rem; margin-top:8px;">{re_asr_desc}</p>
                </div>
                """, unsafe_allow_html=True)

            if st.button("💾 Save Reassigned Densities to Database", type="primary"):
                meta_update = {
                    "sample_id": sample_row["sample_id"],
                    "job_number": sample_row["job_number"],
                    "sample_order": sample_row["sample_order"],
                    "sample_type": sample_row["sample_type"],
                    "operator": sample_row["operator"],
                    "total_points": sample_row["total_points"],
                    "valid_points": sample_row["valid_points"],
                    "area_mm2": sample_row["bounding_area_mm2"],
                    "linear_traverse_mm": sample_row["linear_traverse_mm"],
                    "lithology_code": sample_row.get("lithology_code", "")
                }
                
                df_details["density_gcm3"] = edited_grid["density_gcm3"]
                df_details["wt_percent"] = edited_grid["wt_percent"]
                
                df_details_to_save = df_details.rename(columns={
                    "phase_name": "Standardized_Phase",
                    "point_count": "Point_Count",
                    "modal_percent": "Modal_Percent (%)",
                    "std_error": "Std_Error (%)",
                    "abundance": "Abundance",
                    "density_gcm3": "Density (g/cm³)",
                    "wt_percent": "Weight_Percent (%)",
                    "en933_class": "EN933_Class",
                    "carbonation": "Carbonation",
                    "liberation": "Liberation"
                })
                
                save_sample_to_db(meta_update, df_details_to_save)
                st.success(f"Sample '{selected_db_id}' successfully updated in database!")
    else:
        st.info("No saved samples available in the database to edit yet.")

# =========================================================
# TAB 3: BENTO GRID CONCRETE MIX DESIGN ENGINE
# =========================================================
with main_tab3:
    st.header("🏗️ Hardened Concrete Mix Reverse-Engineering & Performance Engine")
    st.info(r"Convert stereological area fractions into batch quantities ($1\text{ m}^3$), 28-day cube strengths, and slump workability classes.")

    mix_source = st.radio("Select Modal Data Source:", ["Active Session Point Count", "Load Sample from Database"], horizontal=True)

    df_mix_input = pd.DataFrame()
    mix_sample_label = "Unassigned"
    cur_job_num = "JOB-2026-001"
    cur_sample_ord = "1"
    cur_op_name = "Analyst"

    if mix_source == "Active Session Point Count":
        if "active_summary" in st.session_state and not st.session_state["active_summary"].empty:
            df_mix_input = st.session_state["active_summary"].copy()
            df_mix_input = df_mix_input.rename(columns={"Standardized_Phase": "Phase", "Modal_Percent (%)": "Vol_Pct"})
            mix_sample_label = st.session_state.get("active_sample_id", "Active Sample")
            cur_job_num = st.session_state.get("active_job_number", "JOB-2026-001")
            cur_sample_ord = st.session_state.get("active_sample_order", "1")
            cur_op_name = st.session_state.get("active_operator", "Analyst")
            st.success(f"Loaded active point count data for: **{mix_sample_label}**")
        else:
            st.warning("No active sample point count found in Tab 1. Upload a sample or choose 'Load Sample from Database'.")
    else:
        df_all_db = load_all_samples()
        if not df_all_db.empty:
            concrete_db = df_all_db[df_all_db["sample_type"].str.contains("Concrete", case=False)]
            if not concrete_db.empty:
                sel_db_mix = st.selectbox("Select Concrete Sample from Database:", concrete_db["sample_id"].unique(), key="sel_mix_db")
                db_row = concrete_db[concrete_db["sample_id"] == sel_db_mix].iloc[0]
                df_modal_db = load_sample_details(sel_db_mix)
                df_mix_input = df_modal_db.rename(columns={"phase_name": "Phase", "modal_percent": "Vol_Pct"}).copy()
                mix_sample_label = sel_db_mix
                cur_job_num = db_row["job_number"]
                cur_sample_ord = db_row["sample_order"]
                cur_op_name = db_row["operator"]
                st.success(f"Loaded database point count data for: **{mix_sample_label}**")
            else:
                st.warning("No samples categorized as 'Concrete' found in the database.")
        else:
            st.info("The database is currently empty. Process samples in Tab 1 first.")

    if not df_mix_input.empty:
        st.markdown("---")
        
        # 1. Config Ribbon
        with st.expander("⚙️ Component Classification & Physical Densities", expanded=False):
            def auto_classify(p_name):
                p = str(p_name).lower()
                if any(k in p for k in ["paste", "cement", "binder", "matrix"]):
                    return "Cement Paste / Binder"
                elif any(k in p for k in ["void", "air", "pore"]):
                    return "Air Voids"
                elif any(k in p for k in ["sand", "fine", "matrix sand"]):
                    return "Fine Aggregate"
                else:
                    return "Coarse Aggregate"

            classifications = []
            c_cols = st.columns(min(len(df_mix_input), 4))
            for idx, row in df_mix_input.iterrows():
                p_name = row["Phase"]
                def_cat = auto_classify(p_name)
                with c_cols[idx % len(c_cols)]:
                    cat = st.selectbox(
                        f"Role: {p_name}",
                        ["Cement Paste / Binder", "Fine Aggregate", "Coarse Aggregate", "Air Voids"],
                        index=["Cement Paste / Binder", "Fine Aggregate", "Coarse Aggregate", "Air Voids"].index(def_cat),
                        key=f"cat_{mix_sample_label}_{p_name}"
                    )
                    classifications.append(cat)
                    
            df_mix_input["Role"] = classifications

        v_paste = float(df_mix_input.loc[df_mix_input["Role"] == "Cement Paste / Binder", "Vol_Pct"].sum())
        v_fine = float(df_mix_input.loc[df_mix_input["Role"] == "Fine Aggregate", "Vol_Pct"].sum())
        v_coarse = float(df_mix_input.loc[df_mix_input["Role"] == "Coarse Aggregate", "Vol_Pct"].sum())
        v_void = float(df_mix_input.loc[df_mix_input["Role"] == "Air Voids", "Vol_Pct"].sum())

        # Parameters & SCM Proportional Replacement Sliders
        p_col1, p_col2 = st.columns([1, 1.2])
        with p_col1:
            wb_ratio = st.slider("Estimated w/b Ratio:", 0.25, 0.80, 0.48, 0.01)
            calc_approach = st.radio("Approach:", ["Approach B: Mass Balance (BS 1881-124)", "Approach A: Powers-Brownyard"], horizontal=True)
            
            c_d1, c_d2 = st.columns(2)
            rho_fine = float(c_d1.number_input("Fine Agg ρ (g/cm³)", value=2.65, step=0.01))
            rho_coarse = float(c_d2.number_input("Coarse Agg ρ (g/cm³)", value=2.68, step=0.01))

        with p_col2:
            st.write("##### Binder Composition & SCM Proportional Replacement")
            main_binder = st.selectbox("Main Binder Type:", list(MAIN_BINDER_TYPES.keys()))
            rho_main_binder = MAIN_BINDER_TYPES[main_binder]

            st.caption("Adjust percentage replacement proportions (by mass):")
            sc_1, sc_2 = st.columns(2)
            pct_ggbs = sc_1.slider("GGBS (Slag) %", 0, 80, 0, step=5)
            pct_pfa = sc_2.slider("PFA (Fly Ash) %", 0, 50, 0, step=5)
            
            sc_3, sc_4 = st.columns(2)
            pct_sf = sc_3.slider("Microsilica (Silica Fume) %", 0, 15, 0, step=1)
            pct_mk = sc_4.slider("Metakaolin %", 0, 20, 0, step=1)

            total_scm_pct = pct_ggbs + pct_pfa + pct_sf + pct_mk
            pct_main = max(0.0, 100.0 - total_scm_pct)

            if total_scm_pct > 100.0:
                st.error("⚠️ Total SCM replacement exceeds 100%! Please reduce slider proportions.")
                st.stop()

            # Harmonic mean calculation for composite effective binder density
            inv_rho_b = (
                (pct_main / 100.0) / rho_main_binder +
                (pct_ggbs / 100.0) / SCM_DENSITIES["GGBS"] +
                (pct_pfa / 100.0) / SCM_DENSITIES["PFA"] +
                (pct_sf / 100.0) / SCM_DENSITIES["Microsilica"] +
                (pct_mk / 100.0) / SCM_DENSITIES["Metakaolin"]
            )
            rho_cement = 1.0 / inv_rho_b

            binder_desc_str = f"{pct_main:.0f}% {main_binder.split(' - ')[0]}"
            if pct_ggbs > 0: binder_desc_str += f" + {pct_ggbs}% GGBS"
            if pct_pfa > 0: binder_desc_str += f" + {pct_pfa}% PFA"
            if pct_sf > 0: binder_desc_str += f" + {pct_sf}% Microsilica"
            if pct_mk > 0: binder_desc_str += f" + {pct_mk}% Metakaolin"

            st.markdown(f"""
            <div class="metric-card" style="padding:6px 12px; margin-top:4px;">
                <b>Effective Binder ρ:</b> {rho_cement:.3f} g/cm³ &nbsp;|&nbsp; 
                <b>Main Portland:</b> {pct_main:.0f}% &nbsp;|&nbsp; 
                <b>Total SCM:</b> {total_scm_pct:.0f}%
            </div>
            """, unsafe_allow_html=True)

        # Reconstructed batch physics
        rho_water = 1.00
        v_total_agg = v_fine + v_coarse

        # Approach B (BS 1881-124)
        v_paste_avail_m3 = max(0.05, 1.0 - ((v_total_agg + v_void) / 100.0))
        denom_b = (1.0 / (rho_cement * 1000.0)) + (wb_ratio / (rho_water * 1000.0))
        c_kg_b = v_paste_avail_m3 / denom_b
        w_kg_b = c_kg_b * wb_ratio
        fine_kg_b = 10.0 * v_fine * rho_fine
        coarse_kg_b = 10.0 * v_coarse * rho_coarse
        fresh_density_b = c_kg_b + w_kg_b + fine_kg_b + coarse_kg_b

        # Approach A (Powers-Brownyard)
        v_c0_pct = v_paste / (1.0 + wb_ratio * (rho_cement / rho_water)) if (1.0 + wb_ratio * (rho_cement / rho_water)) > 0 else 0
        c_kg_a = 10.0 * v_c0_pct * rho_cement
        w_kg_a = c_kg_a * wb_ratio
        fine_kg_a = 10.0 * v_fine * rho_fine
        coarse_kg_a = 10.0 * v_coarse * rho_coarse
        fresh_density_a = c_kg_a + w_kg_a + fine_kg_a + coarse_kg_a

        is_approach_b = "Approach B" in calc_approach
        c_kg = c_kg_b if is_approach_b else c_kg_a
        w_kg = w_kg_b if is_approach_b else w_kg_a
        fine_kg = fine_kg_b if is_approach_b else fine_kg_a
        coarse_kg = coarse_kg_b if is_approach_b else coarse_kg_a
        fresh_density = fresh_density_b if is_approach_b else fresh_density_a

        v_c_L = c_kg / rho_cement
        v_w_L = w_kg / rho_water
        v_fine_L = fine_kg / rho_fine
        v_coarse_L = coarse_kg / rho_coarse
        v_air_L = v_void * 10.0
        total_batch_vol_L = v_c_L + v_w_L + v_fine_L + v_coarse_L + v_air_L

        c_wt_pct = (c_kg / fresh_density) * 100.0 if fresh_density > 0 else 0.0
        w_wt_pct = (w_kg / fresh_density) * 100.0 if fresh_density > 0 else 0.0
        fine_wt_pct = (fine_kg / fresh_density) * 100.0 if fresh_density > 0 else 0.0
        coarse_wt_pct = (coarse_kg / fresh_density) * 100.0 if fresh_density > 0 else 0.0
        total_agg_wt_pct = fine_wt_pct + coarse_wt_pct

        c_vol_pct = (v_c_L / total_batch_vol_L) * 100.0 if total_batch_vol_L > 0 else 0.0
        w_vol_pct = (v_w_L / total_batch_vol_L) * 100.0 if total_batch_vol_L > 0 else 0.0
        fine_vol_pct = (v_fine_L / total_batch_vol_L) * 100.0 if total_batch_vol_L > 0 else 0.0
        coarse_vol_pct = (v_coarse_L / total_batch_vol_L) * 100.0 if total_batch_vol_L > 0 else 0.0
        air_vol_pct = (v_air_L / total_batch_vol_L) * 100.0 if total_batch_vol_L > 0 else 0.0

        prop_fine = fine_kg / c_kg if c_kg > 0 else 0.0
        prop_coarse = coarse_kg / c_kg if c_kg > 0 else 0.0
        total_agg_ratio = prop_fine + prop_coarse

        # ------------------ BENTO 2X2 ANALYTICAL GRID ------------------
        bento_r1_c1, bento_r1_c2 = st.columns(2)
        
        with bento_r1_c1:
            st.markdown("#### 1. Reconstructed Batch Matrix (1 m³)")
            df_mix_res = pd.DataFrame({
                "Constituent": ["Binder (Composite)", "Water", "Fine Agg", "Coarse Agg", "Air Voids"],
                "Batch Weight (kg/m³)": [f"{c_kg:.1f}", f"{w_kg:.1f}", f"{fine_kg:.1f}", f"{coarse_kg:.1f}", "0.0"],
                "Determined Mass % (wt%)": [f"{c_wt_pct:.1f}%", f"{w_wt_pct:.1f}%", f"{fine_wt_pct:.1f}%", f"{coarse_wt_pct:.1f}%", "0.0%"],
                "Batch Volume (L/m³)": [f"{v_c_L:.1f}", f"{v_w_L:.1f}", f"{v_fine_L:.1f}", f"{v_coarse_L:.1f}", f"{v_air_L:.1f}"],
                "Determined Vol %": [f"{c_vol_pct:.1f}%", f"{w_vol_pct:.1f}%", f"{fine_vol_pct:.1f}%", f"{coarse_vol_pct:.1f}%", f"{air_vol_pct:.1f}%"],
                "Mix Proportion (1 : X)": ["1.00", f"{wb_ratio:.2f}", f"{prop_fine:.2f}", f"{prop_coarse:.2f}", f"{v_void:.1f}% Air"]
            })
            st.dataframe(df_mix_res, width="stretch")
            st.markdown(f"""
            <div class="metric-card">
                <b>Fresh Density:</b> {fresh_density:.0f} kg/m³ &nbsp;|&nbsp; 
                <b>Agg/Binder Ratio:</b> 1 : {total_agg_ratio:.2f} &nbsp;|&nbsp; 
                <b>Total Aggregates:</b> {total_agg_wt_pct:.1f} wt%
            </div>
            """, unsafe_allow_html=True)

        with bento_r1_c2:
            st.markdown("#### 2. Mass & Volumetric Distribution")
            df_stacked_mass = pd.DataFrame([
                {"Category": "Mass (wt%)", "Constituent": "Binder", "Percentage": c_wt_pct},
                {"Category": "Mass (wt%)", "Constituent": "Water", "Percentage": w_wt_pct},
                {"Category": "Mass (wt%)", "Constituent": "Fine Agg", "Percentage": fine_wt_pct},
                {"Category": "Mass (wt%)", "Constituent": "Coarse Agg", "Percentage": coarse_wt_pct}
            ])
            fig_mass = px.bar(
                df_stacked_mass, y="Category", x="Percentage", color="Constituent",
                orientation='h', barmode='stack', text="Percentage",
                color_discrete_map={"Binder": "#3366cc", "Water": "#22aa99", "Fine Agg": "#ff9900", "Coarse Agg": "#dc3912"}
            )
            fig_mass.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
            fig_mass.update_layout(xaxis=dict(ticksuffix="%", range=[0, 100]), yaxis_title=None, height=200, margin=dict(t=10, b=10))
            st.plotly_chart(fig_mass, width="stretch")

        bento_r2_c1, bento_r2_c2 = st.columns(2)
        
        with bento_r2_c1:
            st.markdown("#### 3. 28-Day Strength (UK DoE 1B)")
            s_c1, s_c2 = st.columns(2)
            cement_grade = s_c1.selectbox("Cement Strength Class:", ["CEM I 52.5N / 52.5R", "CEM I 42.5N / 42.5R", "CEM II 32.5N"])
            coarse_type = s_c2.selectbox("Aggregate Mineral Texture:", ["Crushed Rock", "Uncrushed Gravel"])

            base_f = 46.0 if "52.5" in cement_grade and "Crushed" in coarse_type else (
                     42.0 if "52.5" in cement_grade else (
                     39.0 if "42.5" in cement_grade and "Crushed" in coarse_type else (
                     35.0 if "42.5" in cement_grade else 31.0)))

            calc_fcu_28 = base_f - 24.0 * np.log(max(0.20, wb_ratio) / 0.50)
            air_excess = max(0.0, v_void - 1.0)
            air_penalty_factor = max(0.50, 1.0 - (0.055 * air_excess))
            final_fcu_28 = max(5.0, calc_fcu_28 * air_penalty_factor)
            ci_low = max(5.0, final_fcu_28 - 4.5)
            ci_high = final_fcu_28 + 4.5

            st.markdown(f"""
            <div class="metric-card">
                <b>Estimated Cube Strength:</b> <span style="font-size:1.25rem; font-weight:700; color:#111111;">{final_fcu_28:.1f} MPa</span><br>
                <b>95% Confidence Interval:</b> {ci_low:.1f} – {ci_high:.1f} MPa<br>
                <b>Air Void Penalty:</b> {((1.0 - air_penalty_factor)*100):.1f}% reduction ({v_void:.1f}% voids)
            </div>
            """, unsafe_allow_html=True)

        with bento_r2_c2:
            st.markdown("#### 4. Slump & Workability Estimator")
            w_c1, w_c2 = st.columns(2)
            nom_max_agg = w_c1.selectbox("Max Aggregate Size:", ["20 mm", "10 mm", "40 mm"])
            admixture_scenario = w_c2.selectbox("Admixture 'What-If':", ["Plain (beta=1.00)", "Lignosulfonate (beta=0.90)", "SNF Mid-Range (beta=0.82)", "PCE High-Range (beta=0.70)"])

            w_base = 230.0 if "10 mm" in nom_max_agg else (190.0 if "40 mm" in nom_max_agg else 210.0)
            beta_admix = 0.90 if "Lignosulfonate" in admixture_scenario else (0.82 if "SNF" in admixture_scenario else (0.70 if "PCE" in admixture_scenario else 1.00))
            
            p_sand = (v_fine / v_total_agg * 100.0) if v_total_agg > 0 else 35.0
            w_sand = 0.5 * (p_sand - 35.0)
            effective_target_water = (w_base + w_sand) * beta_admix
            delta_w = w_kg - effective_target_water
            est_slump = 50.0 + (2.5 * delta_w)

            slump_class = "S1 (10-40 mm)" if est_slump <= 25 else ("S2 (50-90 mm)" if est_slump <= 90 else ("S3 (100-150 mm)" if est_slump <= 150 else "S4 (160-210 mm)"))
            est_slump_clamped = max(5.0, min(250.0, est_slump))

            st.markdown(f"""
            <div class="metric-card">
                <b>Estimated Slump:</b> <span style="font-size:1.25rem; font-weight:700; color:#111111;">{est_slump_clamped:.0f} mm</span><br>
                <b>Workability Class:</b> <span class="status-pill pill-blue">{slump_class}</span><br>
                <b>Target Water Demand:</b> {effective_target_water:.1f} kg/m³
            </div>
            """, unsafe_allow_html=True)

        # PDF Export
        st.markdown("---")
        pdf_meta = {
            "sample_id": mix_sample_label, "job_number": cur_job_num, "sample_order": cur_sample_ord,
            "operator": cur_op_name, "approach": "Approach B (Mass Balance)" if is_approach_b else "Approach A (Powers-Brownyard)",
            "wb_ratio": wb_ratio, "fresh_density": fresh_density,
            "binder_desc": binder_desc_str, "rho_cement": rho_cement
        }
        pdf_strength = {"strength": final_fcu_28, "ci_low": ci_low, "ci_high": ci_high, "air_penalty": (1.0 - air_penalty_factor) * 100.0, "air_pct": v_void}
        pdf_slump = {"slump": est_slump_clamped, "class": slump_class}
        pdf_mass_pcts = {"Binder": c_wt_pct, "Water": w_wt_pct, "Fine Agg": fine_wt_pct, "Coarse Agg": coarse_wt_pct}

        try:
            pdf_bytes = generate_mix_design_pdf(pdf_meta, df_mix_res, pdf_strength, pdf_slump, pdf_mass_pcts)
            st.download_button(
                label="📥 Download Concrete Mix Design Report (PDF)",
                data=pdf_bytes,
                file_name=f"{cur_job_num}_{mix_sample_label}_mix_report.pdf",
                mime="application/pdf",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error generating PDF report: {e}")

# =========================================================
# TAB 4: MULTI-SAMPLE COMPARISON PLACEHOLDER
# =========================================================
with main_tab4:
    st.header("📊 Multi-Sample Comparative Analysis Dashboard")
    st.info("Compare petrographic results across multiple samples of the same sample type.")
    
    df_samples = load_all_samples()
    if not df_samples.empty:
        comp_type = st.selectbox("Select Sample Type to Compare", df_samples["sample_type"].unique(), key="comp_type_sel")
        filtered_s = df_samples[df_samples["sample_type"] == comp_type]
        st.write(f"Found **{len(filtered_s)}** samples for type: **{comp_type}**")
        st.dataframe(filtered_s, width="stretch")
    else:
        st.info("No saved database samples available.")

# =========================================================
# TAB 5: GEOLOGICAL DRIVE RESOURCES PLACEHOLDER
# =========================================================
with main_tab5:
    st.header("🌐 Geological Google Drive Resources")
    st.info("Connect to Google Drive hosted geological manuals, thin section reference atlases, and standards.")
    st.text_input("Google Drive Folder ID / Share Link", value="https://drive.google.com/drive/folders/placeholder")
    st.button("🔗 Connect to Google Workspace Drive")

# =========================================================
# TAB 6: LOCAL DATABASE BROWSER
# =========================================================
with main_tab6:
    st.subheader("📁 Saved Geological Samples History")
    df_history = load_all_samples()
    
    if not df_history.empty:
        st.dataframe(df_history, width="stretch")
        selected_sample = st.selectbox("Select Sample to View Modal Breakdown", df_history["sample_id"].unique(), key="hist_db_sel")
        if selected_sample:
            df_modal_history = load_sample_details(selected_sample)
            st.write(f"##### Modal Breakdown for: **{selected_sample}**")
            st.dataframe(df_modal_history, width="stretch")
    else:
        st.info("No saved samples found in the local database yet.")

# =========================================================
# TAB 7: APPLICATION GUIDE, DB SCHEMA & DEVELOPER INFO
# =========================================================
with main_tab7:
    st.header("ℹ️ Application Guide, Relational DB Schema & Developer Information")
    
    st.markdown("### 📘 Functional Tab Overview & User Guide")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(r"""
        * **Tab 1: 🔬 Sample Point Count Analysis**  
          *Ingest, standardize, and analyze raw optical counter coordinates.* Use the split-pane interface to standardise terms and evaluate Van der Plas & Tobi 2-sigma bounds, Clark-Evans spatial dispersion, and I.S. 465 good vs. bad aggregate ratios. Export dedicated petrographic reports to PDF.
        
        * **Tab 2: ⚖️ Aggregate Density & DB Editor**  
          *Post-hoc density management & spreadsheet recalculations.* Reassign phase densities ($\text{g/cm}^3$) directly inside the interactive spreadsheet grid to update mass-based weight percentages ($\text{wt}\%$) and BS 8500-2 ASR risk ratings.
          
        * **Tab 3: 🏗️ Concrete Mix Design (DB)**  
          *Mix reverse-engineering & engineering performance predictions.* Converts hardened paste, aggregate, and air void fractions into batch weights per $\text{m}^3$ using BS 1881-124 mass balance or Powers-Brownyard stoichiometry. Adjust composite binders with SCM replacement sliders and export mix design PDF reports.
        """)
    with col_g2:
        st.markdown("""
        * **Tab 4: 📊 Multi-Sample Comparison**  
          *Cross-sample analytics.* Cross-examine variations and modal distributions across multiple thin sections belonging to the same project or quarry source.
          
        * **Tab 5: 🌐 Geological Drive Resources**  
          *Cloud repository integration.* Connects to shared Google Drive folders for rapid reference to standard optical atlases and testing protocols.
          
        * **Tab 6: 📁 Local Database Browser**  
          *Master database record inspection.* Browse parent `samples` and child `modal_results` records, verify integrity, and inspect historical timestamps.
        """)
        
    st.markdown("---")
    st.markdown("### 🗄️ Relational Database Schema Architecture")
    st.code("""
  +---------------------------------------------------------------------------------+
  |                                  SAMPLES (Master)                               |
  +---------------------------------------------------------------------------------+
  | sample_id (PK)         : VARCHAR(100) -> Unique sample identifier               |
  | job_number             : VARCHAR(100) -> Job / Project QC tracking code         |
  | sample_order           : VARCHAR(50)  -> Sequential order / thin section ID     |
  | sample_type            : VARCHAR(100) -> Aggregates, Concrete, Rock, etc.       |
  | operator               : VARCHAR(100) -> Analyst name                           |
  | total_points           : INTEGER      -> Total gross points counted             |
  | valid_points           : INTEGER      -> Points excluding "NA"                  |
  | bounding_area_mm2      : REAL         -> Calculated thin section area (mm²)     |
  | linear_traverse_mm     : REAL         -> Total traverse length (mm)             |
  | lithology_code         : VARCHAR(50)  -> 3-letter I.S. 465 code (e.g., MCM)     |
  | created_at             : TIMESTAMP    -> Date & time of commitment              |
  +---------------------------------------------------------------------------------+
                                           |
                                           | 1 : N (One-to-Many Relationship)
                                           v
  +---------------------------------------------------------------------------------+
  |                               MODAL_RESULTS (Child)                             |
  +---------------------------------------------------------------------------------+
  | id (PK)                : INTEGER AUTOINCREMENT / SERIAL                         |
  | sample_id (FK)         : VARCHAR(100) -> Foreign Key references SAMPLES        |
  | phase_name             : VARCHAR(100) -> Standardized mineral/phase name        |
  | point_count            : INTEGER      -> Raw points attributed to this phase    |
  | modal_percent          : REAL         -> Stereological volume percentage (vol%) |
  | std_error              : REAL         -> 2-Sigma confidence interval (±%)       |
  | abundance              : VARCHAR(50)  -> Major, Minor, or Trace classification  |
  | density_gcm3           : REAL         -> Assigned specific density (g/cm³)      |
  | wt_percent             : REAL         -> Recalculated mass percentage (wt%)     |
  | en933_class            : VARCHAR(100) -> Recycled aggregate class (Rc, Ru, etc)|
  | carbonation            : VARCHAR(50)  -> Carbonation state                      |
  | liberation             : VARCHAR(50)  -> Particle liberation index              |
  +---------------------------------------------------------------------------------+
    """, language="text")

    st.markdown("---")
    st.markdown("### 👨‍💻 About the Developer")
    st.info("""
    **Developer:** **Bradley Staniforth**  
    **Application Purpose:** Designed and engineered as a comprehensive digital laboratory tool for petrographers, materials engineers, and geoscientists to streamline the ingestion, standardization, spatial indexing, regulatory classification (BS 8500-2, I.S. 465, EN 933-11), reverse-engineering, and comparative interpretation of geomaterial point count data.
    """)