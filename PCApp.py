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

st.title("🔬 Geological Petrography Analysis Platform")

SCALE_UM_PER_PX = 1.82  # 1.82 microns per pixel
DB_FILE = "petrography_data.db"

# ---------------------------------------------------------
# 0. DATABASE PERSISTENCE & AUTO-MIGRATION MODULE
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. Base Samples Table
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

    # 2. Base Modal Results Table
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
        # Sanitize any columns that could contain corrupted binary buffers
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
# PDF REPORT GENERATOR HELPER FUNCTION (REPORTLAB)
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
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1b365d')
    )
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1b365d')
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12
    )
    bold_style = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    story.append(Paragraph("Concrete Mix Reverse-Engineering & Petrographic Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 12))
    
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
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[110, 150, 110, 150])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f7fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#d0d7de')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))

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
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fbfcfd')),
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
# MAIN TAB NAVIGATION
# ---------------------------------------------------------
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
    "🔬 Sample Point Count Analysis", 
    "⚖️ Aggregate Density & DB Editor",
    "🏗️ Concrete Mix Design (DB)", 
    "📊 Multi-Sample Comparison", 
    "🌐 Geological Drive Resources",
    "📁 Local Database Browser"
])

# =========================================================
# TAB 1: CURRENT SAMPLE ANALYSIS
# =========================================================
with main_tab1:
    st.sidebar.header("1. Upload & Job Metadata")
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

        st.subheader("📋 Step 1: Ingestion & Metadata Overview")
        col1, col2, col3, col4 = st.columns(4)
        sample_id = col1.text_input("Sample ID", value=df_raw["Image_File"].iloc[0].split(".")[0])
        col2.metric("Job Number", job_number)
        col3.metric("Sample Order", sample_order)
        col4.metric("Total Gross Points", len(df_raw))

        litho_tag = ""
        if sample_type == "Concrete - I.S.465":
            litho_tag = st.selectbox("I.S. 465 Main Aggregate Lithology Tag (3-Letter Code)", IS465_LITHOLOGY_CODES)

        st.markdown("---")
        st.subheader("📐 Point Count Geometry & Traverse Metrics")

        df_raw["X_um"] = df_raw["X"] * SCALE_UM_PER_PX
        df_raw["Y_um"] = df_raw["Y"] * SCALE_UM_PER_PX

        x_min_um, x_max_um = df_raw["X_um"].min(), df_raw["X_um"].max()
        y_min_um, y_max_um = df_raw["Y_um"].min(), df_raw["Y_um"].max()

        width_um = x_max_um - x_min_um
        height_um = y_max_um - y_min_um
        width_mm = width_um / 1000.0
        height_mm = height_um / 1000.0
        area_mm2 = width_mm * height_mm

        unique_y = np.sort(df_raw["Y_um"].unique())
        unique_x = np.sort(df_raw["X_um"].unique())

        num_traverses = len(unique_y)
        num_transects = len(unique_x)

        x_spacing_um = np.mean(np.diff(unique_x)) if len(unique_x) > 1 else 0.0
        y_spacing_um = np.mean(np.diff(unique_y)) if len(unique_y) > 1 else 0.0
        total_linear_traverse_mm = num_traverses * width_mm

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("X-Span (Width)", f"{width_mm:.2f} mm")
        m_col2.metric("Y-Span (Height)", f"{height_mm:.2f} mm")
        m_col3.metric("Bounding Area", f"{area_mm2:.2f} mm²")
        m_col4.metric("Total Linear Traverse", f"{total_linear_traverse_mm:.2f} mm")

        m_col5, m_col6, m_col7, m_col8 = st.columns(4)
        m_col5.metric("Traverses (Rows)", num_traverses)
        m_col6.metric("Transects (Cols)", num_transects)
        m_col7.metric("X-Spacing", f"{x_spacing_um:.1f} μm")
        m_col8.metric("Y-Spacing", f"{y_spacing_um:.1f} μm")

        st.markdown("---")
        st.subheader("🛠️ Step 2: Quality Control (QC) & Term Mapping")
        
        unique_terms = df_raw["Raw_Term"].unique().tolist()
        lookup = DEFAULT_LOOKUPS.get(sample_type, {})
        
        mapping_dict = {}
        qc_cols = st.columns(2)
        
        for idx, term in enumerate(unique_terms):
            default_val = lookup.get(term, term)
            with qc_cols[idx % 2]:
                mapping_dict[term] = st.text_input(
                    label=f"Raw Term: **{term}**", 
                    value=default_val, 
                    key=f"map_{term}"
                )

        df_mapped = df_raw.copy()
        df_mapped["Standardized_Phase"] = df_mapped["Raw_Term"].map(mapping_dict)

        df_valid = df_mapped[df_mapped["Standardized_Phase"].str.upper() != "NA"].copy()
        valid_points_count = len(df_valid)
        na_count = len(df_mapped) - valid_points_count

        st.caption(f"ℹ️ **Exclusion Notice:** {na_count} 'NA' points excluded from modal calculations. Valid count: {valid_points_count}.")

        tagging_data = {}
        if sample_type == "Aggregate - Recycled":
            st.markdown("---")
            st.subheader("🏷️ Recycled Aggregate Phase Tagging (EN 933-11)")
            
            active_phases = df_valid["Standardized_Phase"].unique().tolist()
            for phase in active_phases:
                st.write(f"##### Phase: **{phase}**")
                t_col1, t_col2, t_col3 = st.columns(3)
                
                en_class = t_col1.selectbox(f"EN933-11 Class ({phase})", EN933_11_CLASSES, key=f"en_{phase}")
                carb_state = t_col2.selectbox(f"Carbonation ({phase})", CARBONATION_STATES, key=f"carb_{phase}")
                lib_idx = t_col3.selectbox(f"Liberation Index ({phase})", LIBERATION_INDEXES, key=f"lib_{phase}")
                
                tagging_data[phase] = {
                    "EN933_Class": en_class,
                    "Carbonation": carb_state,
                    "Liberation": lib_idx
                }

            df_valid["EN933_Class"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["EN933_Class"])
            df_valid["Carbonation"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["Carbonation"])
            df_valid["Liberation"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["Liberation"])

        st.markdown("---")
        st.subheader("📊 Step 3: Modal Analysis Results")

        summary = df_valid.groupby("Standardized_Phase").size().reset_index(name="Point_Count")
        summary["Modal_Percent (%)"] = (summary["Point_Count"] / valid_points_count) * 100
        
        p = summary["Modal_Percent (%)"] / 100
        summary["Std_Error (%)"] = 2 * np.sqrt((p * (1 - p)) / valid_points_count) * 100

        if sample_type in ["Aggregates", "Aggregate - Recycled", "Rock"]:
            summary["Abundance"] = summary["Modal_Percent (%)"].apply(classify_abundance)

        summary["Modal_Percent (%)"] = summary["Modal_Percent (%)"].round(2)
        summary["Std_Error (%)"] = summary["Std_Error (%)"].round(2)

        st.session_state["active_sample_id"] = sample_id
        st.session_state["active_summary"] = summary.copy()
        st.session_state["active_job_number"] = job_number
        st.session_state["active_sample_order"] = sample_order
        st.session_state["active_operator"] = operator

        col_tbl, col_chart = st.columns([1.2, 1])
        with col_tbl:
            st.write("##### Phase Modal Percentage Table")
            st.dataframe(summary, width="stretch")

        with col_chart:
            st.write("##### Modal Composition Chart")
            fig_bar = px.bar(
                summary, x="Standardized_Phase", y="Modal_Percent (%)", 
                text="Modal_Percent (%)", color="Standardized_Phase"
            )
            st.plotly_chart(fig_bar, width="stretch")

        # QAPF Ternary Plot
        if sample_type in ["Rock", "Aggregates"]:
            st.markdown("---")
            st.subheader("🔺 QAPF Ternary Classification")
            
            q_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Quartz", case=False), "Point_Count"].sum()
            a_cnt = summary.loc[summary["Standardized_Phase"].str.contains("K-Feldspar|Alkali", case=False), "Point_Count"].sum()
            p_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Plagioclase", case=False), "Point_Count"].sum()
            
            qap_sum = q_cnt + a_cnt + p_cnt
            if qap_sum > 0:
                norm_q = (q_cnt / qap_sum) * 100
                norm_a = (a_cnt / qap_sum) * 100
                norm_p = (p_cnt / qap_sum) * 100

                fig_ternary = go.Figure(go.Scatterternary({
                    'mode': 'markers',
                    'a': [norm_q],
                    'b': [norm_a],
                    'c': [norm_p],
                    'text': [f"Sample: {sample_id}"],
                    'marker': {'symbol': 'circle', 'size': 14, 'color': 'red'}
                }))
                fig_ternary.update_layout({
                    'ternary': {
                        'sum': 100,
                        'aaxis': {'title': 'Quartz (Q)'},
                        'baxis': {'title': 'Alkali Feldspar (A)'},
                        'caxis': {'title': 'Plagioclase (P)'}
                    },
                    'title': 'Q-A-P Ternary Plot'
                })
                st.plotly_chart(fig_ternary, width="stretch")

        # Concrete I.S. 465 Aggregate Derivations
        if sample_type == "Concrete - I.S.465":
            st.markdown("---")
            st.subheader("🧱 Concrete - I.S. 465 Aggregate Derivations")
            
            good_agg_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Good Aggregate", case=False), "Point_Count"].sum()
            bad_agg_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Bad Aggregate", case=False), "Point_Count"].sum()
            total_agg_cnt = good_agg_cnt + bad_agg_cnt

            if total_agg_cnt > 0:
                total_agg_pct = (total_agg_cnt / valid_points_count) * 100
                pct_good_of_agg = (good_agg_cnt / total_agg_cnt) * 100
                pct_bad_of_agg = (bad_agg_cnt / total_agg_cnt) * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Overall Aggregate Content", f"{total_agg_pct:.2f} %")
                m2.metric("% Good Aggregate (of Total Agg)", f"{pct_good_of_agg:.2f} %")
                m3.metric("% Bad Aggregate (of Total Agg)", f"{pct_bad_of_agg:.2f} %")
            else:
                st.warning("No points mapped to 'Good Aggregate' or 'Bad Aggregate' were found in this sample.")

        # Aggregate Density, wt%, and BS 8500 ASR Module
        if sample_type in ["Aggregates", "Aggregate - Recycled"]:
            st.markdown("---")
            st.subheader("⚖️ Aggregate Density, Weight % (wt%), & BS 8500-2 ASR Risk")

            agg_tab1, agg_tab2 = st.tabs(["Phase Densities & wt% Recalculation", "BS 8500-2 Annex C ASR Assessment"])

            with agg_tab1:
                st.write("Enter density values (g/cm³) to calculate Weight Percentages (wt%):")
                densities = []
                d_cols = st.columns(len(summary))
                
                for idx, row in summary.iterrows():
                    phase_n = row["Standardized_Phase"]
                    def_d = DEFAULT_DENSITIES.get(phase_n, 2.65)
                    with d_cols[idx % len(d_cols)]:
                        d_val = st.number_input(f"Density: {phase_n}", value=def_d, step=0.05, key=f"d_{phase_n}")
                        densities.append(d_val)
                
                summary["Density (g/cm³)"] = densities
                mass_factor = summary["Modal_Percent (%)"] * summary["Density (g/cm³)"]
                summary["Weight_Percent (%)"] = (mass_factor / mass_factor.sum()) * 100
                summary["Weight_Percent (%)"] = summary["Weight_Percent (%)"].round(2)

                st.dataframe(summary[["Standardized_Phase", "Point_Count", "Modal_Percent (%)", "Density (g/cm³)", "Weight_Percent (%)"]], width="stretch")

            with agg_tab2:
                asr_risk, asr_desc = evaluate_asr_risk(summary)
                st.metric("BS 8500-2 Annex C ASR Risk Rating", asr_risk)
                st.info(asr_desc)

        # Spatial Scatter Map & Dispersion
        st.markdown("---")
        st.subheader("📍 Thin Section Point Spatial Distribution & Index")

        c_map, c_index = st.columns([2, 1])

        with c_map:
            fig_scatter = px.scatter(
                df_valid, x="X_um", y="Y_um", color="Standardized_Phase",
                title="Spatial Point Distribution Across Thin Section (μm)"
            )
            fig_scatter.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_scatter, width="stretch")

        with c_index:
            st.write("##### Spatial Dispersion Coefficient")
            R_val, R_desc = calculate_spatial_index(df_valid)
            if R_val is not None:
                st.metric("Nearest Neighbor Ratio (R)", f"{R_val:.3f}")
                st.info(f"**Interpretation:** {R_desc}")

        # Step 5: Save & Export Section
        st.markdown("---")
        st.subheader("💾 Step 5: Save & Export")
        
        col_save, col_dl = st.columns(2)
        
        if col_save.button("✅ Confirm QC & Save to Local Database", type="primary"):
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
                tag_df = pd.DataFrame.from_dict(tagging_data, orient='index').reset_index()
                tag_df = tag_df.rename(columns={'index': 'Standardized_Phase'})
                summary = summary.merge(tag_df, on="Standardized_Phase", how="left")
                
            save_sample_to_db(meta_payload, summary)
            st.success(f"Sample '{sample_id}' successfully saved to database!")

        csv_data = summary.to_csv(index=False).encode('utf-8')
        col_dl.download_button(
            label="📥 Download Interpreted Modal Report (CSV)",
            data=csv_data,
            file_name=f"{job_number}_{sample_id}_modal_report.csv",
            mime="text/csv"
        )
    else:
        st.info("👈 Please upload a point count CSV/TSV file from the sidebar to begin analysis.")

# =========================================================
# TAB 2: AGGREGATE DENSITY & DB RE-EVALUATION EDITOR
# =========================================================
with main_tab2:
    st.header("⚖️ Database Aggregate Density & wt% Editor")
    st.info("Load any previously saved sample from the database to inspect, reassign density values (g/cm³), and update weight percentages (wt%).")

    df_db_samples = load_all_samples()
    if not df_db_samples.empty:
        selected_db_id = st.selectbox("Select Saved Sample to Edit/Re-evaluate", df_db_samples["sample_id"].unique(), key="db_edit_sel")
        
        if selected_db_id:
            sample_row = df_db_samples[df_db_samples["sample_id"] == selected_db_id].iloc[0]
            df_details = load_sample_details(selected_db_id)
            
            st.markdown("---")
            st.write(f"##### Selected Sample: **{selected_db_id}** | Job: **{sample_row['job_number']}** | Type: **{sample_row['sample_type']}**")

            st.write("Reassign densities (g/cm³) for each phase:")
            edited_densities = []
            
            ed_cols = st.columns(len(df_details))
            for idx, r in df_details.iterrows():
                p_name = r["phase_name"]
                cur_d = r["density_gcm3"] if pd.notnull(r["density_gcm3"]) else DEFAULT_DENSITIES.get(p_name, 2.65)
                with ed_cols[idx % len(ed_cols)]:
                    new_d = st.number_input(f"Density: {p_name}", value=float(cur_d), step=0.05, key=f"db_d_{selected_db_id}_{p_name}")
                    edited_densities.append(new_d)
                    
            df_details["density_gcm3"] = edited_densities
            mass_f = df_details["modal_percent"] * df_details["density_gcm3"]
            df_details["wt_percent"] = (mass_f / mass_f.sum()) * 100
            df_details["wt_percent"] = df_details["wt_percent"].round(2)
            
            st.write("##### Updated Modal & Weight Percentage Table")
            st.dataframe(df_details[["phase_name", "point_count", "modal_percent", "density_gcm3", "wt_percent", "abundance"]], width="stretch")

            # Re-evaluate BS 8500 ASR Risk
            df_details_eval = df_details.rename(columns={"phase_name": "Standardized_Phase", "modal_percent": "Modal_Percent (%)"})
            re_asr_risk, re_asr_desc = evaluate_asr_risk(df_details_eval)
            st.metric("Re-evaluated BS 8500-2 Annex C ASR Risk", re_asr_risk)
            st.info(re_asr_desc)

            if st.button("💾 Overwrite & Save Updated Densities to DB", type="primary"):
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
                st.success(f"Sample '{selected_db_id}' updated in database with new density & wt% calculations!")
    else:
        st.info("No saved samples available in the database to edit yet.")

# =========================================================
# TAB 3: CONCRETE MIX DESIGN ENGINE (APPROACH B & A)
# =========================================================
with main_tab3:
    st.header("🏗️ Hardened Concrete Mix Reverse-Engineering & Performance Engine")
    st.info("Back-calculate original concrete mix proportions ($1\\text{ m}^3$), 28-day cube strength (UK DoE 1B), and fresh slump using stereological modal points.")

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
        st.subheader("1. Component Classification & Physical Densities")
        
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
                    f"Role: **{p_name}** ({row['Vol_Pct']:.1f}%)",
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

        st.write("##### Consolidated Volumetric Breakdown (Area % = Volume %)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Cement Paste Volume", f"{v_paste:.2f} %")
        b2.metric("Fine Aggregate Volume", f"{v_fine:.2f} %")
        b3.metric("Coarse Aggregate Volume", f"{v_coarse:.2f} %")
        b4.metric("Macroscopic Air Voids", f"{v_void:.2f} %")

        st.markdown("---")
        st.subheader("2. Petrographic Mix Inputs & Reverse-Engineering Model")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            wb_ratio = st.slider(
                "Petrographically Estimated Water/Binder Ratio (w/b):",
                min_value=0.25, max_value=0.80, value=0.48, step=0.01,
                help="Derived from fluorescent thin section dye intensity against reference paste standards and capillary porosity."
            )
            calc_approach = st.radio(
                "Calculation Formulation:",
                ["Approach B: Mass Balance (BS 1881-124 / ASTM C856) [Default]", 
                 "Approach A: Physical Hydration (Powers-Brownyard)"],
                index=0
            )

        with p_col2:
            scm_type = st.selectbox(
                "Binder Composition (SCM Replacement):",
                ["100% CEM I (Portland)", "30% GGBS (Slag)", "25% PFA (Fly Ash)", "7% Silica Fume"]
            )
            d_col1, d_col2, d_col3 = st.columns(3)
            
            if "GGBS" in scm_type:
                def_rho_b = 3.07
            elif "PFA" in scm_type:
                def_rho_b = 2.94
            elif "Silica Fume" in scm_type:
                def_rho_b = 3.08
            else:
                def_rho_b = 3.15

            rho_cement = float(d_col1.number_input("Binder Density (g/cm³)", value=float(def_rho_b), step=0.01))
            rho_fine = float(d_col2.number_input("Fine Agg Density (g/cm³)", value=2.65, step=0.01))
            rho_coarse = float(d_col3.number_input("Coarse Agg Density (g/cm³)", value=2.68, step=0.01))

        # ---------------------------------------------------------
        # MIX RECONSTRUCTION CALCULATIONS
        # ---------------------------------------------------------
        rho_water = 1.00  # g/cm³ (= 1000 kg/m³ or 1.0 kg/L)
        v_total_agg = v_fine + v_coarse

        # APPROACH B: Absolute Volume Conservation (BS 1881-124)
        v_paste_avail_m3 = max(0.05, 1.0 - ((v_total_agg + v_void) / 100.0))
        denom_b = (1.0 / (rho_cement * 1000.0)) + (wb_ratio / (rho_water * 1000.0))
        c_kg_b = v_paste_avail_m3 / denom_b
        w_kg_b = c_kg_b * wb_ratio
        fine_kg_b = 10.0 * v_fine * rho_fine
        coarse_kg_b = 10.0 * v_coarse * rho_coarse
        fresh_density_b = c_kg_b + w_kg_b + fine_kg_b + coarse_kg_b

        # APPROACH A: Powers-Brownyard Physical Hydration Stoichiometry
        v_c0_pct = v_paste / (1.0 + wb_ratio * (rho_cement / rho_water)) if (1.0 + wb_ratio * (rho_cement / rho_water)) > 0 else 0
        c_kg_a = 10.0 * v_c0_pct * rho_cement
        w_kg_a = c_kg_a * wb_ratio
        fine_kg_a = 10.0 * v_fine * rho_fine
        coarse_kg_a = 10.0 * v_coarse * rho_coarse
        fresh_density_a = c_kg_a + w_kg_a + fine_kg_a + coarse_kg_a

        # Select values based on user toggle
        is_approach_b = "Approach B" in calc_approach
        c_kg = c_kg_b if is_approach_b else c_kg_a
        w_kg = w_kg_b if is_approach_b else w_kg_a
        fine_kg = fine_kg_b if is_approach_b else fine_kg_a
        coarse_kg = coarse_kg_b if is_approach_b else coarse_kg_a
        fresh_density = fresh_density_b if is_approach_b else fresh_density_a

        # Batch Volumes (L/m³)
        v_c_L = c_kg / rho_cement
        v_w_L = w_kg / rho_water
        v_fine_L = fine_kg / rho_fine
        v_coarse_L = coarse_kg / rho_coarse
        v_air_L = v_void * 10.0
        total_batch_vol_L = v_c_L + v_w_L + v_fine_L + v_coarse_L + v_air_L

        # Determined Mass % (wt%) & Volumetric % Content
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

        # Mix Proportions 1 : X
        prop_c = 1.0
        prop_fine = fine_kg / c_kg if c_kg > 0 else 0.0
        prop_coarse = coarse_kg / c_kg if c_kg > 0 else 0.0
        total_agg_ratio = prop_fine + prop_coarse

        st.markdown("---")
        st.subheader("3. Reconstructed Mix Proportions & Determined Content")

        df_mix_res = pd.DataFrame({
            "Constituent": ["Binder (Cementitious)", "Water", "Fine Aggregate", "Coarse Aggregate", "Air Voids"],
            "Batch Weight (kg/m³)": [f"{c_kg:.1f}", f"{w_kg:.1f}", f"{fine_kg:.1f}", f"{coarse_kg:.1f}", "0.0"],
            "Determined Mass % (wt%)": [f"{c_wt_pct:.1f}%", f"{w_wt_pct:.1f}%", f"{fine_wt_pct:.1f}%", f"{coarse_wt_pct:.1f}%", "0.0%"],
            "Batch Volume (L/m³)": [f"{v_c_L:.1f}", f"{v_w_L:.1f}", f"{v_fine_L:.1f}", f"{v_coarse_L:.1f}", f"{v_air_L:.1f}"],
            "Determined Vol %": [f"{c_vol_pct:.1f}%", f"{w_vol_pct:.1f}%", f"{fine_vol_pct:.1f}%", f"{coarse_vol_pct:.1f}%", f"{air_vol_pct:.1f}%"],
            "Mix Proportion (1 : X)": [
                "1.00",
                f"{wb_ratio:.2f} (w/b)",
                f"{prop_fine:.2f}",
                f"{prop_coarse:.2f}",
                f"{v_void:.1f}% Air"
            ]
        })

        col_res1, col_res2 = st.columns([1.6, 1])
        with col_res1:
            st.dataframe(df_mix_res, width="stretch")

        with col_res2:
            st.metric("Total Fresh Density", f"{fresh_density:.0f} kg/m³")
            st.metric("Total Aggregate Content", f"{total_agg_wt_pct:.1f} wt% ({v_total_agg:.1f} vol%)")
            st.metric("Binder Content", f"{c_wt_pct:.1f} wt% ({c_kg:.0f} kg/m³)")
            st.metric("Water Content", f"{w_wt_pct:.1f} wt% ({w_kg:.0f} kg/m³)")

        # ---------------------------------------------------------
        # GRAPHICAL 100% STACKED BAR CHARTS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Mix Design Composition Visualisation")

        col_chart_m, col_chart_v = st.columns(2)

        with col_chart_m:
            df_stacked_mass = pd.DataFrame([
                {"Category": "Total Mass %", "Constituent": "Binder (Cement)", "Percentage": c_wt_pct},
                {"Category": "Total Mass %", "Constituent": "Water", "Percentage": w_wt_pct},
                {"Category": "Total Mass %", "Constituent": "Fine Aggregate", "Percentage": fine_wt_pct},
                {"Category": "Total Mass %", "Constituent": "Coarse Aggregate", "Percentage": coarse_wt_pct}
            ])
            fig_mass = px.bar(
                df_stacked_mass, y="Category", x="Percentage", color="Constituent",
                orientation='h', barmode='stack', text="Percentage",
                title="100% Stacked Chart: Batch Mass (wt%)",
                color_discrete_map={
                    "Binder (Cement)": "#3366cc",
                    "Water": "#22aa99",
                    "Fine Aggregate": "#ff9900",
                    "Coarse Aggregate": "#dc3912"
                }
            )
            fig_mass.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
            fig_mass.update_layout(xaxis=dict(ticksuffix="%", range=[0, 100]), yaxis_title=None, height=260)
            st.plotly_chart(fig_mass, width="stretch")

        with col_chart_v:
            df_stacked_vol = pd.DataFrame([
                {"Category": "Total Vol %", "Constituent": "Binder", "Percentage": c_vol_pct},
                {"Category": "Total Vol %", "Constituent": "Water", "Percentage": w_vol_pct},
                {"Category": "Total Vol %", "Constituent": "Fine Aggregate", "Percentage": fine_vol_pct},
                {"Category": "Total Vol %", "Constituent": "Coarse Aggregate", "Percentage": coarse_vol_pct},
                {"Category": "Total Vol %", "Constituent": "Air Voids", "Percentage": air_vol_pct}
            ])
            fig_vol = px.bar(
                df_stacked_vol, y="Category", x="Percentage", color="Constituent",
                orientation='h', barmode='stack', text="Percentage",
                title="100% Stacked Chart: Volumetric Distribution (vol%)",
                color_discrete_map={
                    "Binder": "#3366cc",
                    "Water": "#22aa99",
                    "Fine Aggregate": "#ff9900",
                    "Coarse Aggregate": "#dc3912",
                    "Air Voids": "#999999"
                }
            )
            fig_vol.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
            fig_vol.update_layout(xaxis=dict(ticksuffix="%", range=[0, 100]), yaxis_title=None, height=260)
            st.plotly_chart(fig_vol, width="stretch")

        # ---------------------------------------------------------
        # 4. PERFORMANCE & PROPERTY ESTIMATION
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("4. Engineering Performance & Workability Estimator")

        p_tab1, p_tab2 = st.tabs(["Estimated Cube Strength (DoE 1B)", "Original Slump & Admixture 'What-If'"])

        with p_tab1:
            st.write("##### 28-Day Compressive Cube Strength (UK DoE / BRE Option 1B)")
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                cement_grade = st.selectbox(
                    "Cement Strength Class:",
                    ["CEM I 52.5N / 52.5R (High Early Strength)", 
                     "CEM I 42.5N / 42.5R (Standard Portland)", 
                     "CEM II 32.5N (Blended/Masonry)"]
                )
                coarse_type = st.selectbox("Aggregate Lithological Surface Type:", ["Crushed Rock (Basalt/Limestone/Granite)", "Uncrushed Natural River Gravel"])

            if "52.5" in cement_grade:
                base_f = 46.0 if "Crushed" in coarse_type else 42.0
            elif "42.5" in cement_grade:
                base_f = 39.0 if "Crushed" in coarse_type else 35.0
            else:
                base_f = 31.0 if "Crushed" in coarse_type else 27.0

            calc_fcu_28 = base_f - 24.0 * np.log(max(0.20, wb_ratio) / 0.50)

            air_excess = max(0.0, v_void - 1.0)
            air_penalty_factor = max(0.50, 1.0 - (0.055 * air_excess))
            final_fcu_28 = max(5.0, calc_fcu_28 * air_penalty_factor)
            ci_low = max(5.0, final_fcu_28 - 4.5)
            ci_high = final_fcu_28 + 4.5

            with s_col2:
                st.metric("Estimated 28-Day Cube Strength (fcu, 28)", f"{final_fcu_28:.1f} MPa")
                st.info(f"**95% Confidence Interval:** {ci_low:.1f} – {ci_high:.1f} MPa\n\n*Applied Air Void Penalty:* {((1.0 - air_penalty_factor)*100):.1f}% reduction due to {v_void:.1f}% voids.")

        with p_tab2:
            st.write("##### Original Fresh Slump & Admixture 'What-If' Selector")
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                nom_max_agg = st.selectbox("Coarse Aggregate Maximum Size:", ["20 mm (Standard)", "10 mm", "40 mm"])
                agg_angularity = st.selectbox("Particle Shape / Angularity:", ["Angular / Crushed Rock", "Sub-rounded / Irregular", "Rounded River Gravel", "Flaky / Elongated"])
                admixture_scenario = st.selectbox(
                    "Admixture 'What-If' Scenario:",
                    ["Plain Concrete (No Water Reducer, beta = 1.00)",
                     "Normal Water Reducer / Lignosulfonate (8-12% water reduction, beta = 0.90)",
                     "Mid-Range Plasticizer / SNF (15-20% water reduction, beta = 0.82)",
                     "High-Range Superplasticizer / PCE (25-35% water reduction, beta = 0.70)"]
                )

            if "10 mm" in nom_max_agg:
                w_base = 230.0
            elif "40 mm" in nom_max_agg:
                w_base = 190.0
            else:
                w_base = 210.0

            if "Rounded" in agg_angularity:
                w_shape = -20.0
            elif "Sub-rounded" in agg_angularity:
                w_shape = -10.0
            elif "Flaky" in agg_angularity:
                w_shape = +15.0
            else:
                w_shape = 0.0

            p_sand = (v_fine / v_total_agg * 100.0) if v_total_agg > 0 else 35.0
            w_sand = 0.5 * (p_sand - 35.0)

            if "Lignosulfonate" in admixture_scenario:
                beta_admix = 0.90
            elif "Mid-Range" in admixture_scenario:
                beta_admix = 0.82
            elif "High-Range" in admixture_scenario:
                beta_admix = 0.70
            else:
                beta_admix = 1.00

            effective_target_water = (w_base + w_shape + w_sand) * beta_admix
            delta_w = w_kg - effective_target_water
            est_slump = 50.0 + (2.5 * delta_w)

            if est_slump <= 25:
                slump_class = "S1 (Very Stiff / Semi-Dry, 10 - 40 mm)"
            elif 25 < est_slump <= 90:
                slump_class = "S2 (Medium Workability, 50 - 90 mm)"
            elif 90 < est_slump <= 150:
                slump_class = "S3 (High Workability, 100 - 150 mm)"
            elif 150 < est_slump <= 210:
                slump_class = "S4 (Very High Workability, 160 - 210 mm)"
            else:
                slump_class = "S5 / Collapse (> 220 mm / Flowing Concrete)"

            est_slump_clamped = max(5.0, min(250.0, est_slump))

            with w_col2:
                st.metric("Estimated Fresh Slump", f"{est_slump_clamped:.0f} mm")
                st.info(f"**Workability Class:** {slump_class}\n\n*Reference Water Demand:* {effective_target_water:.1f} kg/m³ vs *Actual Reconstructed:* {w_kg:.1f} kg/m³.")

        # ---------------------------------------------------------
        # 5. PDF REPORT EXPORT SECTION
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📑 5. Export Mix Design Report (PDF)")
        
        pdf_meta = {
            "sample_id": mix_sample_label,
            "job_number": cur_job_num,
            "sample_order": cur_sample_ord,
            "operator": cur_op_name,
            "approach": "Approach B (Mass Balance BS 1881-124)" if is_approach_b else "Approach A (Powers-Brownyard)",
            "wb_ratio": wb_ratio,
            "fresh_density": fresh_density
        }
        
        pdf_strength = {
            "strength": final_fcu_28,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "air_penalty": (1.0 - air_penalty_factor) * 100.0,
            "air_pct": v_void
        }
        
        pdf_slump = {
            "slump": est_slump_clamped,
            "class": slump_class
        }
        
        pdf_mass_pcts = {
            "Binder": c_wt_pct,
            "Water": w_wt_pct,
            "Fine Agg": fine_wt_pct,
            "Coarse Agg": coarse_wt_pct
        }
        
        try:
            pdf_bytes = generate_mix_design_pdf(
                meta_dict=pdf_meta,
                df_results=df_mix_res,
                strength_dict=pdf_strength,
                slump_dict=pdf_slump,
                mass_pcts=pdf_mass_pcts
            )
            
            st.download_button(
                label="📥 Download Concrete Mix Design Report (PDF)",
                data=pdf_bytes,
                file_name=f"{cur_job_num}_{mix_sample_label}_mix_design_report.pdf",
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
        st.caption("⚙️ Multi-sample trend plots, ternary compilation diagrams, and aggregate variability tables will render here.")
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
    st.caption("⚙️ PDF reference viewer and aggregate lithology optical reference photos will load here.")

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
        