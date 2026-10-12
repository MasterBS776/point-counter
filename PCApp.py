import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import distance_matrix

st.set_page_config(page_title="Geological Petrography QC & Analysis", layout="wide")

st.title("🔬 Geological Petrography Analysis Platform")

SCALE_UM_PER_PX = 1.82  # 1.82 microns per pixel
DB_FILE = "petrography_data.db"

# ---------------------------------------------------------
# 0. DATABASE PERSISTENCE MODULE (SQLITE)
# ---------------------------------------------------------
def init_db():
    """Initialize local SQLite database tables if they do not exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. Samples Master Table
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
    
    # 2. Modal Percentage Results Table
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
    conn.commit()
    conn.close()

init_db()

def save_sample_to_db(sample_meta, df_summary):
    """Save sample metadata and modal results to local SQLite DB."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Insert or Replace Sample Metadata
    cursor.execute("""
        INSERT OR REPLACE INTO samples 
        (sample_id, job_number, sample_order, sample_type, operator, total_points, valid_points, bounding_area_mm2, linear_traverse_mm)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sample_meta["sample_id"], sample_meta["job_number"], sample_meta["sample_order"],
        sample_meta["sample_type"], sample_meta["operator"], sample_meta["total_points"],
        sample_meta["valid_points"], sample_meta["area_mm2"], sample_meta["linear_traverse_mm"]
    ))
    
    # Clean old results for this sample ID if re-saving
    cursor.execute("DELETE FROM modal_results WHERE sample_id = ?", (sample_meta["sample_id"],))
    
    # Insert Modal Results
    for _, row in df_summary.iterrows():
        cursor.execute("""
            INSERT INTO modal_results 
            (sample_id, phase_name, point_count, modal_percent, std_error, abundance, en933_class, carbonation, liberation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_meta["sample_id"],
            row.get("Standardized_Phase", ""),
            int(row.get("Point_Count", 0)),
            float(row.get("Modal_Percent (%)", 0.0)),
            float(row.get("Std_Error (%)", 0.0)),
            row.get("Abundance", None),
            row.get("EN933_Class", None),
            row.get("Carbonation", None),
            row.get("Liberation", None)
        ))
        
    conn.commit()
    conn.close()

def load_all_samples():
    """Retrieve summary of all locally saved samples."""
    conn = sqlite3.connect(DB_FILE)
    df_samples = pd.read_sql_query("SELECT * FROM samples ORDER BY created_at DESC", conn)
    conn.close()
    return df_samples

def load_sample_details(sample_id):
    """Retrieve modal analysis details for a selected sample."""
    conn = sqlite3.connect(DB_FILE)
    df_results = pd.read_sql_query("SELECT * FROM modal_results WHERE sample_id = ?", conn, params=(sample_id,))
    conn.close()
    return df_results

# ---------------------------------------------------------
# 1. SAMPLE TYPES & LOOKUPS
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

EN933_11_CLASSES = [
    "Rc - Concrete & Mortar",
    "Ru - Unbound / Natural Stone",
    "Rb - Clay Masonry & Bricks",
    "Ra - Bituminous / Asphalt",
    "FL - Floating Material (Wood/Plastic)",
    "X - Other Contaminants (Glass/Metal/Soil)"
]

CARBONATION_STATES = ["Uncarbonated", "Carbonated", "Non-cementitious"]
LIBERATION_INDEXES = [
    "I - Liberated",
    "II - Partly locked",
    "III - Fully locked",
    "IV - Paste only"
]

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

def classify_abundance(pct):
    if pct > 50.0:
        return "Major+"
    elif 10.0 <= pct <= 50.0:
        return "Major"
    elif 2.0 <= pct < 10.0:
        return "Minor"
    else:
        return "Trace"

def calculate_spatial_index(df_points):
    if len(df_points) < 5:
        return None, "Insufficient points"
    
    coords = df_points[["X", "Y"]].values
    dist_matrix = distance_matrix(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)
    
    nearest_dists = np.min(dist_matrix, axis=1)
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
# MAIN TAB NAVIGATION
# ---------------------------------------------------------
main_tab1, main_tab2 = st.tabs(["🔬 Current Sample Analysis", "📁 Local Database Browser"])

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

        # ---------------------------------------------------------
        # POINT COUNT GEOMETRY & TRAVERSE METRICS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📐 Point Count Geometry & Traverse Metrics")

        df_raw["X_um"] = df_raw["X"] * SCALE_UM_PER_PX
        df_raw["Y_um"] = df_raw["Y"] * SCALE_UM_PER_PX

        x_min_um, x_max_um = df_raw["X_um"].min(), df_raw["X_um"].max()
        y_min_um, y_max_um = df_raw["Y_um"].min(), df_raw["Y_um"].max()

        width_um = x_max_um - x_