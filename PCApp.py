import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist  # Replaces distance_matrix to prevent SciPy deprecation warnings

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
    
    # Auto-migrate 'samples' table if missing 'lithology_code'
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
    
    # Auto-migrate 'modal_results' table if missing density columns
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

def save_sample_to_db(sample_meta, df_summary):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO samples 
        (sample_id, job_number, sample_order, sample_type, operator, total_points, valid_points, bounding_area_mm2, linear_traverse_mm, lithology_code)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sample_meta["sample_id"], sample_meta["job_number"], sample_meta["sample_order"],
        sample_meta["sample_type"], sample_meta["operator"], sample_meta["total_points"],
        sample_meta["valid_points"], sample_meta["area_mm2"], sample_meta["linear_traverse_mm"],
        sample_meta.get("lithology_code", "")
    ))
    
    cursor.execute("DELETE FROM modal_results WHERE sample_id = ?", (sample_meta["sample_id"],))
    
    for _, row in df_summary.iterrows():
        cursor.execute("""
            INSERT INTO modal_results 
            (sample_id, phase_name, point_count, modal_percent, std_error, abundance, density_gcm3, wt_percent, en933_class, carbonation, liberation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_meta["sample_id"],
            row.get("Standardized_Phase", ""),
            int(row.get("Point_Count", 0)),
            float(row.get("Modal_Percent (%)", 0.0)),
            float(row.get("Std_Error (%)", 0.0)),
            row.get("Abundance", None),
            float(row.get("Density (g/cm³)", 2.65)) if pd.notnull(row.get("Density (g/cm³)")) else None,
            float(row.get("Weight_Percent (%)", 0.0)) if pd.notnull(row.get("Weight_Percent (%)")) else None,
            row.get("EN933_Class", None),
            row.get("Carbonation", None),
            row.get("Liberation", None)
        ))
        
    conn.commit()
    conn.close()

def load_all_samples():
    conn = sqlite3.connect(DB_FILE)
    df_samples = pd.read_sql_query("SELECT * FROM samples ORDER BY created_at DESC", conn)
    conn.close()
    return df_samples

def load_sample_details(sample_id):
    conn = sqlite3.connect(DB_FILE)
    df_results = pd.read_sql_query("SELECT * FROM modal_results WHERE sample_id = ?", conn, params=(sample_id,))
    conn.close()
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