import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist  # Replaces distance_matrix to avoid SciPy deprecation warnings

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
    
    high_pct = 0.0
    norm_pct = 0.0
    
    for _, row in df_summary.iterrows():
        p_name = str(row["Standardized_Phase"]).lower()
        pct = row["Modal_Percent (%)"]
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
# MAIN TAB NAVIGATION
# ---------------------------------------------------------
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
    "🔬 Sample Point Count Analysis", 
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

        col_tbl, col_chart = st.columns([1.2, 1])
        with col_tbl:
            st.write("##### Phase Modal Percentage Table")
            st.dataframe(summary, width="stretch")

        with col_chart:
            st.write("##### Modal Composition")
            fig_bar = px.bar(
                summary, x="Standardized_Phase", y="Modal_Percent (%)", 
                text="Modal_Percent (%)", color="Standardized_Phase"
            )
            st.plotly_chart(fig_bar, width="stretch")

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

        st.markdown("---")
        st.subheader("💾 Step 5: Save & Export")
        
        col_save, col_dl = st.columns(2)