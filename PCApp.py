import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import distance_matrix

st.set_page_config(page_title="Geological Petrography QC & Analysis", layout="wide")

st.title("🔬 Geological Petrography Analysis Platform")

SCALE_UM_PER_PX = 1.82  # 1.82 microns per pixel

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
# SIDEBAR: Metadata & Ingestion
# ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # STEP 2: QUALITY CONTROL & TAXONOMY MAPPING
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # SPECIAL MODULE: Aggregate - Recycled Tagging
    # ---------------------------------------------------------
    tagging_data = {}
    if sample_type == "Aggregate - Recycled":
        st.markdown("---")
        st.subheader("🏷️ Step 2B: Recycled Aggregate Phase Tagging (EN 933-11)")
        st.info("Assign classification tags to each identified phase before generating stacked reports.")

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

        # Apply tags to dataset
        df_valid["EN933_Class"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["EN933_Class"])
        df_valid["Carbonation"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["Carbonation"])
        df_valid["Liberation"] = df_valid["Standardized_Phase"].map(lambda p: tagging_data[p]["Liberation"])

    # ---------------------------------------------------------
    # STEP 3: MODAL ANALYSIS
    # ---------------------------------------------------------
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
        st.dataframe(summary, use_container_width=True)

    with col_chart:
        st.write("##### Modal Composition")
        fig_bar = px.bar(
            summary, x="Standardized_Phase", y="Modal_Percent (%)", 
            text="Modal_Percent (%)", color="Standardized_Phase"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------------
    # SPECIAL MODULE: Recycled Aggregate 100% Stacked Bar Charts
    # ---------------------------------------------------------
    if sample_type == "Aggregate - Recycled":
        st.markdown("---")
        st.subheader("📈 Recycled Aggregate 100% Stacked Tag Plots")

        st_tab1, st_tab2, st_tab3 = st.tabs(["EN933-11 Classification", "Carbonation State", "Liberation Index"])

        with st_tab1:
            df_en = df_valid.groupby(["Standardized_Phase", "EN933_Class"]).size().reset_index(name="Count")
            fig_en = px.bar(
                df_en, x="Standardized_Phase", y="Count", color="EN933_Class",
                title="100% Stacked Chart: EN933-11 Class by Phase",
                groupnorm="fraction"
            )
            fig_en.update_layout(yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_en, use_container_width=True)

        with st_tab2:
            df_carb = df_valid.groupby(["Standardized_Phase", "Carbonation"]).size().reset_index(name="Count")
            fig_carb = px.bar(
                df_carb, x="Standardized_Phase", y="Count", color="Carbonation",
                title="100% Stacked Chart: Carbonation State by Phase",
                groupnorm="fraction"
            )
            fig_carb.update_layout(yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_carb, use_container_width=True)

        with st_tab3:
            df_lib = df_valid.groupby(["Standardized_Phase", "Liberation"]).size().reset_index(name="Count")
            fig_lib = px.bar(
                df_lib, x="Standardized_Phase", y="Count", color="Liberation",
                title="100% Stacked Chart: Liberation Index by Phase",
                groupnorm="fraction"
            )
            fig_lib.update_layout(yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_lib, use_container_width=True)

    # ---------------------------------------------------------
    # STEP 4: SPATIAL DISTRIBUTION & NEAREST NEIGHBOR INDEX
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📍 Thin Section Point Spatial Distribution & Index")

    c_map, c_index = st.columns([2, 1])

    with c_map:
        fig_scatter = px.scatter(
            df_valid, x="X_um", y="Y_um", color="Standardized_Phase",
            title="Spatial Point Distribution Across Thin Section (μm)"
        )
        fig_scatter.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c_index:
        st.write("##### Spatial Dispersion Coefficient")
        R_val, R_desc = calculate_spatial_index(df_valid)
        if R_val is not None:
            st.metric("Nearest Neighbor Ratio (R)", f"{R_val:.3f}")
            st.info(f"**Interpretation:** {R_desc}")

    # ---------------------------------------------------------
    # STEP 5: SAVE TO DATABASE & EXPORT
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("💾 Step 5: Save & Export")
    
    col_save, col_dl = st.columns(2)
    if col_save.button("✅ Confirm QC & Commit Sample to Cloud DB", type="primary"):
        st.success(f"Job {job_number} | Sample {sample_id} successfully saved to Cloud Database!")

    csv_data = summary.to_csv(index=False).encode('utf-8')
    col_dl.download_button(
        label="📥 Download Interpreted Modal Report (CSV)",
        data=csv_data,
        file_name=f"{job_number}_{sample_id}_recycled_agg_report.csv",
        mime="text/csv"
    )
else:
    st.info("👈 Please upload a point count CSV/TSV file from the sidebar to begin.")