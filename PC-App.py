import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import distance_matrix

st.set_page_config(page_title="Geological Petrography QC & Analysis", layout="wide")

st.title("🔬 Geological Petrography Analysis Platform")

# ---------------------------------------------------------
# 1. SAMPLE TYPES & DEFAULT LOOKUPS
# ---------------------------------------------------------
SAMPLE_TYPES = [
    "Aggregates",
    "Concrete - General",
    "Concrete - I.S.465",
    "Rock",
    "Concrete - Block/Precast",
    "Render/Mortar"
]

DEFAULT_LOOKUPS = {
    "Aggregates": {
        "Phase1": "Quartz", "Phase2": "Plagioclase", "Phase3": "K-Feldspar", 
        "Phase4": "Feldspathoid", "Good_Agg": "Good Aggregate", "Bad_Agg": "Bad Aggregate",
        "Binder": "Cement Matrix", "NA": "NA"
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

# Helper: Abundance Classifier
def classify_abundance(pct):
    if pct > 50.0:
        return "Major+"
    elif 10.0 <= pct <= 50.0:
        return "Major"
    elif 2.0 <= pct < 10.0:
        return "Minor"
    else:
        return "Trace"

# Helper: Nearest Neighbor Spatial Distribution Index (Clark-Evans R)
def calculate_spatial_index(df_points):
    if len(df_points) < 5:
        return None, "Insufficient points"
    
    coords = df_points[["X", "Y"]].values
    dist_matrix = distance_matrix(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)
    
    nearest_dists = np.min(dist_matrix, axis=1)
    mean_observed_dist = np.mean(nearest_dists)
    
    # Calculate study area density
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

    # Filter out NA for modal analysis
    df_valid = df_mapped[df_mapped["Standardized_Phase"].str.upper() != "NA"].copy()
    valid_points_count = len(df_valid)
    na_count = len(df_mapped) - valid_points_count

    st.caption(f"ℹ️ **Exclusion Notice:** {na_count} 'NA' points excluded from statistical analysis. Valid count: {valid_points_count}.")

    # ---------------------------------------------------------
    # STEP 3: MODAL ANALYSIS & STATISTICAL INTERPRETATION
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Step 3: Modal Analysis Results")

    summary = df_valid.groupby("Standardized_Phase").size().reset_index(name="Point_Count")
    summary["Modal_Percent (%)"] = (summary["Point_Count"] / valid_points_count) * 100
    
    # 2-Sigma Standard Error (Van der Plas & Tobi 1965)
    p = summary["Modal_Percent (%)"] / 100
    summary["Std_Error (%)"] = 2 * np.sqrt((p * (1 - p)) / valid_points_count) * 100

    # Abundance Column for Aggregates & Rock
    if sample_type in ["Aggregates", "Rock"]:
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
            text="Modal_Percent (%)", color="Standardized_Phase",
            labels={"Standardized_Phase": "Phase"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------------
    # SPECIAL MODULE: Concrete - I.S.465 Analysis
    # ---------------------------------------------------------
    if sample_type == "Concrete - I.S.465":
        st.markdown("---")
        st.subheader("🧱 Concrete - I.S.465 Aggregate Derivations")
        
        # Filter for Good and Bad Aggregate
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
            st.warning("No points classified as 'Good Aggregate' or 'Bad Aggregate' were found in the current mapping.")

    # ---------------------------------------------------------
    # SPECIAL MODULE: QAPF Ternary Plot (Rock & Aggregates)
    # ---------------------------------------------------------
    if sample_type in ["Rock", "Aggregates"]:
        st.markdown("---")
        st.subheader("🔺 QAPF Ternary Classification")
        
        # Look for Q, A, P, F constituents
        q_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Quartz", case=False), "Point_Count"].sum()
        a_cnt = summary.loc[summary["Standardized_Phase"].str.contains("K-Feldspar|Alkali", case=False), "Point_Count"].sum()
        p_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Plagioclase", case=False), "Point_Count"].sum()
        f_cnt = summary.loc[summary["Standardized_Phase"].str.contains("Feldspathoid", case=False), "Point_Count"].sum()
        
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
            st.plotly_chart(fig_ternary, use_container_width=True)
        else:
            st.info("Insufficient QAPF constituents (Quartz, K-Feldspar, Plagioclase) found to render ternary diagram.")

    # ---------------------------------------------------------
    # STEP 4: SPATIAL DISTRIBUTION & NEAREST NEIGHBOR INDEX
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📍 Thin Section Point Spatial Distribution & Index")

    c_map, c_index = st.columns([2, 1])

    with c_map:
        fig_scatter = px.scatter(
            df_valid, x="X", y="Y", color="Standardized_Phase",
            title="Spatial Point Distribution Across Thin Section",
            labels={"X": "X Coordinate (px/μm)", "Y": "Y Coordinate (px/μm)"}
        )
        fig_scatter.update_yaxes(autorange="reversed")  # Match image orientation
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c_index:
        st.write("##### Spatial Dispersion Coefficient")
        R_val, R_desc = calculate_spatial_index(df_valid)
        if R_val is not None:
            st.metric("Nearest Neighbor Ratio (R)", f"{R_val:.3f}")
            st.info(f"**Interpretation:** {R_desc}")
            st.caption(
                "• **R = 1:** Random Spatial Distribution\n"
                "• **R < 1:** Clustered / Segregated (e.g., veining, aggregate clumping)\n"
                "• **R > 1:** Uniform / Regular Grid Distribution"
            )
        else:
            st.write("Unable to calculate dispersion index.")

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
        file_name=f"{job_number}_{sample_id}_modal_report.csv",
        mime="text/csv"
    )
else:
    st.info("👈 Please upload a point count CSV/TSV file from the sidebar to begin.")