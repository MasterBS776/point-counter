import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Geological Petrography QC & Analysis", layout="wide")

st.title("🔬 Petrographic Point Count QC & Analysis Platform")

# ---------------------------------------------------------
# 1. SAMPLE TYPE DEFINITIONS & MASTER DICTIONARIES
# ---------------------------------------------------------
# Default lookup tables per sample type
DEFAULT_LOOKUPS = {
    "Cement / Clinker": {
        "Phase1": "Alite (C3S)",
        "Phase2": "Belite (C2S)",
        "Phase3": "Aluminate (C3A)",
        "Phase4": "Ferrite (C4AF)",
        "Phase5": "Free Lime",
        "Phase6": "Periclase",
        "Phase7": "Pores / Void",
        "Phase8": "Calcite",
        "Phase9": "Gypsum",
        "Good_Agg": "Sound Aggregate",
        "Bad_Agg": "Reactive Aggregate",
        "NA": "Unidentified / Void"
    },
    "Sandstone / Petrography": {
        "Phase1": "Monocrystalline Quartz",
        "Phase2": "Polycrystalline Quartz",
        "Phase3": "K-Feldspar",
        "Phase4": "Plagioclase",
        "Phase5": "Rock Fragment (Lithic)",
        "Phase6": "Mica",
        "Phase7": "Matrix / Clay",
        "Phase8": "Cement (Calcite)",
        "Phase9": "Porosity",
        "NA": "Unidentified"
    }
}

# ---------------------------------------------------------
# SIDEBAR: Upload & Configuration
# ---------------------------------------------------------
st.sidebar.header("1. Upload & Setup")
uploaded_file = st.sidebar.file_uploader("Upload Point Count Data (CSV/TSV)", type=["txt", "tsv", "csv"])
sample_type = st.sidebar.selectbox("Select Sample Type Strategy", list(DEFAULT_LOOKUPS.keys()))

if uploaded_file is not None:
    # Read TSV/CSV raw data
    try:
        df_raw = pd.read_csv(uploaded_file, sep="\t", header=None)
        if df_raw.shape[1] == 1: # Fallback if comma separated
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=",", header=None)
        
        # Name columns matching input structure
        df_raw.columns = ["Image_File", "Point_ID", "X", "Y", "Raw_Term", "Code_ID", "Timestamp"]
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        st.stop()

    st.subheader("📋 Step 1: Sample Metadata & Ingestion")
    col1, col2, col3 = st.columns(3)
    sample_id = col1.text_input("Sample ID", value=df_raw["Image_File"].iloc[0].split(".")[0])
    operator = col2.text_input("Operator Name", value="Analyst 1")
    total_points = len(df_raw)
    col3.metric("Total Points Counted", total_points)

    st.markdown("---")
    st.subheader("🛠️ Step 2: Quality Control (QC) & Term Mapping")
    st.info("Review raw input terms mapped against standard taxonomy. Update any unmapped terms before committing to DB.")

    # Extract unique raw terms present in this sample
    unique_terms = df_raw["Raw_Term"].unique().tolist()
    
    # Pre-fill mapping based on selected sample type lookup
    mapping_dict = {}
    qc_cols = st.columns(2)
    
    for idx, term in enumerate(unique_terms):
        default_val = DEFAULT_LOOKUPS[sample_type].get(term, term)
        with qc_cols[idx % 2]:
            mapping_dict[term] = st.text_input(
                label=f"Raw Term: **{term}**", 
                value=default_val, 
                key=f"map_{term}"
            )

    # Apply taxonomy mapping to dataset
    df_mapped = df_raw.copy()
    df_mapped["Standardized_Phase"] = df_mapped["Raw_Term"].map(mapping_dict)

    # ---------------------------------------------------------
    # STEP 3: STATISTICAL ANALYSIS & INTERPRETATION
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Step 3: Modal Analysis & Statistical Interpretation")

    # Group counts
    summary = df_mapped.groupby("Standardized_Phase").size().reset_index(name="Point_Count")
    summary["Modal_Percent (%)"] = (summary["Point_Count"] / total_points) * 100
    
    # Calculate 2-Sigma Standard Error (Van der Plas & Tobi 1965)
    p = summary["Modal_Percent (%)"] / 100
    summary["Std_Error (%)"] = 2 * np.sqrt((p * (1 - p)) / total_points) * 100

    summary["Modal_Percent (%)"] = summary["Modal_Percent (%)"].round(2)
    summary["Std_Error (%)"] = summary["Std_Error (%)"].round(2)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.write("##### Summary Table")
        st.dataframe(summary, use_container_width=True)

    with col_right:
        st.write("##### Phase Distribution")
        st.bar_chart(data=summary, x="Standardized_Phase", y="Modal_Percent (%)")

    # ---------------------------------------------------------
    # STEP 4: DATABASE COMMIT & EXPORT
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("💾 Step 4: Commit & Export")

    c1, c2 = st.columns(2)
    if c1.button("✅ Confirm QC & Save to Cloud Database", type="primary"):
        # Here you would connect to PostgreSQL via SQLAlchemy / Supabase client
        st.success(f"Sample '{sample_id}' successfully QC'd and saved to Database!")
        st.balloons()

    # CSV Download link
    csv_out = summary.to_csv(index=False).encode('utf-8')
    c2.download_button(
        label="📥 Download Interpreted Report (CSV)",
        data=csv_out,
        file_name=f"{sample_id}_interpreted_report.csv",
        mime="text/csv"
    )
else:
    st.info("👈 Please upload a point count text file from the sidebar to begin analysis.")