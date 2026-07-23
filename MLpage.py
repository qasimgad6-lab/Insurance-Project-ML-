from pathlib import Path
import pandas as pd
import joblib
import plotly.express as px
import streamlit as st
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================
# Custom Styling & Colors
# =========================
PRIMARY_COLOR = "#636EFA"
SECONDARY_COLOR = "#00CC96"
ACCENT_COLOR = "#EF553B"
BG_COLOR = "#0E1117"
GRID_COLOR = "#2A2E39"

custom_template = dict(
    layout=dict(
        font=dict(family="Cairo, Arial", size=14, color="white"),
        title=dict(x=0.05, xanchor="left"),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
        margin=dict(l=40, r=40, t=50, b=40),
    )
)

# =========================
# Data Loading & Auto-Cleaning
# =========================
from pathlib import Path

@st.cache_data
def load_data():
    # 1. Get the directory where MLpage.py actually lives
    try:
        BASE_DIR = Path(__file__).resolve().parent
    except NameError:
        BASE_DIR = Path.cwd()

    # 2. Point to the CSV relative to the script directory
    csv_path = BASE_DIR / "insurance_cleaned.csv"

    # 3. Fallback to current working directory if not found in BASE_DIR
    if not csv_path.exists():
        csv_path = Path.cwd() / "insurance_cleaned.csv"

    # 4. Load & Clean
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error(f"⚠️ Could not locate `insurance_cleaned.csv` at `{csv_path}`. "
                 "Make sure the CSV file is pushed to your GitHub repository.")
        st.stop()

    # 1. Reconstruct 'region' if it was one-hot encoded
    region_cols = [c for c in df.columns if c.startswith("region_")]
    if "region" not in df.columns and region_cols:

        def get_region(row):
            for col in region_cols:
                if row[col] == 1:
                    return col.replace("region_", "").capitalize()
            return "Northeast"

        df["region"] = df.apply(get_region, axis=1)

    # 2. Map encoded binary numeric values back to string labels for clean visuals
    if "smoker" in df.columns and pd.api.types.is_numeric_dtype(df["smoker"]):
        df["smoker"] = df["smoker"].map({1: "yes", 0: "no"}).fillna(df["smoker"])

    if "sex" in df.columns and pd.api.types.is_numeric_dtype(df["sex"]):
        df["sex"] = df["sex"].map({1: "male", 0: "female"}).fillna(df["sex"])

    return df


df = load_data()

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("🔎 Filters")

selected_sex = st.sidebar.multiselect(
    "Sex",
    options=df["sex"].unique(),
    default=df["sex"].unique(),
)

selected_smoker = st.sidebar.multiselect(
    "Smoker",
    options=df["smoker"].unique(),
    default=df["smoker"].unique(),
)

selected_region = st.sidebar.multiselect(
    "Region",
    options=df["region"].unique(),
    default=df["region"].unique(),
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df["age"].min()),
    max_value=int(df["age"].max()),
    value=(int(df["age"].min()), int(df["age"].max())),
)

filtered_df = df[
    (df["sex"].isin(selected_sex))
    & (df["smoker"].isin(selected_smoker))
    & (df["age"].between(age_range[0], age_range[1]))
]

# =========================
# Dashboard Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📌 Overview",
        "📊 Univariate",
        "📈 Bivariate",
        "📉 Multivariate & Correlation",
    ]
)

# =========================
# TAB 1: Overview
# =========================
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", len(filtered_df))
    col2.metric(
        "Avg Charges",
        f"${int(filtered_df['charges'].mean()):,}"
        if not filtered_df.empty
        else "$0",
    )
    col3.metric(
        "Avg Age",
        f"{filtered_df['age'].mean():.1f}" if not filtered_df.empty else "0",
    )
    col4.metric(
        "Avg BMI",
        f"{filtered_df['bmi'].mean():.1f}" if not filtered_df.empty else "0",
    )

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(10), use_container_width=True)


# =========================
# TAB 2: Univariate
# =========================
with tab2:
    st.subheader("📊 Univariate Analysis")

    num_tab, cat_tab = st.tabs(["📈 Numerical", "📊 Categorical"])

    # =========================
    # 📈 Numerical Features
    # =========================
    with num_tab:

        # --- Age ---
        st.markdown("### 🎂 Age Analysis")

        # Display basic stats for Age
        with st.expander("📌 Basic stats for Age", expanded=False):
            st.dataframe(df["age"].describe().to_frame().T, use_container_width=True)

        fig_age = px.histogram(
            filtered_df,
            x="age",
            nbins=50,
            title="Distribution of age (Plotly)",
            color_discrete_sequence=["indigo"],
        )
        fig_age.update_layout(template=custom_template, bargap=0.1)
        st.plotly_chart(fig_age, use_container_width=True)

        # --- BMI ---
        st.markdown("### ⚖️ BMI Analysis")
        with st.expander("📌 Basic stats for BMI", expanded=False):
            st.dataframe(df["bmi"].describe().to_frame().T, use_container_width=True)

        fig_bmi = px.histogram(
            filtered_df,
            x="bmi",
            nbins=50,
            title="Distribution of BMI",
            color_discrete_sequence=[SECONDARY_COLOR],
        )
        fig_bmi.update_layout(template=custom_template, bargap=0.1)
        st.plotly_chart(fig_bmi, use_container_width=True)

        # --- Charges ---
        st.markdown("### 💰 Charges Analysis")
        with st.expander("📌 Basic stats for Charges", expanded=False):
            st.dataframe(
                df["charges"].describe().to_frame().T, use_container_width=True
            )

        fig_charges = px.histogram(
            filtered_df,
            x="charges",
            nbins=50,
            title="Distribution of Charges",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        fig_charges.update_layout(template=custom_template, bargap=0.1)
        st.plotly_chart(fig_charges, use_container_width=True)

    # =========================
    # 📊 Categorical Features
    # =========================
    with cat_tab:

        # --- Sex Distribution ---
        st.markdown("### 👤 Sex Analysis")

        col_s1, col_s2 = st.columns([1, 2])

        with col_s1:
            st.markdown("**📌 Basic stats for Sex:**")
            st.dataframe(df["sex"].describe().to_frame(), use_container_width=True)

        with col_s2:
            # Value counts for Pie Chart
            sex_counts = filtered_df["sex"].value_counts().reset_index()
            sex_counts.columns = ["sex", "Count"]

            fig_sex_pie = px.pie(
                sex_counts,
                names="sex",
                values="Count",
                title="Sex Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_sex_pie.update_traces(textinfo="percent+label")
            fig_sex_pie.update_layout(template=custom_template)
            st.plotly_chart(fig_sex_pie, use_container_width=True)

        # Sex Histogram
        fig_sex_hist = px.histogram(
            filtered_df,
            x="sex",
            nbins=50,
            title="Distribution of sex (Plotly)",
            color_discrete_sequence=["indigo"],
        )
        fig_sex_hist.update_layout(template=custom_template, bargap=0.1)
        st.plotly_chart(fig_sex_hist, use_container_width=True)

        # --- Smoker Distribution ---
        st.markdown("---")
        st.markdown("### 🚬 Smoker Analysis")

        smoker_counts = filtered_df["smoker"].value_counts().reset_index()
        smoker_counts.columns = ["smoker", "Count"]

        fig_smoker_pie = px.pie(
            smoker_counts,
            names="smoker",
            values="Count",
            title="Smoker Distribution",
            color_discrete_sequence=["#e74c3c", "#2ecc71"],
        )
        fig_smoker_pie.update_traces(textinfo="percent+label")
        fig_smoker_pie.update_layout(template=custom_template)
        st.plotly_chart(fig_smoker_pie, use_container_width=True)

# =========================
# TAB 3: Bivariate
# =========================
with tab3:
    st.subheader("📈 Feature Impact on Medical Charges")

    # Line Chart: Average Charges by Age
    age_charges = (
        filtered_df.groupby("age")["charges"].mean().reset_index()
    )
    fig_age = px.line(
        age_charges,
        x="age",
        y="charges",
        title="📈 Average Charges by Age",
        markers=True,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig_age.update_layout(template=custom_template)
    st.plotly_chart(fig_age, use_container_width=True)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        # Bar Chart: Average Charges by Sex
        sex_charges = (
            filtered_df.groupby("sex")["charges"].mean().reset_index()
        )
        fig_sex = px.bar(
            sex_charges,
            x="sex",
            y="charges",
            title="👤 Average Charges by Sex",
            color="sex",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_sex.update_layout(template=custom_template, showlegend=False)
        st.plotly_chart(fig_sex, use_container_width=True)

        # Bar Chart: Average Charges by Children
        children_charges = (
            filtered_df.groupby("children")["charges"].mean().reset_index()
        )
        fig_child = px.bar(
            children_charges,
            x="children",
            y="charges",
            title="👶 Average Charges by Number of Children",
            color="charges",
            color_continuous_scale="Blues",
        )
        fig_child.update_layout(
            template=custom_template, coloraxis_showscale=False
        )
        st.plotly_chart(fig_child, use_container_width=True)

    with col_b2:
        # Bar Chart: Average Charges by Smoking Status
        smoker_charges = (
            filtered_df.groupby("smoker")["charges"].mean().reset_index()
        )
        fig_smoker = px.bar(
            smoker_charges,
            x="smoker",
            y="charges",
            title="🚬 Average Charges by Smoking Status",
            color="smoker",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        fig_smoker.update_layout(template=custom_template, showlegend=False)
        st.plotly_chart(fig_smoker, use_container_width=True)

        # Scatter Plot: BMI vs Charges
        fig_bmi = px.scatter(
            filtered_df,
            x="bmi",
            y="charges",
            title="⚖️ BMI vs Charges",
            opacity=0.6,
            color_discrete_sequence=[SECONDARY_COLOR],
        )
        fig_bmi.update_layout(template=custom_template)
        st.plotly_chart(fig_bmi, use_container_width=True)

# =========================
# TAB 4: Multivariate & Correlation
# =========================
with tab4:
    st.subheader("📉 Advanced Interactions & Correlation")

    # Violin Plot equivalent: Sex, Charges, Smoker Interaction
    fig_violin = px.violin(
        filtered_df,
        x="sex",
        y="charges",
        color="smoker",
        box=True,
        points="all",
        title="🔥 Multivariate Analysis: Medical Charges by Sex and Smoking Status",
        color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"},
    )
    fig_violin.update_layout(template=custom_template)
    st.plotly_chart(fig_violin, use_container_width=True)

    # Scatter: BMI vs Charges colored by Smoker
    fig_scatter = px.scatter(
        filtered_df,
        x="bmi",
        y="charges",
        color="smoker",
        size="age",
        title="⚖️ BMI vs Charges (Colored by Smoker, Sized by Age)",
        color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"},
        opacity=0.7,
    )
    fig_scatter.update_layout(template=custom_template)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation Matrix Heatmap
    st.subheader("📊 Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])

    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",  # ✅ Valid Plotly palette (Red to Blue, reversed)
            title="Correlation Matrix of Numeric Features",
            aspect="auto",
    )
        fig_corr.update_layout(template=custom_template)
        st.plotly_chart(fig_corr, use_container_width=True)
