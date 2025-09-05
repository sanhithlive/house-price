# --- macOS-inspired Streamlit UI/UX enhancement (no functional changes) ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Assuming these are custom modules you have created.
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from utils import format_currency, create_prediction_explanation

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Hyderabad House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# macOS theme helpers
# ---------------------------
def apply_theme(high_contrast: bool = False, reduced_motion: bool = False):
    """
    Injects macOS Big Sur/Sonoma-inspired design:
    - Fluid gradient background with frosted glass cards
    - SF-like typography stack
    - Subtle motion (respecting reduced-motion)
    - Focus rings & high-contrast mode
    """
    # Color tokens
    base_text = "#0B0C0E" if not high_contrast else "#000000"
    muted_text = "rgba(11,12,14,0.7)" if not high_contrast else "#000000"
    glass_bg = "rgba(255,255,255,0.55)" if not high_contrast else "rgba(255,255,255,0.85)"
    card_border = "rgba(255,255,255,0.35)" if not high_contrast else "rgba(0,0,0,0.25)"
    ring = "#0A84FF"  # macOS accent blue
    shadow = "0 10px 30px rgba(0,0,0,0.08), 0 1px 0 rgba(255,255,255,0.2) inset"

    motion = "0.3s cubic-bezier(.2,.8,.2,1)" if not reduced_motion else "0s linear"

    st.markdown(f"""
    <style>
      :root {{
        --text: {base_text};
        --text-muted: {muted_text};
        --glass: {glass_bg};
        --ring: {ring};
        --card-border: {card_border};
        --shadow: {shadow};
        --radius: 18px;
        --radius-lg: 22px;
        --transition: {motion};
      }}

      /* Background: soft, layered macOS gradient with blurred blobs */
      .stApp {{
        background: radial-gradient(1200px 700px at 10% -10%, #dbe6ff 0%, rgba(255,255,255,0) 60%),
                    radial-gradient(900px 700px at 110% 0%, #ffd7e5 0%, rgba(255,255,255,0) 55%),
                    linear-gradient(180deg, #f6f8ff 0%, #ffffff 100%);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                     "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      }}

      /* Sidebar glass */
      section[data-testid="stSidebar"] > div:first-child {{
        background: var(--glass);
        backdrop-filter: saturate(150%) blur(18px);
        border-right: 1px solid var(--card-border);
      }}

      /* Global headings */
      h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {{ letter-spacing: -0.02em; }}

      /* Mac window chrome bar */
      .mac-chrome {{
        position: sticky;
        top: 0;
        z-index: 10;
        margin: -1rem -1rem 1rem -1rem;
        padding: 10px 16px;
        background: var(--glass);
        backdrop-filter: blur(18px) saturate(160%);
        border-bottom: 1px solid var(--card-border);
      }}
      .dots {{
        display: inline-flex; gap: 8px; align-items: center;
      }}
      .dot {{
        width: 12px; height: 12px; border-radius: 999px; box-shadow: 0 1px 0 rgba(0,0,0,0.25) inset;
        transition: transform var(--transition), filter var(--transition);
      }}
      .dot.red {{ background: #FF605C; }}
      .dot.yellow {{ background: #FFBD44; }}
      .dot.green {{ background: #00CA4E; }}
      .dot:hover {{ transform: scale(1.06); filter: saturate(110%); }}

      /* Glass cards */
      .card {{
        background: var(--glass);
        border: 1px solid var(--card-border);
        backdrop-filter: blur(18px) saturate(160%);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow);
        transition: transform var(--transition), box-shadow var(--transition);
      }}
      .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.08);
      }}

      /* Buttons */
      button[kind="primary"], .stButton button {{
        border-radius: 14px !important;
        padding: 0.6rem 1rem !important;
        border: 1px solid rgba(0,0,0,0.04) !important;
        box-shadow: 0 8px 20px rgba(10,132,255,0.15) !important;
        transition: transform var(--transition), box-shadow var(--transition), background var(--transition);
      }}
      .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 10px 26px rgba(10,132,255,0.22) !important;
      }}
      .stButton button:focus-visible {{
        outline: 3px solid var(--ring) !important;
        outline-offset: 2px !important;
      }}

      /* Inputs */
      .stSelectbox, .stNumberInput, .stTextInput {{
        border-radius: 14px !important;
      }}
      .stSelectbox > div > div, .stNumberInput > div, .stTextInput > div {{
        border-radius: 14px !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
      }}

      /* Metrics: subtle sheen */
      [data-testid="stMetricValue"] {{
        font-variation-settings: "wght" 650;
      }}
      [data-testid="stMetric"] {{
        background: var(--glass);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 10px 14px;
        box-shadow: var(--shadow);
      }}

      /* Links & labels */
      a, .stMarkdown a {{ color: #0A84FF; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}

      /* Reduce motion (prefers + toggle) */
      @media (prefers-reduced-motion: reduce) {{
        .dot, .card, .stButton button {{ transition: none !important; }}
      }}
    </style>
    """, unsafe_allow_html=True)

def mac_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="mac-chrome">
          <div class="dots">
            <span class="dot red"></span>
            <span class="dot yellow"></span>
            <span class="dot green"></span>
          </div>
        </div>
        <div class="card" style="padding: 20px; margin-bottom: 1rem;">
          <h1 style="margin:0">{title}</h1>
          <p style="margin-top:6px; color: var(--text-muted)">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def plotly_mac_layout(fig, title=None):
    """Unifies chart look with the UI (font, spacing, grid, rounded corners)."""
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        template="plotly_white",
        font=dict(family='-apple-system, "SF Pro Text", Inter, Segoe UI, Roboto, Arial', size=14),
        title_x=0.02,
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ---------------------------
# Session state
# ---------------------------
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# ---------------------------
# Data & model functions (unchanged)
# ---------------------------
def load_and_process_data():
    """Load and process the house data"""
    try:
        data_file = "Hyderabad_House_Data_1755501086395.csv"
        if not os.path.exists(data_file):
            st.error(f"Data file not found: {data_file}")
            st.info("Please ensure the data file is in the correct location.")
            return None

        processor = DataProcessor()
        processed_data = processor.load_and_clean_data(data_file)

        if processed_data is None or processed_data.empty:
            st.error("Failed to process the data. Please check the data file format.")
            return None

        return processed_data

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_model(data):
    """Train the machine learning model"""
    try:
        trainer = ModelTrainer()
        model, metrics, feature_names = trainer.train_model(data)
        return model, metrics, feature_names
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

# ---------------------------
# Pages (functionality intact)
# ---------------------------
def show_prediction_page():
    """Main prediction interface"""
    st.header("HYD Predict House Price")

    if not st.session_state.model_trained:
        st.warning("Model is still training. Please wait...")
        return

    # Create input form in glass cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card" style="padding:16px">', unsafe_allow_html=True)
        st.subheader("Property Details")
        bedrooms = st.selectbox(
            "Number of Bedrooms:",
            options=["1 BHK", "2 BHK", "3 BHK", "4 BHK", "Studio"],
            index=2,
            help="Select the type of accommodation"
        )
        bathrooms = st.number_input(
            "Number of Bathrooms:",
            min_value=1, max_value=5, value=2,
            help="Enter the number of bathrooms"
        )
        furnishing = st.selectbox(
            "Furnishing Type:",
            options=["Furnished", "Semi-Furnished", "Unfurnished"],
            index=1, help="Select the furnishing status"
        )
        tenant_type = st.selectbox(
            "Tenant Preference:",
            options=["Bachelors/Family", "Family", "Bachelors"],
            index=0, help="Select preferred tenant type"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card" style="padding:16px">', unsafe_allow_html=True)
        st.subheader("Location & Area")
        localities = sorted(st.session_state.processed_data['Locality_clean'].unique())
        locality = st.selectbox("Locality:", options=localities, help="Select the locality/area")
        area = st.number_input(
            "Area (sq ft):", min_value=300, max_value=5000, value=1500, step=50,
            help="Enter the property area in square feet"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="padding:16px; margin-top: 10px;">', unsafe_allow_html=True)
    if st.button("🔮 Predict Price", type="primary"):
        try:
            processor = DataProcessor()
            input_data = processor.prepare_prediction_input(
                bedrooms, bathrooms, furnishing, tenant_type, area, locality,
                st.session_state.processed_data
            )
            prediction = st.session_state.model.predict([input_data])[0]
            st.success("Prediction completed!")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Price", format_currency(prediction), help="Estimated monthly rent")

            confidence_range = prediction * 0.15
            lower_bound = max(0, prediction - confidence_range)
            upper_bound = prediction + confidence_range
            with m2:
                st.metric("Price Range", f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}", help="Estimated price range (±15%)")

            price_per_sqft = prediction / area
            with m3:
                st.metric("Price per Sq Ft", f"₹{price_per_sqft:.0f}", help="Rent per square foot")

            st.subheader("Prediction Explanation")
            explanation = create_prediction_explanation(bedrooms, locality, area, furnishing, prediction)
            st.info(explanation)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_performance():
    """Display model performance metrics"""
    st.header("Model Performance")

    if not st.session_state.model_trained or st.session_state.model_metrics is None:
        st.warning("Model metrics not available. Please train the model first.")
        return

    metrics = st.session_state.model_metrics

    # Metric grid in a glass card
    st.markdown('<div class="card" style="padding:16px; margin-bottom: 12px;">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("R² Score", f"{metrics['r2']:.3f}", help="Coefficient of determination (higher is better)")
    with c2:
        st.metric("MAE", f"₹{metrics['mae']:,.0f}", help="Mean Absolute Error")
    with c3:
        st.metric("RMSE", f"₹{metrics['rmse']:,.0f}", help="Root Mean Square Error")
    with c4:
        st.metric("CV Score", f"{metrics['cv_score']:.3f}", help="Cross-validation score")
    st.markdown('</div>', unsafe_allow_html=True)

    if 'feature_importance' in metrics:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'Importance': metrics['feature_importance']
        }).sort_values('Importance', ascending=False)

        fig = px.bar(
            importance_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
        )
        fig = plotly_mac_layout(fig, "Top 10 Most Important Features")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Details")
    st.info("""
    **Model Type:** Random Forest Regressor

    **Training Details:**
    - Algorithm: Random Forest with 100 estimators
    - Cross-validation: 5-fold
    - Features: Bedrooms, Bathrooms, Area, Locality, Furnishing, Tenant Type
    - Target: Monthly Rent (INR)

    **Performance Interpretation:**
    - **R² Score:** Explains the proportion of variance in house prices
    - **MAE:** Average absolute difference between predicted and actual prices
    - **RMSE:** Root mean square error, penalizes larger errors more
    - **CV Score:** Average performance across different data splits
    """)

def show_data_insights():
    """Display data insights and visualizations"""
    st.header("Data Analysis & Insights")

    data = st.session_state.processed_data

    st.subheader("Dataset Summary")
    st.markdown('<div class="card" style="padding:16px; margin-bottom: 12px;">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Properties", len(data))
    with c2:
        st.metric("Average Price", format_currency(data['Price_clean'].mean()))
    with c3:
        st.metric("Unique Localities", data['Locality_clean'].nunique())
    with c4:
        st.metric("Price Range", f"₹{data['Price_clean'].min():,.0f} - ₹{data['Price_clean'].max():,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.box(
            data, x='Bedrooms_clean', y='Price_clean',
            labels={'Price_clean': 'Price (INR)', 'Bedrooms_clean': 'Bedroom Type'}
        )
        fig1 = plotly_mac_layout(fig1, "Price Distribution by Bedroom Type")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.violin(
            data, x='Furnishing', y='Price_clean',
            labels={'Price_clean': 'Price (INR)', 'Furnishing': 'Furnishing Type'}
        )
        fig2 = plotly_mac_layout(fig2, "Price Distribution by Furnishing Type")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top Localities by Average Price")
    locality_avg = data.groupby('Locality_clean')['Price_clean'].agg(['mean', 'count']).reset_index()
    locality_avg = locality_avg[locality_avg['count'] >= 3].sort_values('mean', ascending=False).head(15)

    fig3 = px.bar(
        locality_avg, x='mean', y='Locality_clean', orientation='h',
        labels={'mean': 'Average Price (INR)', 'Locality_clean': 'Locality'}
    )
    fig3.update_layout(height=600)
    fig3 = plotly_mac_layout(fig3, "Average Rent by Locality (Top 15)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Price vs Area Analysis")
    fig4 = px.scatter(
        data, x='Area_clean', y='Price_clean', color='Bedrooms_clean',
        labels={'Area_clean': 'Area (sq ft)', 'Price_clean': 'Price (INR)'},
        hover_data=['Locality_clean']
    )
    fig4 = plotly_mac_layout(fig4, "Price vs Area by Bedroom Type")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# Main
# ---------------------------
def main():
    # Quick-access accessibility controls in the sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox("Choose a page:", ["Price Prediction", "Model Performance", "Data Insights"])
        st.markdown("---")
        st.caption("Accessibility")
        high_contrast = st.toggle("High contrast", value=False, help="Increase contrast for readability")
        reduced_motion = st.toggle("Reduce motion", value=False, help="Disable decorative animations")

    # Apply theme after reading toggles
    apply_theme(high_contrast=high_contrast, reduced_motion=reduced_motion)

    mac_header(
        "Hyderabad House Price Predictor",
        "Intelligent estimates for Hyderabad rentals ."
    )

    # Load data if not already loaded
    if st.session_state.processed_data is None:
        with st.spinner("Loading and processing data..."):
            st.session_state.processed_data = load_and_process_data()

    # Train model if not already trained
    if not st.session_state.model_trained and st.session_state.processed_data is not None:
        with st.spinner("Training machine learning model..."):
            model, metrics, feature_names = train_model(st.session_state.processed_data)
            if model is not None:
                st.session_state.model = model
                st.session_state.model_metrics = metrics
                st.session_state.feature_names = feature_names
                st.session_state.model_trained = True
                st.success("Model trained successfully!")

    if st.session_state.processed_data is None:
        st.error("Unable to load data. Please check the data file.")
        return

    # Page routing
    if page == "Price Prediction":
        show_prediction_page()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Data Insights":
        show_data_insights()

if __name__ == "__main__":
    main()
