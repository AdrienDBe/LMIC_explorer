import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import requests
from functools import lru_cache
import time

st.set_page_config(
    page_title="LMIC Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.markdown("""
<style>
/* ==================== FORCE LIGHT MODE - COMPREHENSIVE ==================== */

/* Root containers */
html, body {
    background-color: #FFFFFF !important;
    color: #262730 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
}

[data-testid="stSidebar"] {
    background-color: #F0F2F6 !important;
}

[data-testid="stSidebarContent"] {
    background-color: #F0F2F6 !important;
}

/* Main content */
div.stMain {
    background-color: #FFFFFF !important;
}

div.block-container {
    background-color: #FFFFFF !important;
}

/* ==================== TEXT & TYPOGRAPHY ==================== */

p, span, label, div {
    color: #262730 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #262730 !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #262730 !important;
}

/* ==================== FORMS & INPUTS ==================== */

input, textarea, select {
    background-color: #FFFFFF !important;
    color: #262730 !important;
    border: 1px solid #D3D3D3 !important;
}

input::placeholder {
    color: #999999 !important;
}

textarea::placeholder {
    color: #999999 !important;
}

/* Focus states */
input:focus, textarea:focus, select:focus {
    background-color: #FFFFFF !important;
    color: #262730 !important;
    border: 2px solid #4682b4 !important;
}

/* ==================== BUTTONS ==================== */

button, [data-testid="baseButton-primary"] {
    background-color: #4682b4 !important;
    color: #FFFFFF !important;
    border: none !important;
}

button:hover {
    background-color: #356a99 !important;
    color: #FFFFFF !important;
}

/* Pill buttons */
.stPills button {
    background-color: #E8E8E8 !important;
    color: #262730 !important;
    border: 1px solid #D3D3D3 !important;
}

.stPills button[data-testid*="pill-button"]:not([aria-pressed="false"]) {
    background-color: #4682b4 !important;
    color: #FFFFFF !important;
    border: 1px solid #4682b4 !important;
}

/* ==================== DATAFRAMES & TABLES ==================== */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #FFFFFF !important;
}

[data-testid="stDataFrame"] th,
[data-testid="stTable"] th {
    background-color: #E8E8E8 !important;
    color: #262730 !important;
}

[data-testid="stDataFrame"] td,
[data-testid="stTable"] td {
    background-color: #FFFFFF !important;
    color: #262730 !important;
    border-color: #D3D3D3 !important;
}

/* Data editor */
[data-testid="stDataEditor"] {
    background-color: #FFFFFF !important;
}

/* ==================== EXPANDERS & CONTAINERS ==================== */

[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D3D3D3 !important;
}

[data-testid="stExpander"] button {
    color: #262730 !important;
}

[data-testid="stExpander"] summary {
    color: #262730 !important;
}

/* ==================== SELECTBOX & DROPDOWN ==================== */

[data-testid="stSelectbox"] {
    background-color: #FFFFFF !important;
}

[role="listbox"] {
    background-color: #FFFFFF !important;
    color: #262730 !important;
}

[role="option"] {
    color: #262730 !important;
}

[role="option"]:hover {
    background-color: #E8E8E8 !important;
    color: #262730 !important;
}

/* ==================== RADIO & CHECKBOX ==================== */

[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #262730 !important;
}

/* ==================== METRIC VALUES ==================== */

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: #262730 !important;
}

/* ==================== POPOVER ==================== */

[data-testid="stPopover"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D3D3D3 !important;
}

/* ==================== COLUMNS & LAYOUT ==================== */

[data-testid="column"] {
    background-color: #FFFFFF !important;
}

/* ==================== PLOTLY CHARTS ==================== */

.plotly {
    background-color: #FFFFFF !important;
}

.plotly .bg {
    fill: #FFFFFF !important;
}

.plotly-notebooklogo {
    display: none !important;
}

/* ==================== MESSAGES ==================== */

[data-testid="stAlert"] {
    background-color: #F5F5F5 !important;
    color: #262730 !important;
    border-color: #D3D3D3 !important;
}

/* ==================== SIDEBAR SPECIFIC ==================== */

.sidebar-content {
    background-color: #F0F2F6 !important;
    color: #262730 !important;
}

/* Remove any dark overlays */
div[role="region"] {
    background-color: transparent !important;
}

/* ==================== SCROLLBAR ==================== */

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #F0F2F6 !important;
}

::-webkit-scrollbar-thumb {
    background: #B0B0B0 !important;
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: #808080 !important;
}

/* ==================== REMOVE DARK MODE ARTIFACTS ==================== */

[data-testid="stAppViewContainer"].dark {
    background-color: #FFFFFF !important;
}

/* Force all text to be dark */
* {
    color: #262730 !important;
}

/* Except for white text that should stay white */
button, [data-testid="baseButton-primary"] {
    color: #FFFFFF !important;
}

/* ==================== RESPONSIVE FIXES ==================== */

@media (prefers-color-scheme: dark) {
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        color: #262730 !important;
    }
}

</style>
""", unsafe_allow_html=True)






st.sidebar.markdown("")

if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = True


# --- OPTIMIZED CACHING FUNCTIONS ---

@st.cache_data(ttl=3600)
def load_world_bank_metadata():
    """Load World Bank metadata with caching"""
    try:
        url = "http://api.worldbank.org/v2/country?per_page=400&format=json"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        records = []
        for c in data[1]:
            records.append({
                "iso2c": c["id"],
                "name": c["name"],
                "region": c["region"]["value"],
                "incomeLevel": c["incomeLevel"]["value"]
            })
        df_wb = pd.DataFrame(records)
        df_wb = df_wb[df_wb['region'] != 'Aggregates']
        return df_wb
    except Exception as e:
        st.warning(f"Failed to load World Bank data: {e}")
        return pd.DataFrame(columns=["iso2c", "name", "region", "incomeLevel"])

@st.cache_data
def get_country_name_mapping():
    """Return country name mapping as cached dictionary"""
    return {
        'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
        'Egypt': 'Egypt, Arab Rep.',
        'Gambia': 'Gambia, The',
        'Iran': 'Iran, Islamic Rep.',
        'Ivory Coast': "Cote d'Ivoire",
        'Kyrgyzstan': 'Kyrgyz Republic',
        'Republic of the Congo': 'Congo, Rep.',
        'Russia': 'Russian Federation',
        'Saint Kitts and Nevis': 'St. Kitts and Nevis',
        'Slovakia': 'Slovak Republic',
        'South Korea': 'Korea, Rep.',
        'Syria': 'Syrian Arab Republic',
        'Taiwan': 'China',
        'Turkey': 'Turkiye',
        'Venezuela': 'Venezuela, RB',
        'Vietnam': 'Viet Nam',
        'Yemen': 'Yemen, Rep.',
        'USA': 'United States',
        'US': 'United States',
        'UK': 'United Kingdom'
    }

@st.cache_data
def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage"""
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        if optimized_df[col].dtype == 'object':
            unique_ratio = len(optimized_df[col].unique()) / len(optimized_df)
            if unique_ratio < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
    
    return optimized_df

@st.cache_data
def load_and_preprocess_data(filepath):
    """Load the combined publications CSV file with preprocessing"""
    try:
        dtype_dict = {
            'Name': 'string',
            'Organization': 'string',
            'Country': 'string',
            'Publication type': 'string',
            'Open Access': 'string'
        }
        
        df = pd.read_csv(filepath, low_memory=False, dtype=dtype_dict)
        
        # CRITICAL: Remove duplicates early
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['Name', 'Organization', 'Country', 'Publications'])
        df = df[df['Country'] != '']
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"Removed {rows_removed} duplicate rows during preprocessing")
        
        country_name_mapping = get_country_name_mapping()
        wb_countries = load_world_bank_metadata()
        
        if 'Country' in df.columns:
            df['Country'] = df['Country'].map(
                lambda x: country_name_mapping.get(x, x) if pd.notna(x) else x
            )
        
        if not wb_countries.empty and 'Country' in df.columns:
            df = df.merge(
                wb_countries[['name', 'region', 'incomeLevel']], 
                left_on='Country', 
                right_on='name', 
                how='left'
            )
            df = df.rename(columns={
                'region': 'Region',
                'incomeLevel': 'Income Level'
            })
            df['Income Level'] = df['Income Level'].replace('Not classified', 'Low income')
            
            if 'name' in df.columns:
                df = df.drop('name', axis=1)
        
        numeric_columns = ['Publications', 'Citations', 'Citations Mean']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        text_columns = ['Name', 'Organization', 'Publication type', 'Open Access']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        
        if 'Country' in df.columns:
            df['Country'] = df['Country'].fillna('Unknown').astype(str).str.strip()
            df['Country'] = df['Country'].replace('', 'Unknown')
        
        df = optimize_dataframe_memory(df)
        
        return df, country_name_mapping
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, {}

@st.cache_data
def get_unique_values(df, column):
    """Get unique values for a column with caching"""
    if column in df.columns:
        return sorted(df[column].dropna().unique().tolist())
    return []

@st.cache_data
def get_regional_hub_countries():
    """Return regional hub country mappings"""
    return {
        "A*STAR SIgN": ["Singapore", "Indonesia", "Philippines", "Viet Nam", "Thailand", "Malaysia"],
        "Institut Pasteur Network": ["Algeria", "Argentina", "Brazil", "Cambodia", "Cameroon", "Central African Republic", 
                                    "Chad", "China", "Côte d'Ivoire", "France", "French Guiana", "Gabon", "Greece", 
                                    "Guatemala", "Iran, Islamic Rep.", "Iraq", "Korea, Rep.", "Laos", "Madagascar", "Mongolia", "Morocco", 
                                    "New Caledonia", "Niger", "Senegal", "Tunisia", "Uruguay"],
        "KEMRI-Wellcome": ["Kenya", "Uganda", "Tanzania", "Ethiopia"],
        "AHRI": ["South Africa", "Botswana", "Eswatini", "Lesotho", "Malawi", "Mozambique", "Namibia", "Zambia", "Zimbabwe"]
    }

@st.cache_data
def get_country_coordinates():
    """Return approximate coordinates for countries (centroids)"""
    return {
        "Singapore": {"lat": 1.3521, "lon": 103.8198},
        "Indonesia": {"lat": -0.7893, "lon": 113.9213},
        "Philippines": {"lat": 12.8797, "lon": 121.7740},
        "Vietnam": {"lat": 14.0583, "lon": 108.2772},
        "Viet Nam": {"lat": 14.0583, "lon": 108.2772},
        "Thailand": {"lat": 15.8700, "lon": 100.9925},
        "Malaysia": {"lat": 4.2105, "lon": 101.9758},
        
        "Algeria": {"lat": 28.0339, "lon": 1.6596},
        "Argentina": {"lat": -38.4161, "lon": -63.6167},
        "Brazil": {"lat": -14.2350, "lon": -51.9253},
        "Cambodia": {"lat": 12.5657, "lon": 104.9910},
        "Cameroon": {"lat": 7.3697, "lon": 12.3547},
        "Central African Republic": {"lat": 6.6111, "lon": 20.9394},
        "Chad": {"lat": 15.4542, "lon": 18.7322},
        "China": {"lat": 35.8617, "lon": 104.1954},
        "Côte d'Ivoire": {"lat": 7.5399, "lon": -5.5471},
        "France": {"lat": 46.2276, "lon": 2.2137},
        "French Guiana": {"lat": 3.9339, "lon": -53.1258},
        "Gabon": {"lat": -0.8037, "lon": 11.6094},
        "Greece": {"lat": 39.0742, "lon": 21.8243},
        "Guatemala": {"lat": 15.7835, "lon": -90.2308},
        "Iran": {"lat": 32.4279, "lon": 53.6880},
        "Iran, Islamic Rep.": {"lat": 32.4279, "lon": 53.6880},
        "Iraq": {"lat": 33.2232, "lon": 43.6793},
        "Korea, Rep.": {"lat": 35.9078, "lon": 127.7669},
        "Laos": {"lat": 19.8563, "lon": 102.4955},
        "Madagascar": {"lat": -18.7669, "lon": 46.8691},
        "Mongolia": {"lat": 46.8625, "lon": 103.8467},
        "Morocco": {"lat": 31.7917, "lon": -7.0926},
        "New Caledonia": {"lat": -20.9043, "lon": 165.6180},
        "Niger": {"lat": 17.6078, "lon": 8.0817},
        "Senegal": {"lat": 14.4974, "lon": -14.4524},
        "Tunisia": {"lat": 33.8869, "lon": 9.5375},
        "Uruguay": {"lat": -32.5228, "lon": -55.7658},
        
        "Kenya": {"lat": -0.0236, "lon": 37.9062},
        "Uganda": {"lat": 1.3733, "lon": 32.2903},
        "Tanzania": {"lat": -6.3690, "lon": 34.8888},
        "Ethiopia": {"lat": 9.1450, "lon": 40.4897},
        
        "South Africa": {"lat": -30.5595, "lon": 22.9375},
        "Botswana": {"lat": -22.3285, "lon": 24.6849},
        "Eswatini": {"lat": -26.5225, "lon": 31.4659},
        "Lesotho": {"lat": -29.6097, "lon": 28.2336},
        "Malawi": {"lat": -13.2543, "lon": 34.3015},
        "Mozambique": {"lat": -18.6657, "lon": 35.5296},
        "Namibia": {"lat": -22.9576, "lon": 18.4904},
        "Zambia": {"lat": -13.1339, "lon": 27.8493},
        "Zimbabwe": {"lat": -19.0154, "lon": 29.1549}
    }

def filter_data_by_selections(df, income_category, selected_income, selected_region, selected_pub_type, selected_oa):
    """Apply filters to dataframe"""
    filtered_df = df.copy()

    if 'All' not in selected_oa and selected_oa:
        filtered_df = filtered_df[filtered_df['Open Access'].isin(selected_oa)]

    if 'All' not in selected_pub_type and selected_pub_type:
        filtered_df = filtered_df[filtered_df['Publication type'].isin(selected_pub_type)]

    if 'All' not in selected_region and selected_region and 'Region' in df.columns:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_region)]

    if income_category == "LMIC" and selected_income and 'Income Level' in df.columns:
        filtered_df = filtered_df[filtered_df['Income Level'].isin(selected_income)]

    return filtered_df

def calculate_map_data(df, map_display_type, regional_hubs_tuple):
    """Calculate country counts for map display - REMOVED CACHING for real-time updates"""
    map_filtered_df = df.copy()
    
    # Convert tuple back to list for processing
    regional_hubs = list(regional_hubs_tuple) if isinstance(regional_hubs_tuple, tuple) else regional_hubs_tuple
    
    if "All" not in regional_hubs and regional_hubs:
        hub_countries = get_regional_hub_countries()
        selected_countries = []
        for hub in regional_hubs:
            if hub in hub_countries:
                selected_countries.extend(hub_countries[hub])
        
        if selected_countries:
            map_filtered_df = map_filtered_df[map_filtered_df['Country'].isin(selected_countries)]

    if map_display_type == "Publications":
        country_counts = map_filtered_df.groupby('Country')['Publications'].sum().reset_index()
        country_counts.columns = ['Country', 'Count']
        display_label = "Publications"
    elif map_display_type == "Authors":
        country_counts = map_filtered_df.groupby('Country')['Name'].nunique().reset_index()
        country_counts.columns = ['Country', 'Count']
        display_label = "Authors"
    else:
        country_counts = map_filtered_df.groupby('Country')['Organization'].nunique().reset_index()
        country_counts.columns = ['Country', 'Count']
        display_label = "Organizations"
    
    country_counts = country_counts[country_counts['Country'] != 'Unknown']
    country_counts = country_counts[country_counts['Count'] > 0]  # Only countries with data
    country_counts = country_counts.sort_values('Count', ascending=False)
    
    return country_counts, display_label, map_filtered_df

def calculate_search_results(df, search_term, search_org, selected_countries_pills, regional_hubs_tuple):
    """Calculate search results - REMOVED CACHING for real-time updates"""
    search_results = df.copy()
    
    # Convert tuple back to list for processing
    regional_hubs = list(regional_hubs_tuple) if isinstance(regional_hubs_tuple, tuple) else regional_hubs_tuple
    
    if "All" not in regional_hubs and regional_hubs:
        hub_countries = get_regional_hub_countries()
        selected_countries = []
        for hub in regional_hubs:
            if hub in hub_countries:
                selected_countries.extend(hub_countries[hub])
        
        if selected_countries:
            search_results = search_results[search_results['Country'].isin(selected_countries)]

    if "All" not in selected_countries_pills and selected_countries_pills:
        search_results = search_results[search_results['Country'].isin(selected_countries_pills)]

    if search_term:
        search_results = search_results[
            (search_results['Name'].str.contains(search_term, case=False, na=False)) |
            (search_results['Organization'].str.contains(search_term, case=False, na=False))
        ]

    if search_org != 'All':
        search_results = search_results[search_results['Organization'] == search_org]
    
    return search_results
    

def process_grouped_data(search_results, display_type):
    """Process grouped data - SIMPLIFIED"""
    
    try:
        # Make a clean copy
        df = search_results.copy()
        
        # Check required columns exist
        required_cols = ['Name', 'Organization', 'Country', 'Publications', 'Citations']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame(), []
        
        # CRITICAL FIX: Convert category columns back to strings to avoid groupby issues
        for col in ['Name', 'Organization', 'Country']:
            if col in df.columns:
                if df[col].dtype.name == 'category':
                    df[col] = df[col].astype(str)
        
        # Clean data step by step
        df = df[df['Name'].notna()]
        df = df[df['Organization'].notna()]
        df = df[df['Country'].notna()]
        df = df[df['Name'] != 'Unknown']
        df = df[df['Organization'] != 'Unknown']
        df = df[df['Country'] != 'Unknown']
        
        # Convert numeric columns
        df['Publications'] = pd.to_numeric(df['Publications'], errors='coerce').fillna(0)
        df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce').fillna(0)
        
        # Remove any rows with 0 publications
        df = df[df['Publications'] > 0]
        
        if len(df) == 0:
            return pd.DataFrame(), []
        
        if display_type == "Authors":
            grouped_data = df.groupby(
                ['Name', 'Organization', 'Country'],
                as_index=False
            ).agg({
                'Publications': 'sum',
                'Citations': 'sum'
            })
            
            # Calculate Citations Mean
            grouped_data['Citations Mean'] = (
                grouped_data['Citations'] / grouped_data['Publications']
            ).round(2)
            
            display_cols = ['Name', 'Organization', 'Country', 'Publications', 'Citations', 'Citations Mean']
            
        else:  # Organizations
            grouped_data = df.groupby(
                ['Organization', 'Country'],
                as_index=False
            ).agg({
                'Publications': 'sum',
                'Citations': 'sum',
                'Name': 'nunique'
            })
            
            # Rename using rename method
            grouped_data = grouped_data.rename(columns={'Name': 'Authors'})
            
            # Calculate Citations Mean
            grouped_data['Citations Mean'] = (
                grouped_data['Citations'] / grouped_data['Publications']
            ).round(2)
            
            display_cols = ['Organization', 'Country', 'Authors', 'Publications', 'Citations', 'Citations Mean']

        # Sort and clean
        grouped_data = grouped_data.sort_values('Publications', ascending=False)
        grouped_data = grouped_data.reset_index(drop=True)
        
        # Fill any NaN values with 0
        grouped_data = grouped_data.fillna(0)

        return grouped_data, display_cols
        
    except Exception as e:
        st.error(f"Error in process_grouped_data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), []

# Helper function
@lru_cache(maxsize=128)
def handle_all_selection(current_selection_tuple, all_options_tuple):
    """Handle 'All' selection logic"""
    current_selection = list(current_selection_tuple)
    all_options = list(all_options_tuple)
    
    if not current_selection:
        return ['All']
    
    if 'All' in current_selection and len(current_selection) > 1:
        return [item for item in current_selection if item != 'All']
    
    if current_selection == ['All']:
        return ['All']
    
    non_all_options = [opt for opt in all_options if opt != 'All']
    if len(current_selection) == len(non_all_options) and all(item in non_all_options for item in current_selection):
        return ['All']
    
    return current_selection

# --- MAIN APP LOGIC ---

@st.cache_resource
def initialize_app():
    """Initialize app with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    default_path = "data/combined_all_publications_data_Final.csv"
    
    status_text.text("Loading data...")
    progress_bar.progress(25)
    
    try:
        if os.path.exists(default_path):
            df, country_name_mapping = load_and_preprocess_data(default_path)
            data_path = default_path
        elif os.path.exists(fallback_path):
            df, country_name_mapping = load_and_preprocess_data(fallback_path)
            data_path = fallback_path
        else:
            st.error(f"❌ File not found. Please ensure the data file exists.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    progress_bar.progress(75)
    status_text.text("Processing data...")
    
    if df is None or df.empty:
        st.error("❌ No data loaded.")
        st.stop()
    
    progress_bar.progress(100)
    status_text.text("✅ Data loaded successfully!")
    time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    return df, country_name_mapping, data_path

df, country_name_mapping, data_path = initialize_app()


@st.cache_data
def get_filter_options(df):
    """Get all filter options in one go"""
    options = {}
    
    if 'Region' in df.columns:
        options['regions'] = ['All'] + get_unique_values(df, 'Region')
    else:
        options['regions'] = ['All']
    
    if 'Publication type' in df.columns:
        options['pub_types'] = ['All'] + get_unique_values(df, 'Publication type')
    else:
        options['pub_types'] = ['All']
    
    if 'Open Access' in df.columns:
        options['oa_types'] = ['All'] + get_unique_values(df, 'Open Access')
    else:
        options['oa_types'] = ['All']
    
    if 'Income Level' in df.columns:
        lmic_income_levels = ['Upper middle income', 'Lower middle income', 'Low income']
        options['lmic_levels'] = [level for level in lmic_income_levels 
                                 if level in df['Income Level'].dropna().unique()]
    else:
        options['lmic_levels'] = []
    
    return options

filter_options = get_filter_options(df)

# --- SIDEBAR FILTERS ---

if 'Income Level' in df.columns:
    income_category = st.sidebar.pills(
        "Filter by Income Category:",
        ["All regions", "LMIC"],
        selection_mode="single",
        default="LMIC"
    )
    
    if income_category == "LMIC" and filter_options['lmic_levels']:
        selected_lmic_raw = st.sidebar.pills(
            "Narrow Income level:",
            filter_options['lmic_levels'],
            selection_mode="multi",
            default=filter_options['lmic_levels']
        )
        selected_income = selected_lmic_raw if selected_lmic_raw else filter_options['lmic_levels']
    else:
        selected_income = [] if income_category == "LMIC" else ['All']
else:
    income_category = "All regions"
    selected_income = ['All']

with st.sidebar:
    st.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

selected_region_raw = st.sidebar.pills(
    "Filter by Region:",
    filter_options['regions'],
    selection_mode="multi",
    default=['All']
)
selected_region = handle_all_selection(tuple(selected_region_raw), tuple(filter_options['regions']))

with st.sidebar:
    st.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

regional_hubs = st.sidebar.pills(
    "Regional Excellence Hub:",
    options=["All", "A*STAR SIgN", "Institut Pasteur Network", "KEMRI-Wellcome", "AHRI"],
    selection_mode="multi",
    default=["All"]
)

if "All" in regional_hubs and len(regional_hubs) > 1:
    regional_hubs = [item for item in regional_hubs if item != "All"]
elif not regional_hubs:
    regional_hubs = ["All"]

with st.sidebar:
    st.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

selected_pub_type_raw = st.sidebar.pills(
    "Publication Type:",
    filter_options['pub_types'],
    selection_mode="multi",
    default=['All']
)
selected_pub_type = handle_all_selection(tuple(selected_pub_type_raw), tuple(filter_options['pub_types']))

# Apply filters
filtered_df = filter_data_by_selections(df, income_category, selected_income, selected_region, selected_pub_type, ['All'])

# --- MAIN CONTENT ---

if 'Country' not in filtered_df.columns:
    st.error("Country information not available in the data")
else:
    col_map1, col_map2 = st.columns([1, 3])

    with col_map1:
        map_display_type = st.radio("",
            options=["Publications", "Authors", "Organizations"],
            index=1,
            horizontal=True
        )

        country_counts, display_label, map_filtered_df = calculate_map_data(
            filtered_df, map_display_type, tuple(regional_hubs)
        )

        hide_top_countries = "Show all countries"
        num_to_hide = 0

        if hide_top_countries == "Hide top countries" and len(country_counts) > num_to_hide:
            display_data = country_counts.iloc[num_to_hide:].copy()
        else:
            display_data = country_counts.copy()

        display_data['Log_Count'] = np.log10(display_data['Count'] + 1)

        st.markdown(f"Top 5 Countries (per {display_label}):")
        table_placeholder = st.empty()
        table_placeholder.dataframe(
            display_data.head(5)[['Country', 'Count']],
            hide_index=True,
            height=200
        )

        with st.popover("🔧 Display Options"):
            hide_top_countries = st.radio(
                "Display options:",
                options=["Show all countries", "Hide top countries"],
                index=0,
                horizontal=True,
                key="hide_option"
            )

            if hide_top_countries == "Hide top countries":
                max_countries_to_hide = min(10, len(country_counts) - 1)
                num_to_hide = st.number_input(
                    "Countries to hide:",
                    min_value=1,
                    max_value=max_countries_to_hide,
                    value=1,
                    step=1,
                    key="num_hide"
                )
            else:
                num_to_hide = 0

        if hide_top_countries == "Hide top countries" and len(country_counts) > num_to_hide:
            display_data = country_counts.iloc[num_to_hide:].copy()
            hidden_countries = country_counts.head(num_to_hide)
            st.caption(
                f"Hiding top {num_to_hide} countries: "
                f"{', '.join(hidden_countries['Country'].tolist())}"
            )
        else:
            display_data = country_counts.copy()

        display_data['Log_Count'] = np.log10(display_data['Count'] + 1)

        table_placeholder.dataframe(
            display_data.head(5)[['Country', 'Count']],
            hide_index=True,
            height=200
        )

    with col_map2:
        # Only use display_data (which may have top countries hidden)
        fig_heatmap = px.choropleth(
            display_data,
            locations='Country',
            locationmode='country names',
            color='Log_Count',
            hover_name='Country',
            hover_data={
                'Count': ':,',
                'Country': False,
                'Log_Count': False
            },
            color_continuous_scale=[[0, '#f0f8ff'], [0.5, '#82C5E0'], [1, '#4682b4']],
            labels={'Log_Count': f'{display_label} (log scale)'}
        )
        
        hub_countries = get_regional_hub_countries()
        country_coords = get_country_coordinates()
        hub_colors = {
            "A*STAR SIgN": "#f95738",
            "Institut Pasteur Network": "#c77dff", 
            "KEMRI-Wellcome": "#f4d35e",
            "AHRI": "#4c956c"
        }
        
        # IMPORTANT: Only use display_data (after hiding top countries if applicable)
        countries_with_data = set(display_data['Country'].tolist())
        
        hubs_to_show = []
        if "All" in regional_hubs:
            hubs_to_show = list(hub_countries.keys())
        else:
            hubs_to_show = regional_hubs
        
        for hub_name in hubs_to_show:
            if hub_name in hub_countries and hub_name in hub_colors:
                covered_countries = hub_countries[hub_name]
                # Only show markers for countries that have data in display_data
                countries_to_mark = [country for country in covered_countries 
                                   if country in countries_with_data and country in country_coords]
                
                if countries_to_mark:
                    lats = [country_coords[country]["lat"] for country in countries_to_mark]
                    lons = [country_coords[country]["lon"] for country in countries_to_mark]
                    
                    fig_heatmap.add_trace(
                        go.Scattergeo(
                            lon=lons,
                            lat=lats,
                            mode='markers',
                            marker=dict(
                                size=7,
                                color=hub_colors[hub_name],
                                symbol='diamond',
                                line=dict(width=1, color='black'),
                                opacity=0.8
                            ),
                            name=f"{hub_name} Coverage",
                            hovertemplate="<b>%{text}</b><br>" +
                                        f"{hub_name} Coverage Area<br>" +
                                        "<extra></extra>",
                            text=countries_to_mark,
                            showlegend=True
                        )
                    )
        
        # Update map to fit only to countries with data
        fig_heatmap.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            showcountries=True,
            countrycolor='rgba(200, 200, 200, 0.5)',
            coastlinecolor='rgba(200, 200, 200, 0.5)',
            showlakes=False,
            fitbounds="locations",  # This will fit to the choropleth data
            visible=True
        )
        
        fig_heatmap.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(
                title=f"{display_label}<br>(log scale)",
                tickvals=[0, 1, 2, 3, 4],
                ticktext=['1', '10', '100', '1K', '10K']
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    st.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

    # Second section with search and display
    search_col1, search_col2, search_col3 = st.columns(3)

    with search_col1:
        search_term = st.text_input("Search by name or organization:", placeholder="Enter search term...")

    with search_col2:
        if 'Organization' in filtered_df.columns:
            available_orgs = get_unique_values(filtered_df, 'Organization')
            search_org = st.selectbox(
                "Filter by Organization:",
                options=['All'] + available_orgs[:100],
                index=0
            )
        else:
            search_org = 'All'
        
    available_countries_for_pills = get_unique_values(map_filtered_df, 'Country') if 'map_filtered_df' in locals() else get_unique_values(filtered_df, 'Country')
    available_countries_for_pills = [country for country in available_countries_for_pills if country != 'Unknown']

    col1_table, col2_table = st.columns([1, 2])
    with col2_table:
        if len(available_countries_for_pills) > 12:
            with st.popover("🌍 Select Countries"):
                selected_countries_pills = st.pills(
                    "Filter by Countries:",
                    options=["All"] + available_countries_for_pills,
                    selection_mode="multi",
                    default=["All"]
                )
        else:
            selected_countries_pills = st.pills(
                "",
                options=["All"] + available_countries_for_pills,
                selection_mode="multi",
                label_visibility="collapsed",
                default=["All"]
            )
        
        if "All" in selected_countries_pills and len(selected_countries_pills) > 1:
            selected_countries_pills = [item for item in selected_countries_pills if item != "All"]
        elif not selected_countries_pills:
            selected_countries_pills = ["All"]

    display_type = col1_table.radio(
        "",
        options=["Organizations","Authors"],
        index=0,
        horizontal=True, 
        label_visibility="collapsed"
    )

    # Calculate fresh each time
    search_results = calculate_search_results(
        filtered_df, search_term, search_org, selected_countries_pills, tuple(regional_hubs)
    )

    # Don't cache the groupby - do it fresh
    grouped_data, display_cols = process_grouped_data(search_results, display_type)

    with st.expander("View Detailed Table", expanded=False):
        if len(grouped_data) > 0:
            # CORRECT COUNT - from grouped_data, not search_results
            result_count = len(grouped_data)
            col1_table.write(f"Found {result_count} {display_type.lower()} matching your criteria")
            
            display_results = grouped_data.copy()
            
            # Add Search column for both Authors and Organizations
            if display_type == "Authors":
                display_results['Search'] = display_results.apply(
                    lambda row: f"https://www.google.com/search?q={row['Name'].replace(' ', '+')}+{row['Organization'].replace(' ', '+')}",
                    axis=1
                )
                available_display_cols = [col for col in display_cols + ['Search'] if col in display_results.columns]
            else:  # Organizations
                display_results['Search'] = display_results.apply(
                    lambda row: f"https://www.google.com/search?q={row['Organization'].replace(' ', '+')}+{row['Country'].replace(' ', '+')}",
                    axis=1
                )
                available_display_cols = [col for col in display_cols + ['Search'] if col in display_results.columns]
            
            dynamic_height = min(max(len(display_results) * 35 + 50, 200), 800)
            
            # Use data_editor for both to show the clickable link
            st.data_editor(
                display_results[available_display_cols],
                hide_index=True,
                height=dynamic_height,
                use_container_width=True,
                disabled=True,
                column_config={
                    "Search": st.column_config.LinkColumn(
                        "🔍",
                        display_text="Google Search"
                    )
                }
            )
        else:
            st.write(f"No {display_type.lower()} match your search criteria")

    with st.expander("Visualize Publications and Citations Mean by Organization", expanded=False):
        if len(grouped_data) > 0:
            if display_type == "Organizations":
                plot_data = grouped_data.copy()
                plot_data = plot_data.sort_values('Citations Mean', ascending=True)

                fig_scatter = px.scatter(
                    plot_data,
                    x='Citations Mean',
                    y='Organization',
                    size='Publications',
                    hover_data={
                        'Country': True,
                        'Authors': True,
                        'Publications': ':,',
                        'Citations': ':,',
                        'Citations Mean': ':.2f'
                    },
                    size_max=30,
                    title=f'Organizations: Publications (dot size) vs Citations Mean',
                )
                
            else:
                plot_data = grouped_data.copy()
                plot_data = plot_data.sort_values('Citations Mean', ascending=True)
                
                fig_scatter = px.scatter(
                    plot_data,
                    x='Citations Mean',
                    y='Name',
                    size='Publications',
                    hover_data={
                        'Organization': True,
                        'Country': True,
                        'Publications': ':,',
                        'Citations': ':,',
                        'Citations Mean': ':.2f'
                    },
                    size_max=30,
                    title=f'Top 50 Authors: Publications (dot size) vs Citations Mean',
                )
            
            fig_scatter.update_layout(
                height=max(400, len(plot_data) * 20),
                margin=dict(l=200, r=50, t=50, b=50),
                xaxis_title="Average Citations per Publication",
                yaxis_title=None,
                showlegend=False
            )
            
            fig_scatter.update_traces(
                marker=dict(
                    opacity=1,
                    line=dict(width=1, color='white'),
                    color='#82C5E0'
                )
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"💡 Dot size represents number of publications. Hover over dots for detailed information.")
