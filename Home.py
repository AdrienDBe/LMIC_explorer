import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import time
import gc
import tracemalloc

# Emergency stop if too many reruns
if st.session_state.rerun_count > 50:
    st.error("⚠️ Too many page reloads detected. Clearing cache...")
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.session_state.rerun_count = 0
    st.stop()

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
/* ==================== HIDE/FIX STREAMLIT TOP BAR ==================== */

/* Hide the top toolbar completely */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Remove top padding to prevent cutoff */
.main .block-container {
    padding-top: 2rem !important;
}

/* Ensure content starts at top */
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
}

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

div.stMain {
    background-color: #FFFFFF !important;
    padding: 1rem !important;
}

div.block-container {
    background-color: #FFFFFF !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stAppViewContainer"] > * {
    margin-top: 0 !important;
}

[data-testid="stRadio"],
[data-testid="stPills"] {
    margin-top: 0.5rem !important;
    padding-top: 0.5rem !important;
}

[data-testid="stHeader"] {
    visibility: visible !important;
    display: block !important;
    margin-bottom: 1rem !important;
}

div[data-testid="stToolbar"] {
    visibility: visible !important;
}

p, span, label, div {
    color: #262730 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #262730 !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #262730 !important;
}

.stPills button {
    padding: 0.5rem 1rem !important;
    border-radius: 20px !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
}

.stPills button[kind="pills"],
button.st-emotion-cache-b0zc2i.e1mwqyj910 {
    background-color: #F0F0F0 !important;
    color: #262730 !important;
    border: 1px solid #D3D3D3 !important;
}

.stPills button[kind="pills"]:hover,
button.st-emotion-cache-b0zc2i.e1mwqyj910:hover {
    background-color: #E8E8E8 !important;
    color: #262730 !important;
    border: 1px solid #B0B0B0 !important;
}

button.st-emotion-cache-tx7mgd.e1mwqyj911,
button.st-emotion-cache-tx7mgd.e1mwqyj911[kind="pillsActive"],
.stPills button[kind="pillsActive"],
button[kind="pillsActive"][data-testid="stBaseButton-pillsActive"] {
    background-color: #82C5E0 !important;
    color: #FFFFFF !important;
    border: 1px solid #82C5E0 !important;
    font-weight: 600 !important;
}

button.st-emotion-cache-tx7mgd.e1mwqyj911:hover,
.stPills button[kind="pillsActive"]:hover,
button[kind="pillsActive"][data-testid="stBaseButton-pillsActive"]:hover {
    background-color: #6BADCC !important;
    color: #FFFFFF !important;
    border: 1px solid #6BADCC !important;
}

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

[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #262730 !important;
}

[data-testid="stPopover"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D3D3D3 !important;
}

.plotly {
    background-color: #FFFFFF !important;
}

.plotly .bg {
    fill: #FFFFFF !important;
}

.plotly-notebooklogo {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("")

# --- CACHING FUNCTIONS ---

@st.cache_data
def load_and_preprocess_data(filepath):
    """Load the LMIC-only CSV file"""
    try:
        df = pd.read_csv(filepath, low_memory=False)
        
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Clean numeric columns
        numeric_columns = ['Publications', 'Citations', 'Citations Mean']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Clean text columns
        text_columns = ['Name', 'Organization', 'Publication type']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        
        if 'Country' in df.columns:
            df['Country'] = df['Country'].fillna('Unknown').astype(str).str.strip()
            df['Country'] = df['Country'].replace('', 'Unknown')
        
        if 'Region' in df.columns:
            df['Region'] = df['Region'].fillna('Unknown').astype(str).str.strip()
        
        if 'Income Level' in df.columns:
            df['Income Level'] = df['Income Level'].fillna('Unknown').astype(str).str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def get_unique_values(df, column):
    """Get unique values for a column with caching"""
    if column in df.columns:
        return sorted([str(x) for x in df[column].dropna().unique()])
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
    
def filter_data_by_selections(df, selected_income, selected_region, selected_pub_type):
    """Apply filters to dataframe"""
    filtered_df = df.copy()

    if 'All' not in selected_pub_type and selected_pub_type:
        filtered_df = filtered_df[filtered_df['Publication type'].isin(selected_pub_type)]

    if 'All' not in selected_region and selected_region and 'Region' in df.columns:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_region)]

    if 'All' not in selected_income and selected_income and 'Income Level' in df.columns:
        filtered_df = filtered_df[filtered_df['Income Level'].isin(selected_income)]

    return filtered_df

def calculate_map_data(df, map_display_type, regional_hubs):
    """Calculate country counts for map display"""
    map_filtered_df = df.copy()
    
    if "All" not in regional_hubs and regional_hubs:
        hub_countries = get_regional_hub_countries()
        selected_countries = []
        for hub in regional_hubs:
            if hub in hub_countries:
                selected_countries.extend(hub_countries[hub])
        
        if selected_countries:
            map_filtered_df = df[df['Country'].isin(selected_countries)].copy()

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
    country_counts = country_counts[country_counts['Count'] > 0]
    country_counts = country_counts.sort_values('Count', ascending=False)
    
    return country_counts, display_label, map_filtered_df

def calculate_search_results(df, search_term, search_org, selected_countries_pills, regional_hubs):
    """Calculate search results"""
    search_results = df.copy()
    
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
    """Process grouped data with proper aggregation"""
    try:
        df = search_results.copy()
        
        required_cols = ['Name', 'Organization', 'Country', 'Publications', 'Citations']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame(), []
        
        # Convert category columns to strings
        for col in ['Name', 'Organization', 'Country']:
            if col in df.columns:
                if df[col].dtype.name == 'category':
                    df[col] = df[col].astype(str)
        
        # Clean data
        df = df[df['Name'].notna()]
        df = df[df['Organization'].notna()]
        df = df[df['Country'].notna()]
        df = df[df['Name'] != 'Unknown']
        df = df[df['Organization'] != 'Unknown']
        df = df[df['Country'] != 'Unknown']
        
        # Convert numeric columns
        df['Publications'] = pd.to_numeric(df['Publications'], errors='coerce').fillna(0)
        df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce').fillna(0)
        
        # Remove rows with 0 publications
        df = df[df['Publications'] > 0]
        
        if len(df) == 0:
            return pd.DataFrame(), []
        
        if display_type == "Authors":
            # Group by Name, Organization, Country
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
            # Group by Organization and Country
            grouped_data = df.groupby(
                ['Organization', 'Country'],
                as_index=False
            ).agg({
                'Publications': 'sum',
                'Citations': 'sum',
                'Name': 'nunique'  # Count unique authors
            })
            
            # Rename Name column to Authors
            grouped_data = grouped_data.rename(columns={'Name': 'Authors'})
            
            # Calculate Citations Mean
            grouped_data['Citations Mean'] = (
                grouped_data['Citations'] / grouped_data['Publications']
            ).round(2)
            
            display_cols = ['Organization', 'Country', 'Authors', 'Publications', 'Citations', 'Citations Mean']

        # Sort and clean
        grouped_data = grouped_data.sort_values('Publications', ascending=False)
        grouped_data = grouped_data.reset_index(drop=True)
        grouped_data = grouped_data.fillna(0)

        return grouped_data, display_cols
        
    except Exception as e:
        st.error(f"Error in process_grouped_data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), []
        
def handle_all_selection(current_selection_tuple, all_options_tuple):
    """Handle 'All' selection logic"""
    current_selection = list(current_selection_tuple)
    all_options = list(all_options_tuple)
    
    if current_selection == ['All'] or (not current_selection):
        return ['All']
    
    if 'All' in current_selection and len(current_selection) > 1:
        return [item for item in current_selection if item != 'All']
    
    non_all_options = [opt for opt in all_options if opt != 'All']
    if len(current_selection) == len(non_all_options) and all(item in non_all_options for item in current_selection):
        return ['All']
    
    return current_selection

# --- INITIALIZE APP ---

@st.cache_resource
def initialize_app():
    """Initialize app with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    default_path = "data/combined_all_publications_data_Final_LMIC_Only.csv"
    
    status_text.text("Loading data...")
    progress_bar.progress(25)
    
    try:
        if os.path.exists(default_path):
            df = load_and_preprocess_data(default_path)
            data_path = default_path
        else:
            st.error(f"❌ File not found at {default_path}")
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
    
    return df, data_path

df, data_path = initialize_app()

@st.cache_data
def get_filter_options(df):
    """Get all filter options"""
    options = {}
    
    if 'Region' in df.columns:
        options['regions'] = ['All'] + get_unique_values(df, 'Region')
    else:
        options['regions'] = ['All']
    
    if 'Publication type' in df.columns:
        options['pub_types'] = ['All'] + get_unique_values(df, 'Publication type')
    else:
        options['pub_types'] = ['All']
    
    if 'Income Level' in df.columns:
        options['income_levels'] = ['All'] + get_unique_values(df, 'Income Level')
    else:
        options['income_levels'] = ['All']
    
    return options

filter_options = get_filter_options(df)

# === INITIALIZE FILTER STATE IN SESSION ===
if 'selected_income' not in st.session_state:
    st.session_state.selected_income = ['All']

if 'selected_region' not in st.session_state:
    st.session_state.selected_region = ['All']

if 'regional_hubs' not in st.session_state:
    st.session_state.regional_hubs = ['All']

if 'selected_pub_type' not in st.session_state:
    st.session_state.selected_pub_type = ['All']

# --- SIDEBAR FILTERS ---

st.sidebar.markdown("### Filters")

# Filter by Income Category (renamed from "Narrow Income level")
selected_income_raw = st.sidebar.pills(
    "Filter by Income Category:",
    filter_options['income_levels'],
    selection_mode="multi",
    key="income_category_input"
)
selected_income = handle_all_selection(
    tuple(selected_income_raw) if selected_income_raw else ('All',),
    tuple(filter_options['income_levels'])
)
st.session_state.selected_income = selected_income

st.sidebar.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

# Region
selected_region_raw = st.sidebar.pills(
    "Filter by Region:",
    filter_options['regions'],
    selection_mode="multi",
    key="region_input"
)
selected_region = handle_all_selection(
    tuple(selected_region_raw) if selected_region_raw else ('All',),
    tuple(filter_options['regions'])
)
st.session_state.selected_region = selected_region

st.sidebar.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

# Regional Hubs
regional_hubs_raw = st.sidebar.pills(
    "Regional Excellence Hub:",
    options=["All", "A*STAR SIgN", "Institut Pasteur Network", "KEMRI-Wellcome", "AHRI"],
    selection_mode="multi",
    key="regional_hubs_input"
)

if "All" in regional_hubs_raw and len(regional_hubs_raw) > 1:
    regional_hubs = [item for item in regional_hubs_raw if item != "All"]
elif not regional_hubs_raw:
    regional_hubs = ["All"]
else:
    regional_hubs = regional_hubs_raw

st.session_state.regional_hubs = regional_hubs

st.sidebar.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

# Publication Type
selected_pub_type_raw = st.sidebar.pills(
    "Publication Type:",
    filter_options['pub_types'],
    selection_mode="multi",
    key="pub_type_input"
)
selected_pub_type = handle_all_selection(
    tuple(selected_pub_type_raw) if selected_pub_type_raw else ('All',),
    tuple(filter_options['pub_types'])
)
st.session_state.selected_pub_type = selected_pub_type

# === APPLY FILTERS ===
filtered_df = filter_data_by_selections(
    df, selected_income, selected_region, selected_pub_type
)

log_memory("After filtering")

# --- MAIN CONTENT ---

if 'Country' not in filtered_df.columns:
    st.error("Country information not available in the data")
else:
    col_map1, col_map2 = st.columns([1, 3])

    with col_map1:
        map_display_type = st.radio(
            "Display Type:",
            options=["Publications", "Authors", "Organizations"],
            index=1,
            key="map_display_type_radio"
        )
    
        # Calculate map data
        country_counts, display_label, map_filtered_df = calculate_map_data(
            filtered_df, map_display_type, regional_hubs
        )
            
        display_data = country_counts.copy()
        display_data['Log_Count'] = np.log10(display_data['Count'] + 1)

        st.markdown(f"Top 5 Countries (per {display_label}):")
        st.dataframe(
            display_data.head(5)[['Country', 'Count']],
            hide_index=True,
            height=200
        )

    with col_map2:
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
        
        countries_with_data = set(display_data['Country'].tolist())
        
        hubs_to_show = list(hub_countries.keys()) if "All" in regional_hubs else regional_hubs
        
        for hub_name in hubs_to_show:
            if hub_name in hub_countries and hub_name in hub_colors:
                covered_countries = hub_countries[hub_name]
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
        
        fig_heatmap.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            showcountries=True,
            countrycolor='rgba(200, 200, 200, 0.5)',
            coastlinecolor='rgba(200, 200, 200, 0.5)',
            showlakes=False,
            fitbounds="locations",
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
        del fig_heatmap
        gc.collect()

        
    st.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

    # Search and display section
    search_col1, search_col2 = st.columns(2)

    with search_col1:
        search_term = st.text_input("Search by name or organization:", placeholder="Enter search term...")

    with search_col2:
        if 'Organization' in filtered_df.columns:
            available_orgs = get_unique_values(filtered_df, 'Organization')
            search_org = st.selectbox(
                "Filter by Organization:",
                options=['All'] + available_orgs,
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
                    default=["All"],
                    key="countries_pills_input"
                )
        else:
            selected_countries_pills = st.pills(
                "Filter by Countries:",
                options=["All"] + available_countries_for_pills,
                selection_mode="multi",
                default=["All"],
                key="countries_pills_input"
            )
        
        if "All" in selected_countries_pills and len(selected_countries_pills) > 1:
            selected_countries_pills = [item for item in selected_countries_pills if item != "All"]
        elif not selected_countries_pills:
            selected_countries_pills = ["All"]

    display_type = col1_table.radio(
        "Display Type:",
        options=["Organizations","Authors"],
        index=0,
        key="display_type_radio"
    )

    # Calculate search results
    search_results = calculate_search_results(
        filtered_df, search_term, search_org, selected_countries_pills, regional_hubs
    )
    
    # Process grouped data
    grouped_data, display_cols = process_grouped_data(search_results, display_type)

    with st.expander("View Detailed Table", expanded=False):
        if len(grouped_data) > 0:
            result_count = len(grouped_data)
            st.write(f"Found {result_count} {display_type.lower()} matching your criteria")
            
            display_results = grouped_data.copy()
            
            if display_type == "Authors":
                display_results['Search'] = display_results.apply(
                    lambda row: f"https://www.google.com/search?q={row['Name'].replace(' ', '+')}+{row['Organization'].replace(' ', '+')}",
                    axis=1
                )
                available_display_cols = [col for col in display_cols + ['Search'] if col in display_results.columns]
            else:
                display_results['Search'] = display_results.apply(
                    lambda row: f"https://www.google.com/search?q={row['Organization'].replace(' ', '+')}+{row['Country'].replace(' ', '+')}",
                    axis=1
                )
                available_display_cols = [col for col in display_cols + ['Search'] if col in display_results.columns]
            
            dynamic_height = min(max(len(display_results) * 35 + 50, 200), 800)
            
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

    with st.expander("Visualize Publications and Citations Mean", expanded=False):
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
                    title=f'Authors: Publications (dot size) vs Citations Mean',
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
                    line=dict(width=0.75, color='black'),
                    color='#82C5E0'
                )
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            del fig_scatter
            gc.collect()
            st.caption(f"💡 Dot size represents number of publications. Hover over dots for detailed information.")
