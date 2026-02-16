import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import requests
import time
import gc
import tracemalloc

tracemalloc.start()

def get_memory_usage():
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024**2, peak / 1024**2

st.set_page_config(
    page_title="LMIC Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# Debug panel
with st.sidebar.expander("🐛 Debug Info", expanded=False):
    st.write(f"**Rerun count:** {st.session_state.get('rerun_count', 0)}")
    current_mem, peak_mem = get_memory_usage()
    st.write(f"**Memory:** {current_mem:.1f}MB / {peak_mem:.1f}MB")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        gc.collect()
        st.rerun()

# CSS
st.markdown("""<style>
html, body { background-color: #FFFFFF !important; color: #262730 !important; }
[data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; }
[data-testid="stSidebar"] { background-color: #F0F2F6 !important; }
</style>""", unsafe_allow_html=True)

# Track reruns
if 'rerun_count' not in st.session_state:
    st.session_state.rerun_count = 0
st.session_state.rerun_count += 1

# Emergency stop
if st.session_state.rerun_count > 100:
    st.error("Too many reruns. Clearing...")
    st.cache_data.clear()
    st.session_state.clear()
    st.stop()

# --- LOAD DATA ---
@st.cache_resource
def load_data():
    default_path = "data/combined_all_publications_data_Final.csv"
    if not os.path.exists(default_path):
        st.error("File not found")
        st.stop()
    
    df = pd.read_csv(default_path, low_memory=False)
    df = df.drop_duplicates(subset=['Name', 'Organization', 'Country', 'Publications'])
    df = df[df['Country'] != '']
    
    # World Bank data
    try:
        url = "http://api.worldbank.org/v2/country?per_page=400&format=json"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        records = [{
            "name": c["name"],
            "region": c["region"]["value"],
            "incomeLevel": c["incomeLevel"]["value"]
        } for c in data[1]]
        df_wb = pd.DataFrame(records)
        df_wb = df_wb[df_wb['region'] != 'Aggregates']
        
        df = df.merge(df_wb, left_on='Country', right_on='name', how='left')
        df = df.rename(columns={'region': 'Region', 'incomeLevel': 'Income Level'})
        df['Income Level'] = df['Income Level'].fillna('Low income')
        if 'name' in df.columns:
            df = df.drop('name', axis=1)
    except:
        pass
    
    return df

df = load_data()

@st.cache_data
def get_unique_vals(col):
    if col in df.columns:
        return sorted([str(x) for x in df[col].dropna().unique()])
    return []

@st.cache_data
def get_filter_options():
    return {
        'regions': ['All'] + get_unique_vals('Region'),
        'pub_types': ['All'] + get_unique_vals('Publication type'),
        'lmic_levels': ['Upper middle income', 'Lower middle income', 'Low income']
    }

@st.cache_data
def get_regional_hub_countries():
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

filter_opts = get_filter_options()

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("### Filters")

income_cat = st.sidebar.radio("Income Category:", ["All regions", "LMIC"], index=0)

if income_cat == "LMIC":
    sel_income = st.sidebar.multiselect("Income level:", filter_opts['lmic_levels'], default=filter_opts['lmic_levels'])
else:
    sel_income = ['All']

st.sidebar.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

sel_region = st.sidebar.multiselect("Region:", filter_opts['regions'], default=['All'])
if "All" in sel_region and len(sel_region) > 1:
    sel_region = [x for x in sel_region if x != "All"]
elif not sel_region:
    sel_region = ['All']

st.sidebar.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

hubs_raw = st.sidebar.multiselect("Regional Hub:", 
    ["All", "A*STAR SIgN", "Institut Pasteur Network", "KEMRI-Wellcome", "AHRI"],
    default=["All"])
regional_hubs = hubs_raw if "All" in hubs_raw else (hubs_raw if hubs_raw else ["All"])
if "All" in regional_hubs and len(regional_hubs) > 1:
    regional_hubs = [x for x in regional_hubs if x != "All"]

st.sidebar.markdown("<hr style='margin:0.3rem 0;'>", unsafe_allow_html=True)

sel_pub_type = st.sidebar.multiselect("Publication Type:", filter_opts['pub_types'], default=['All'])
if "All" in sel_pub_type and len(sel_pub_type) > 1:
    sel_pub_type = [x for x in sel_pub_type if x != "All"]
elif not sel_pub_type:
    sel_pub_type = ['All']

# --- FILTER DATA ---
def apply_filters():
    filtered = df.copy()
    
    if 'All' not in sel_pub_type and sel_pub_type:
        filtered = filtered[filtered['Publication type'].isin(sel_pub_type)]
    
    if 'All' not in sel_region and sel_region and 'Region' in df.columns:
        filtered = filtered[filtered['Region'].isin(sel_region)]
    
    if income_cat == "LMIC" and sel_income and 'Income Level' in df.columns:
        filtered = filtered[filtered['Income Level'].isin(sel_income)]
    
    return filtered

filtered_df = apply_filters()

# --- MAIN CONTENT ---
if 'Country' not in filtered_df.columns:
    st.error("No country data")
else:
    col_map1, col_map2 = st.columns([1, 3])
    
    with col_map1:
        map_type = st.radio("Display:", ["Publications", "Authors", "Organizations"], index=1)
        
        # Calculate country counts
        if map_type == "Publications":
            country_counts = filtered_df.groupby('Country')['Publications'].sum().reset_index()
            country_counts.columns = ['Country', 'Count']
            display_label = "Publications"
        elif map_type == "Authors":
            country_counts = filtered_df.groupby('Country')['Name'].nunique().reset_index()
            country_counts.columns = ['Country', 'Count']
            display_label = "Authors"
        else:
            country_counts = filtered_df.groupby('Country')['Organization'].nunique().reset_index()
            country_counts.columns = ['Country', 'Count']
            display_label = "Organizations"
        
        country_counts = country_counts[country_counts['Country'] != 'Unknown']
        country_counts = country_counts[country_counts['Count'] > 0]
        country_counts = country_counts.sort_values('Count', ascending=False)
        
        st.markdown(f"Top 5 Countries (per {display_label}):")
        st.dataframe(country_counts.head(5), hide_index=True, height=200)
    
    with col_map2:
        country_counts['Log_Count'] = np.log10(country_counts['Count'] + 1)
        
        fig = px.choropleth(
            country_counts,
            locations='Country',
            locationmode='country names',
            color='Log_Count',
            hover_name='Country',
            hover_data={'Count': ':,', 'Country': False, 'Log_Count': False},
            color_continuous_scale=[[0, '#f0f8ff'], [0.5, '#82C5E0'], [1, '#4682b4']],
            labels={'Log_Count': f'{display_label} (log scale)'}
        )
        
        # Add regional hub markers
        hub_countries = get_regional_hub_countries()
        country_coords = get_country_coordinates()
        hub_colors = {
            "A*STAR SIgN": "#f95738",
            "Institut Pasteur Network": "#c77dff", 
            "KEMRI-Wellcome": "#f4d35e",
            "AHRI": "#4c956c"
        }
        
        countries_with_data = set(country_counts['Country'].tolist())
        hubs_to_show = list(hub_countries.keys()) if "All" in regional_hubs else regional_hubs
        
        for hub_name in hubs_to_show:
            if hub_name in hub_countries and hub_name in hub_colors:
                covered_countries = hub_countries[hub_name]
                countries_to_mark = [c for c in covered_countries if c in countries_with_data and c in country_coords]
                
                if countries_to_mark:
                    lats = [country_coords[c]["lat"] for c in countries_to_mark]
                    lons = [country_coords[c]["lon"] for c in countries_to_mark]
                    
                    fig.add_trace(
                        go.Scattergeo(
                            lon=lons,
                            lat=lats,
                            mode='markers',
                            marker=dict(size=7, color=hub_colors[hub_name], symbol='diamond', 
                                       line=dict(width=1, color='black'), opacity=0.8),
                            name=f"{hub_name} Coverage",
                            hovertemplate="<b>%{text}</b><br>" + f"{hub_name}<br><extra></extra>",
                            text=countries_to_mark,
                            showlegend=True
                        )
                    )
        
        fig.update_geos(
            showframe=False, showcoastlines=True, projection_type='natural earth',
            showcountries=True, countrycolor='rgba(200, 200, 200, 0.5)',
            coastlinecolor='rgba(200, 200, 200, 0.5)', fitbounds="locations"
        )
        
        fig.update_layout(
            height=500, margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(title=f"{display_label}<br>(log scale)"),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                       bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        del fig
        gc.collect()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Search section
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        search_term = st.text_input("Search:", placeholder="Name or organization...")
    
    with col_s2:
        if 'Organization' in filtered_df.columns:
            orgs = sorted([str(x) for x in filtered_df['Organization'].dropna().unique()])
            search_org = st.selectbox("Organization:", ['All'] + orgs, index=0)
        else:
            search_org = 'All'
    
    # Search results
    results = filtered_df.copy()
    
    if search_term:
        results = results[
            (results['Name'].str.contains(search_term, case=False, na=False)) |
            (results['Organization'].str.contains(search_term, case=False, na=False))
        ]
    
    if search_org != 'All':
        results = results[results['Organization'] == search_org]
    
    display_type = st.radio("Show:", ["Organizations", "Authors"], index=0, horizontal=True)
    
    # Group results
    if len(results) > 0:
        if display_type == "Authors":
            grouped = results.groupby(['Name', 'Organization', 'Country'], as_index=False).agg({
                'Publications': 'sum',
                'Citations': 'sum'
            })
            grouped['Citations Mean'] = (grouped['Citations'] / grouped['Publications']).round(2)
            display_cols = ['Name', 'Organization', 'Country', 'Publications', 'Citations', 'Citations Mean']
        else:
            grouped = results.groupby(['Organization', 'Country'], as_index=False).agg({
                'Publications': 'sum',
                'Citations': 'sum',
                'Name': 'nunique'
            })
            grouped = grouped.rename(columns={'Name': 'Authors'})
            grouped['Citations Mean'] = (grouped['Citations'] / grouped['Publications']).round(2)
            display_cols = ['Organization', 'Country', 'Authors', 'Publications', 'Citations', 'Citations Mean']
        
        grouped = grouped.sort_values('Publications', ascending=False)
        
        with st.expander(f"View Detailed Results ({len(grouped)} {display_type.lower()} found)", expanded=False):
            st.dataframe(grouped[display_cols], use_container_width=True, height=600)
        
        with st.expander("Visualize Publications and Citations Mean", expanded=False):
            st.write(f"Showing top results visualization")
            plot_data = grouped.copy().sort_values('Citations Mean', ascending=True)
            
            if display_type == "Organizations":
                y_col = 'Organization'
                title = 'Organizations: Publications (dot size) vs Citations Mean'
            else:
                y_col = 'Name'
                title = 'Authors: Publications (dot size) vs Citations Mean'
            
            fig = px.scatter(
                plot_data,
                x='Citations Mean',
                y=y_col,
                size='Publications',
                hover_data=['Publications', 'Citations', 'Citations Mean'],
                size_max=30,
                title=title
            )
            fig.update_layout(
                height=max(400, min(len(plot_data) * 20, 1000)),
                margin=dict(l=200, r=50, t=50, b=50),
                xaxis_title="Average Citations per Publication",
                yaxis_title=None,
                showlegend=False
            )
            fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='white'), color='#82C5E0'))
            st.plotly_chart(fig, use_container_width=True)
            del fig
            gc.collect()
    else:
        st.info("No results found")
