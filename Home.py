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
        return sorted([x for x in df[col].unique() if pd.notna(x)])
    return []

@st.cache_data
def get_filter_options():
    return {
        'regions': ['All'] + get_unique_vals('Region'),
        'pub_types': ['All'] + get_unique_vals('Publication type'),
        'lmic_levels': ['Upper middle income', 'Lower middle income', 'Low income']
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
        
        # Simple calculation without caching
        if map_type == "Publications":
            country_counts = filtered_df.groupby('Country')['Publications'].sum().reset_index()
        elif map_type == "Authors":
            country_counts = filtered_df.groupby('Country')['Name'].nunique().reset_index()
            country_counts.columns = ['Country', 'Publications']
        else:
            country_counts = filtered_df.groupby('Country')['Organization'].nunique().reset_index()
            country_counts.columns = ['Country', 'Publications']
        
        country_counts = country_counts[country_counts['Country'] != 'Unknown']
        country_counts = country_counts[country_counts['Publications'] > 0]
        country_counts = country_counts.sort_values('Publications', ascending=False)
        
        st.markdown(f"Top 5 Countries:")
        st.dataframe(country_counts.head(5), hide_index=True, height=200)
    
    with col_map2:
        country_counts['Log_Count'] = np.log10(country_counts['Publications'] + 1)
        
        fig = px.choropleth(
            country_counts,
            locations='Country',
            locationmode='country names',
            color='Log_Count',
            hover_name='Country',
            color_continuous_scale=[[0, '#f0f8ff'], [0.5, '#82C5E0'], [1, '#4682b4']],
        )
        
        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            showcountries=True,
            fitbounds="locations"
        )
        
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
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
            orgs = sorted(filtered_df['Organization'].unique())[:100]
            search_org = st.selectbox("Organization:", ['All'] + list(orgs), index=0)
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
            grouped['Cit_Mean'] = (grouped['Citations'] / grouped['Publications']).round(2)
            display_cols = ['Name', 'Organization', 'Country', 'Publications', 'Citations', 'Cit_Mean']
        else:
            grouped = results.groupby(['Organization', 'Country'], as_index=False).agg({
                'Publications': 'sum',
                'Citations': 'sum',
                'Name': 'nunique'
            })
            grouped = grouped.rename(columns={'Name': 'Authors'})
            grouped['Cit_Mean'] = (grouped['Citations'] / grouped['Publications']).round(2)
            display_cols = ['Organization', 'Country', 'Authors', 'Publications', 'Citations', 'Cit_Mean']
        
        grouped = grouped.sort_values('Publications', ascending=False)
        
        with st.expander("View Results", expanded=False):
            st.write(f"Found {len(grouped)} {display_type.lower()}")
            st.dataframe(grouped[display_cols], hide_index=True)
        
        with st.expander("Visualize", expanded=False):
            plot_data = grouped.copy().sort_values('Cit_Mean', ascending=True).head(30)
            
            if display_type == "Organizations":
                y_col = 'Organization'
            else:
                y_col = 'Name'
            
            fig = px.scatter(
                plot_data,
                x='Cit_Mean',
                y=y_col,
                size='Publications',
                hover_data=['Publications', 'Citations', 'Cit_Mean'],
                size_max=30,
                title=f'{display_type}: Citations Mean vs Publications'
            )
            fig.update_layout(height=max(400, len(plot_data) * 15))
            st.plotly_chart(fig, use_container_width=True)
            del fig
            gc.collect()
    else:
        st.info("No results found")
