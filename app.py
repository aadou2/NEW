from pathlib import Path
import json, html
from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium
import statsmodels.api as sm
from io import BytesIO
from trauma_options import render_trauma_options

NAVY = '#06264A'
TEAL = '#079DB5'
TURQUOISE = '#20B5C6'
LAVENDER = '#8B83CF'
RED = '#C84459'  # Reserved for fatal victims and adverse rate changes
PALETTE = [NAVY, TEAL, LAVENDER]

st.set_page_config(page_title='Cook County Health | Chicago Gun Violence and Healthcare Insights', page_icon='🏥', layout='wide')
ROOT=Path(__file__).parent/'data'
# Public source registry. A reference is not proof of the origin of a local CSV.
SOURCE_REGISTRY = {
    'cpd': ('Chicago Police Department / City of Chicago', 'Violence Reduction — Victims of Homicides and Non-Fatal Shootings', 'https://data.cityofchicago.org/d/gumc-mgzr'),
    'evictions': ('Law Center for Better Housing', 'Chicago Evictions Data Portal', 'https://eviction.lcbh.org/'),
    'atlas': ('Chicago Health Atlas', 'Community health and social indicators', 'https://chicagohealthatlas.org/'),
    'acs': ('U.S. Census Bureau', 'American Community Survey', 'https://www.census.gov/programs-surveys/acs'),
    'community_acs': ('City of Chicago', 'ACS 5 Year Data by Community Area', 'https://catalog.data.gov/dataset/acs-5-year-data-by-community-area'),
    'boundaries': ('City of Chicago', 'Community Area Boundaries', 'https://data.cityofchicago.org/d/cauq-8yn6'),
    'county_gis': ('Cook County GIS', 'Cook County GIS data portal', 'https://hub-cookcountyil.opendata.arcgis.com/'),
    'cms': ('Centers for Medicare & Medicaid Services', 'Hospital Provider Cost Report', 'https://data.cms.gov/provider-compliance/cost-reports/hospital-provider-cost-report'),
    'idph_trauma': ('Illinois Department of Public Health', 'Trauma Program', 'https://dph.illinois.gov/topics-services/emergency-preparedness-response/ems/trauma-program'),
    'idph_report': ('Illinois Department of Public Health', 'Illinois Hospital Report Card', 'https://healthcarereportcard.illinois.gov/'),
    'cch': ('Cook County Health', 'John H. Stroger Jr. Hospital', 'https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/'),
    'cdph_clinics': ('Chicago Department of Public Health', 'Chicago Primary Care Community Health Centers', 'https://data.cityofchicago.org/d/cjg8-dbka'),
    'osrm': ('Project OSRM / OpenStreetMap', 'Routing API — estimated travel durations', 'https://project-osrm.org/docs/v5.22.0/api/'),
    'osm': ('OpenStreetMap contributors', 'OpenStreetMap basemap', 'https://www.openstreetmap.org/copyright'),
}
# Expected references, not asserted origins of the uploaded project's CSV files.
FIELD_SOURCES = {
    'shot_total': ['cpd'], 'fatal': ['cpd'], 'nonfatal': ['cpd'],
    'population': ['acs','community_acs','atlas'],
    'shot_rate_per_10k': ['cpd','acs','community_acs','atlas'],
    'pct_poverty': ['atlas','acs','community_acs'],
    'pct_unemployed': ['atlas','acs','community_acs'],
    'per_capita_income': ['atlas','acs','community_acs'],
    'pct_rent_burdened': ['atlas','acs','community_acs'],
    'eviction_filing_rate': ['evictions'], 'pct_on_medicaid': ['atlas','acs','community_acs'],
    'hospital_locations': ['idph_trauma','idph_report','cch'],
    'hospital_financials': ['cms','idph_report'],
    'profile_clinics': ['atlas','idph_report','cdph_clinics'],
    'profile_nearest_center': ['idph_trauma','osrm'], 'profile_trauma_drive_min': ['osrm'], 'hospital_coverage': [],
}
# Optional metadata, supplied after checking the original exports and documenting the join.
# An accepted entry records provenance; it is not automatic row-level verification.
def read_provenance():
    path = ROOT / 'source_provenance.json'
    if not path.exists(): return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}

PROVENANCE = read_provenance()
def documented(field):
    item = PROVENANCE.get(field, {})
    if not isinstance(item, dict): return False
    ids = item.get('source_ids', [])
    return (isinstance(ids, list) and bool(ids)
            and all(isinstance(x,str) and x in FIELD_SOURCES.get(field, []) for x in ids)
            and all(str(item.get(k, '')).strip() for k in ('reference_period','original_file','method')))

def provenance_status(field):
    return 'Source documented — reconciliation not established' if documented(field) else 'Unconfirmed local origin / method'

@st.cache_data
def load():
    d=pd.read_csv(ROOT/'chicago_community_area_year_merged.csv')
    h=pd.read_csv(ROOT/'trauma_hospitals.csv')
    geo=json.loads((ROOT/'dashboard_map.geojson').read_text())
    
    coverage_path = ROOT / 'hospital_coverage.geojson'
    coverage_geo = (
        json.loads(coverage_path.read_text())
        if coverage_path.exists()
        else {"type": "FeatureCollection", "features": []}
    )
    d.columns=d.columns.str.strip()
    for c in ['shot_total','fatal','nonfatal','population','pct_poverty','pct_unemployed','shot_rate_per_10k','per_capita_income','profile_trauma_drive_min','profile_clinics','pct_rent_burdened','eviction_filing_rate']:
        if c in d: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['ca_num','ca_name','year']).copy()
    d['ca_num']=pd.to_numeric(d.ca_num,errors='coerce'); d['year']=pd.to_numeric(d.year,errors='coerce')
    d=d.dropna(subset=['ca_num','year'])
    d=d[d.ca_num.between(1,77) & d.ca_num.mod(1).eq(0) & d.year.mod(1).eq(0)]
    # Exclude conflicting keys instead of silently keeping arbitrary values.
    d=d.drop_duplicates()
    d=d[~d.duplicated(['ca_num','year'],keep=False)].copy()
    for field in ['shot_total','fatal','nonfatal','population']:
        d.loc[d[field].lt(0) | d[field].mod(1).ne(0),field]=np.nan
    for field in ['pct_poverty','pct_unemployed','pct_rent_burdened','pct_on_medicaid']:
        if field in d:
            d[field]=pd.to_numeric(d[field],errors='coerce')
            d.loc[~d[field].between(0,100),field]=np.nan
    bad=d[['shot_total','fatal','nonfatal']].notna().all(axis=1) & d.shot_total.ne(d.fatal+d.nonfatal)
    d.loc[bad,['shot_total','fatal','nonfatal']]=np.nan
    d['shot_rate_per_10k']=np.where(d.population.gt(0),d.shot_total/d.population*10000,np.nan)
    # Preserve existing project estimates; never silently substitute invented values.
    for field in ['profile_trauma_drive_min','profile_clinics']:
        if field not in d: d[field]=np.nan
        d.loc[d[field].lt(0),field]=np.nan
    if 'profile_nearest_center' not in d: d['profile_nearest_center']=np.nan
    for field in ['total_costs','net_patient_revenue','charity_care_cost','uncompensated_care_cost']:
        h[field]=pd.to_numeric(h[field],errors='coerce') if field in h else np.nan
    if not documented('hospital_coverage'):
        coverage_geo={'type':'FeatureCollection','features':[]}
    # Historical eviction reference from the local dataset, never assigned to another year.
    d['eviction_reference']='Unavailable'
    if 'eviction_filing_rate' in d:
        for _, group in d.groupby('ca_num'):
            valid=group[group.eviction_filing_rate.notna()].sort_values('year')
            for idx, record in group.iterrows():
                prior=valid[valid.year.le(record.year)]
                if not prior.empty:
                    last=prior.iloc[-1]
                    d.at[idx,'eviction_reference']=f"{last.eviction_filing_rate:.2f} (local {int(last.year)} record; source period unconfirmed)"
    d['ca_num']=d.ca_num.astype(int);d['year']=d.year.astype(int)
    for f in geo['features']: f['properties']['ca_num']=int(f['properties']['ca_num'])
    return d,h,geo,coverage_geo
D,H,G,COVERAGE=load()
# Optional official county boundary overlay. The shooting dataset remains Chicago-only.
# Source: Cook County GIS ArcGIS service, Simplified Cook County Boundary (layer 199).
@st.cache_data(ttl=86400, show_spinner=False)
def load_cook_county_boundary():
    local_boundary = ROOT / 'cook_county_boundary.geojson'
    try:
        if local_boundary.is_file():
            geo = json.loads(local_boundary.read_text(encoding='utf-8'))
        else:
            from urllib.request import Request, urlopen
            from urllib.parse import urlencode
            service = ('https://services.arcgis.com/F7DSX1DSNSiWmOqh/ArcGIS/rest/services/'
                       'Simplified_Cook_County_Boundary/FeatureServer/199/query')
            query = urlencode({'where': '1=1', 'outFields': '*',
                               'returnGeometry': 'true', 'outSR': '4326', 'f': 'geojson'})
            request = Request(service + '?' + query, headers={'User-Agent': 'CookCountyHealthDashboard/1.0'})
            with urlopen(request, timeout=10) as response:
                geo = json.load(response)
        if (geo.get('type') == 'FeatureCollection' and geo.get('features')
                and all(feature.get('geometry') for feature in geo['features'])):
            return geo, None
        return None, 'The county GIS service did not provide a valid boundary.'
    except Exception as exc:
        return None, f'Could not load county boundary: {exc}'

st.markdown("""<style>
/* Cook County Health-inspired theme: respect Streamlit light/dark surface variables. */
:root { --cch-navy:#06264A; --cch-teal:#079DB5; --cch-red:#C84459; }
.block-container { padding-top:1.45rem; max-width:1450px; padding-bottom:3rem; }
.stApp { background-color:#FAFCFE; }
[data-testid='stHeader'] { background:rgba(250,252,254,.92); }
h1 { color:#06264A!important; font-weight:760!important; }
h2,h3 { color:#123C64!important; }
[data-testid='stVerticalBlock'] { row-gap:.65rem; }
[data-testid='stAlert'] { border-radius:10px; }
[data-testid='stExpander'] { border-radius:10px; border-color:#DDE7EF; }
[data-testid='stDataFrame'] { border:1px solid #E3EAF1; }
[data-testid='stMetric'] { box-shadow:0 2px 10px rgba(6,38,74,.035); }
h1,h2,h3 { letter-spacing:-.025em; }
[data-testid="stSidebar"] {
 background:linear-gradient(170deg,#06264A 0%,#09365C 73%,#0B546C 100%)!important;
 border-right:3px solid #20B5C6;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label { color:#EDF9FC!important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#C7E8EF!important; }
[data-testid="stSidebar"] [data-baseweb="select"] { background:var(--secondary-background-color); border-radius:8px; }
/* Sidebar report panel: fixed contrast independent of Streamlit theme. */
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary {
 background:#123C64!important; color:#FFFFFF!important; border-radius:8px;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary [data-testid="stMarkdownContainer"] {
 color:#FFFFFF!important; -webkit-text-fill-color:#FFFFFF!important; opacity:1!important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
 background:#FFFFFF!important; color:#06264A!important; border:1px solid #8DB8CE!important;
 font-weight:650!important; opacity:1!important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button *,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button * {
 color:#06264A!important; -webkit-text-fill-color:#06264A!important; opacity:1!important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {background:#DDF2F6!important;}
[data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:focus-visible,
[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus-visible {outline:3px solid #79EAF2!important; outline-offset:2px;}
[data-testid="stSidebar"] [data-testid="stButton"] button:disabled,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:disabled {background:#CBD5E1!important; opacity:.65!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#D5EAF4!important;}

.cch-brand-wrap { border-bottom:2px solid rgba(32,181,198,.7); padding:3px 0 15px; margin-bottom:16px; }
.cch-brand-wrap img { width:100%; max-width:240px; height:auto; border-radius:9px; display:block; }
.cch-brand-wrap .brand-note { font-size:12px; color:#C7E8EF; margin-top:8px; }
[data-testid="stMetric"] {
 border:1px solid rgba(7,157,181,.32); border-left:4px solid #079DB5;
 border-radius:13px; padding:15px 17px;
 background:var(--secondary-background-color);
}
[data-testid="stMetricLabel"] { font-size:.95rem; font-weight:600!important; }
[data-testid="stMetricValue"] { font-size:2.05rem; font-weight:750!important; }
/* Accessible navigation tabs: explicit colors in light AND dark modes.
   Streamlit renders BaseWeb tabs with nested Markdown, which may inherit
   the page theme unless both the tab and its descendants are targeted. */
[data-testid="stTabs"] [role="tablist"],
[data-testid="stTabs"] [data-baseweb="tab-list"] {
 display:flex!important; gap:7px!important; padding:8px!important;
 border-radius:14px!important; background:#06264A!important;
 border:1px solid #1B4C73!important; overflow-x:auto!important;
 scrollbar-width:thin;
}
[data-testid="stTabs"] button[role="tab"],
[data-testid="stTabs"] [data-baseweb="tab"] {
 flex:1 0 auto!important; padding:11px 14px!important; min-height:46px!important;
 border-radius:9px!important; border:1px solid #2D587A!important;
 background:#123C64!important; background-color:#123C64!important;
 color:#FFFFFF!important; -webkit-text-fill-color:#FFFFFF!important;
 opacity:1!important; font-weight:650!important;
 text-shadow:none!important;
}
/* Override any Streamlit theme rules on Markdown labels inside tabs. */
[data-testid="stTabs"] [role="tab"] *,
[data-testid="stTabs"] [data-baseweb="tab"] *,
[data-testid="stTabs"] [role="tab"] [data-testid="stMarkdownContainer"] *,
[data-testid="stTabs"] [role="tab"] p,
[data-testid="stTabs"] [role="tab"] span,
[data-testid="stTabs"] [role="tab"] div {
 color:#FFFFFF!important; -webkit-text-fill-color:#FFFFFF!important;
 opacity:1!important;
}
[data-testid="stTabs"] [role="tab"]:hover,
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
 background:#176486!important; background-color:#176486!important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
 background:#087F93!important; background-color:#087F93!important;
 border-color:#6CE3EF!important;
}
[data-testid="stTabs"] [role="tab"]:focus-visible {
 outline:3px solid #79EAF2!important; outline-offset:2px!important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none!important; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top:1.1rem!important; }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border-radius:12px; overflow:hidden; }
[data-testid="stMarkdownContainer"] p { line-height:1.6; }
[data-testid="stExpander"] { border-radius:10px; }
[data-testid="stMetric"] { min-height:105px; }
@media(max-width:900px) {
 [data-testid="stTabs"] [role="tab"] { font-size:.82rem!important; padding:9px 11px!important; }
}

/* Cook County Health-inspired research portal: presentation-only overrides. */
.stApp { background: var(--background-color, #F6F9FB); }
[data-testid="stHeader"] { background:var(--background-color, #FFFFFF); }
.block-container { max-width:1560px; padding-top:1.7rem; padding-bottom:3rem; }
[data-testid="stMainBlockContainer"] h1, [data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3 { color:var(--text-color)!important; letter-spacing:-.025em; }
.cch-hero { position:relative; overflow:hidden; padding:30px 34px; border-radius:12px;
 background:linear-gradient(110deg,#06264A 0%,#123C64 75%,#087F93 100%); color:#fff; margin-bottom:8px; }
.cch-hero:after { content:""; position:absolute; right:-55px; top:-95px; width:270px; height:270px;
 border:32px solid rgba(32,181,198,.27); border-radius:50%; pointer-events:none; }
.cch-hero .eyebrow { font-size:12px; letter-spacing:.12em; text-transform:uppercase; font-weight:700; color:#B6F0F3; }
.cch-hero h1 { color:#fff!important; font-size:clamp(1.65rem,3vw,2.65rem); line-height:1.15; margin:12px 0!important; }
.cch-hero p { color:#E6F4FA; max-width:760px; margin:0 0 16px; font-size:1rem; }
.cch-hero .meta { display:flex; flex-wrap:wrap; gap:8px; position:relative; z-index:1; }
.cch-hero .meta span { border:1px solid #6592AC; border-radius:5px; padding:5px 10px; font-size:12px; color:#fff; }
.cch-hero a { color:#C2F5F7!important; font-size:13px; text-decoration:underline; position:relative; z-index:1; }
.cch-brand-wrap a { display:block; background:#fff; padding:8px; border-radius:10px; overflow:hidden; }
.cch-brand-wrap img { display:block; width:100%; max-width:240px; margin:0 auto; height:auto; object-fit:contain; border-radius:6px; padding:0; filter:none; }
.cch-brand-wrap .brand-note { font-size:13px; line-height:1.5; }
[data-testid="stMetric"] { border:1px solid rgba(7,157,181,.35); border-top:3px solid #079DB5; border-left:1px solid rgba(7,157,181,.35); border-radius:9px; padding:16px; box-shadow:0 2px 8px rgba(6,38,74,.06); }
/* Spacious navigation, including Streamlit versions using BaseWeb wrappers. */
[data-testid="stTabs"] [role="tablist"],
[data-testid="stTabs"] [data-baseweb="tab-list"] {
 display:flex!important; align-items:center!important; gap:10px!important;
 padding:10px!important; border-radius:12px!important; overflow-x:auto!important;
}
[data-testid="stTabs"] [role="tab"],
[data-testid="stTabs"] [data-baseweb="tab"] {
 flex:0 0 auto!important; margin:0 4px 0 0!important; padding:12px 17px!important;
 min-height:48px!important; border-radius:9px!important;
 border:1px solid #47708E!important; font-size:.92rem!important; white-space:nowrap!important;
 background:#123C64!important; color:#fff!important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
 background:#087F93!important; border-color:#78DCE5!important;
}
[data-testid="stTabs"] [role="tab"] p { font-size:.92rem!important; white-space:nowrap!important; margin:0!important; }
[data-testid="stTabs"] [role="tab"]:hover { background:#176486!important; }
[data-testid="stTabs"] [role="tab"]:focus-visible { outline:2px solid #79EAF2!important; outline-offset:1px!important; }

[data-testid="stDataFrame"] { border-radius:8px; }
[data-testid="stPlotlyChart"] { border:1px solid rgba(128,128,128,.18); border-radius:10px; padding:6px; }
.cch-footer { border-top:2px solid #079DB5; padding:16px 0; color:var(--text-color); font-size:13px; }
.cch-footer a { color:inherit; text-decoration:underline; }
@media(max-width:700px) { .cch-hero { padding:22px 18px; } .cch-hero:after { opacity:.4; } }

</style>""",unsafe_allow_html=True)

def chart_style(fig, height=None):
    fig.update_layout(font=dict(family='Arial',size=13),
        margin=dict(l=10,r=15,t=25,b=20),
        legend_title_text='',
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True,gridcolor='rgba(128,128,128,.17)',zeroline=False)
    fig.update_yaxes(showgrid=False,zeroline=False)
    if height: fig.update_layout(height=height)
    return fig


def fmt(x,dec=0):
    return 'N/A' if pd.isna(x) else f'{x:,.{dec}f}'
def get(row,col): return row.get(col,np.nan)
def pct(x): return f'{fmt(x,1)}%' if pd.notna(x) else 'N/A'
def metric(row,col): return fmt(get(row,col),1 if col in ['shot_rate_per_10k','profile_trauma_drive_min'] else 0)

# Preserve the actual website logo asset without redrawing or recoloring it.
# An approved original image can also be placed in data/cch_official_logo.png
# (or .svg, .webp, .jpg) for a deterministic offline deployment.
@st.cache_data(ttl=86400, show_spinner=False)
def official_logo():
    import base64
    from html.parser import HTMLParser
    from urllib.request import Request, urlopen
    from urllib.parse import urljoin, urlparse
    for suffix, mime in [('png','image/png'), ('svg','image/svg+xml'), ('webp','image/webp'), ('jpg','image/jpeg')]:
        path = ROOT / ('cch_official_logo.' + suffix)
        if path.is_file():
            return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
    page = 'https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/'
    class LogoParser(HTMLParser):
        def __init__(self):
            super().__init__(); self.candidates = []; self.header_depth = 0
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == 'header': self.header_depth += 1
            if tag != 'img': return
            src = a.get('data-src') or a.get('src') or ''
            desc = ' '.join([src, a.get('alt') or '', a.get('class') or '', a.get('id') or '']).lower()
            if not src or 'logo' not in desc or any(x in desc for x in ('countycare','expresscare','express-care','seal.png','footer','partner')): return
            url = urljoin(page, src)
            if urlparse(url).hostname not in ('cookcountyhealth.org','www.cookcountyhealth.org'): return
            score = 10*self.header_depth + 5*('custom-logo' in desc) + 4*('cook county health' in desc) + 2*('cch' in desc)
            if score: self.candidates.append((score,url))
        def handle_endtag(self, tag):
            if tag == 'header': self.header_depth = max(0,self.header_depth-1)
    try:
        with urlopen(Request(page,headers={'User-Agent':'Mozilla/5.0'}),timeout=8) as response:
            parser = LogoParser(); parser.feed(response.read().decode('utf-8',errors='replace'))
        if parser.candidates:
            return sorted(parser.candidates,key=lambda x:x[0],reverse=True)[0][1]
    except Exception:
        pass
    return None

# Prefer an approved local/official asset; retain the previous working brand artwork as fallback.
CCH_PREVIOUS_LOGO = 'https://pbs.twimg.com/profile_images/1842254877193596928/7cNH2u0x.jpg'
CCH_BRAND_IMAGE = official_logo() or CCH_PREVIOUS_LOGO
if CCH_BRAND_IMAGE:
    brand_style = "background:transparent;padding:0" if CCH_BRAND_IMAGE == CCH_PREVIOUS_LOGO else "background:#fff;padding:8px"
    st.sidebar.markdown(
        f'<div class="cch-brand-wrap"><a style="{brand_style}" href="https://cookcountyhealth.org/" target="_blank" rel="noopener noreferrer"><img src="{html.escape(CCH_BRAND_IMAGE, quote=True)}" alt="Cook County Health logo" /></a>'
        '</div>',
        unsafe_allow_html=True)
else:
    st.sidebar.markdown('**[Cook County Health](https://cookcountyhealth.org/)**')
year=st.sidebar.selectbox('Reporting year',sorted(D.year.unique(),reverse=True))
years_available=sorted(D.year.unique().tolist())
range_start,range_end=years_available[0],years_available[-1]
D_window=D[D.year.between(range_start,range_end)].copy()

current=D[D.year==year].copy()
areas=sorted(current.ca_name.unique())
if 'selected_area' not in st.session_state: st.session_state.selected_area='All Chicago Community Areas'
if st.session_state.selected_area not in ['All Chicago Community Areas']+areas: st.session_state.selected_area='All Chicago Community Areas'
area=st.sidebar.selectbox('Select a community area',['All Chicago Community Areas']+areas,key='selected_area')
st.markdown(f"""
<section class="cch-hero">
  <div class="eyebrow">Community health research | Chicago, Illinois</div>
  <h1>Chicago Gun Violence and Healthcare Insights</h1>
  <p>Explore trends in gun violence, socioeconomic conditions, and access to trauma care across Chicago.</p>
  <p style="margin-top:15px;margin-bottom:0"><a href="https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/" target="_blank" rel="noopener noreferrer">Cook County Health — Stroger Hospital</a></p>
</section>
""", unsafe_allow_html=True)
t1,t2,t3,t4,t6,t5=st.tabs(['🗺️ Overview & Map','📊 Socioeconomic Factors','🏥 Trauma Center Access','🏘️ Community Area Comparison','💵 Economic Impact','📚 Glossary & Resources'])

row=current[current.ca_name==area].iloc[0] if area!='All Chicago Community Areas' else None
selected=current if row is None else current[current.ca_name==area]
shot=selected.shot_total.sum(min_count=len(selected));fatal=selected.fatal.sum(min_count=len(selected));nonfatal=selected.nonfatal.sum(min_count=len(selected));pop=selected.population.sum(min_count=len(selected))
rate=shot/pop*10000 if pop>0 else np.nan

with t1:
    st.subheader(f'{area} · {year}')
    a,b,c,d=st.columns(4)
    a.metric('Shooting victims',fmt(shot));b.metric('Fatal shooting victims',fmt(fatal));c.metric('Nonfatal shooting victims',fmt(nonfatal));d.metric('Victims per 10,000 residents',fmt(rate,1))
    st.markdown('#### Key findings')
    valid_rate=current.dropna(subset=['shot_rate_per_10k'])
    if not valid_rate.empty:
        highest=valid_rate.loc[valid_rate.shot_rate_per_10k.idxmax()]
        insight1=(f"**Highest recorded victim rate:** {highest.ca_name} "
                  f"({highest.shot_rate_per_10k:.1f} per 10,000 residents in {year}).")
    else:
        insight1='**Highest recorded victim rate:** No data available.'
    prior=D[D.year.eq(year-1) & (year < date.today().year)][['ca_num','shot_total','population']].copy()
    prior['prior_rate']=np.where(prior.population.gt(0),prior.shot_total/prior.population*10000,np.nan)
    changes=current[['ca_num','ca_name','shot_rate_per_10k']].merge(
        prior[['ca_num','prior_rate']],on='ca_num',how='inner').dropna()
    if not changes.empty:
        changes['rate_drop']=changes['prior_rate']-changes['shot_rate_per_10k']
        best=changes.loc[changes.rate_drop.idxmax()]
        insight2=(f"**Largest year-over-year rate decrease:** {best.ca_name} "
                  f"(a decrease of {best.rate_drop:.1f} victims per 10,000 residents compared with {year-1}).") if best.rate_drop>0 else                  f'**Year-over-year comparison:** No community area shows a lower rate than {year-1} in the comparable data.'
    else:
        insight2=f'**Year-over-year comparison:** Comparable {year-1} rates are unavailable.'
    valid_drive=current.dropna(subset=['profile_trauma_drive_min'])
    if not valid_drive.empty:
        longest=valid_drive.loc[valid_drive.profile_trauma_drive_min.idxmax()]
        insight3=(f"**Longest estimated driving time to a trauma center:** {longest.ca_name} "
                  f"({longest.profile_trauma_drive_min:.1f} minutes; excludes ambulance response time).")
    else:
        insight3='**Estimated driving time to a trauma center:** No data available.'
    for insight in (insight1,insight2,insight3):
        st.markdown('- '+insight)
    left,right=st.container(),st.container()
    with left:
        st.subheader('Where are shooting victims concentrated?')
        choices={'Victims per 10,000 residents':'shot_rate_per_10k','Total shooting victims':'shot_total','Poverty (%)':'pct_poverty','Unemployment (%)':'pct_unemployed'}
        choice=st.radio('Map shading',list(choices),horizontal=True)
        field=choices[choice]
        values=current.set_index('ca_num')[field].to_dict()
        v=pd.Series(list(values.values())).dropna()
        vmin=float(v.min()) if len(v) else 0; vmax=float(v.max()) if len(v) else 1
        import branca.colormap as bcm
        cmap=bcm.LinearColormap(['#fff7bc','#fec44f','#f03b20','#800026'],vmin=vmin,vmax=max(vmax,vmin+.01),caption=choice)
        show_trauma_on_main=st.checkbox('Show listed trauma hospitals on this map',value=False,key='main_trauma_markers')
        mp=folium.Map(location=[41.84,-87.68],zoom_start=10,tiles='OpenStreetMap',control_scale=True)
        lookup=current.set_index('ca_num')
        def style(feature):
            cid=int(feature['properties']['ca_num']);val=values.get(cid,np.nan)
            return {'fillColor':cmap(val) if pd.notna(val) else '#cbd5e1','color':'#202c3b' if row is not None and int(row.ca_num)==cid else '#657181','weight':3 if row is not None and int(row.ca_num)==cid else .8,'fillOpacity':.78}
        for feature in G['features']:
            cid=int(feature['properties']['ca_num'])
            if cid not in lookup.index: continue
            r=lookup.loc[cid]
            fields=[('Shooting victims',fmt(r.shot_total)),('Fatal shooting victims',fmt(r.fatal)),('Nonfatal shooting victims',fmt(r.nonfatal)),('Victims per 10,000 residents',fmt(r.shot_rate_per_10k,1)),('Poverty (%)',pct(r.pct_poverty)),('Unemployment (%)',pct(r.pct_unemployed)),('Income per person', '$'+fmt(r.per_capita_income)),('Eviction filing rate',get(r,'eviction_reference')),('Estimated driving time to a trauma center (minutes)',fmt(get(r,'profile_trauma_drive_min'),1)),('Estimated number of clinics',fmt(get(r,'profile_clinics'))),('Nearest listed trauma center',str(get(r,'profile_nearest_center')) if pd.notna(get(r,'profile_nearest_center')) else 'Data unavailable'),('Medicaid coverage (%)',pct(get(r,'pct_on_medicaid')))]
            details=''.join(f'<tr><td style="padding:3px 8px">{html.escape(k)}</td><td><b>{html.escape(str(val))}</b></td></tr>' for k,val in fields)
            popup=folium.Popup(f'<div style="font:13px Arial;width:315px"><h3>{html.escape(str(r.ca_name))} · {year}</h3><table>{details}</table><p style="font-size:11px">Some indicator definitions and reporting periods remain unverified. Estimated driving time does not measure ambulance response time.</p></div>',max_width=380)
            folium.GeoJson(feature,style_function=style,highlight_function=lambda f:{'weight':3,'color':'#111827','fillOpacity':.88},tooltip=f'{r.ca_name}: {choice} {fmt(get(r,field),1)}',popup=popup).add_to(mp)
        # Add Stroger Hospital to the Overview map
        folium.Marker(
            location=[41.8564, -87.6663],
            tooltip="John H. Stroger Jr. Hospital",
            popup="Cook County Health - Stroger Hospital",
            icon=folium.Icon(color="darkblue", icon="plus-sign")
        ).add_to(mp)
        if show_trauma_on_main and {'hospital','lat','lon'}.issubset(H.columns):
            trauma_overlay=folium.FeatureGroup(name='Listed hospitals',show=True)
            for _, hosp in H.iterrows():
                lat=pd.to_numeric(hosp.get('lat'),errors='coerce')
                lon=pd.to_numeric(hosp.get('lon'),errors='coerce')
                if pd.isna(lat) or pd.isna(lon):
                    continue
                name=str(hosp.get('hospital','Listed hospital'))
                is_stroger=('stroger' in name.lower() or 'cook county hospital' in name.lower())
                folium.CircleMarker(
                    location=[float(lat),float(lon)],radius=7 if is_stroger else 5,
                    color=NAVY if is_stroger else TEAL,fill=True,fill_opacity=.95,
                    tooltip=name,
                    popup=folium.Popup(html.escape(name)+'<br>Location from hospital dataset',max_width=280)
                ).add_to(trauma_overlay)
            trauma_overlay.add_to(mp)
        # Custom vertical map legend, positioned INSIDE the map on its right edge.
        # Do not add cmap to the map: its default horizontal legend sits on top.
        def legend_number(x):
            return f'{x:,.0f}' if abs(x) >= 100 else f'{x:,.1f}'

        legend_ticks = [vmax - (vmax - vmin) * fraction for fraction in (0, .25, .5, .75, 1)]
        legend_labels = ''.join(
            f'<span>{html.escape(legend_number(number))}</span>'
            for number in legend_ticks
        )
        vertical_legend = f"""
        <style>
          #cch-map-legend {{
            position:fixed; z-index:9999; top:50%; right:14px;
            transform:translateY(-50%); box-sizing:border-box;
            background:transparent; border:none; border-radius:0; padding:0;
            box-shadow:none; font:12px Arial,sans-serif; color:#06264A;
          }}
          #cch-map-legend .legend-title {{
            font-weight:700; font-size:11px; line-height:1.25;
            margin-bottom:8px; text-shadow:0 1px 2px rgba(255,255,255,.9);
          }}
          #cch-map-legend .legend-body {{display:flex;gap:6px;height:320px;align-items:stretch;}}
          #cch-map-legend .legend-gradient {{
            width:12px; min-width:12px; height:100%; border-radius:3px;
            border:1px solid rgba(6,38,74,.18);
            background:linear-gradient(to bottom,#800026 0%,#f03b20 34%,#fec44f 68%,#fff7bc 100%);
          }}
          #cch-map-legend .legend-ticks {{
            display:flex;flex-direction:column;justify-content:space-between;
            text-align:left;white-space:nowrap;font-size:11px;
            text-shadow:0 1px 2px rgba(255,255,255,.92);
          }}
          @media(max-width:560px) {{
            #cch-map-legend {{right:8px;}}
            #cch-map-legend .legend-body {{height:240px;}}
            #cch-map-legend .legend-title {{font-size:10px;}}
            #cch-map-legend .legend-gradient {{width:10px; min-width:10px;}}
          }}
        </style>
        <div id="cch-map-legend" aria-label="Vertical map legend">
          <div class="legend-title">{html.escape(choice)}</div>
          <div class="legend-body">
            <div class="legend-gradient"></div>
            <div class="legend-ticks">{legend_labels}</div>
          </div>
        </div>
        """
        mp.get_root().html.add_child(folium.Element(vertical_legend))
        st_folium(mp,height=520,use_container_width=True,key=f'community_map_{year}_{choice}',returned_objects=['last_active_drawing'])
        st.caption('Click a community area on the map to view its details. Use the sidebar to select a reporting year or focus area.')
    with right:
        st.subheader('10 community areas with the most shooting victims')
        top=current.nlargest(10,'shot_total').sort_values('shot_total')
        fig=px.bar(top,x='shot_total',y='ca_name',orientation='h',labels={'shot_total':'Shooting victims','ca_name':''},height=540)
        fig.update_traces(marker_color=TEAL);fig.update_layout(autosize=True,margin=dict(l=25,r=35,t=20,b=65),showlegend=False);fig.update_yaxes(automargin=True);fig.update_xaxes(automargin=True)
        st.plotly_chart(fig,use_container_width=True)
    if row is not None:
        st.subheader(f'{area}: Socioeconomic and Trauma Care Indicators')
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[('Poverty', 'pct_poverty'),('Unemployment','pct_unemployed'),('Estimated driving time to a trauma center (minutes)','profile_trauma_drive_min'),('Estimated number of clinics','profile_clinics')]):
            col.metric(label,pct(get(row,key)) if key in ['pct_poverty','pct_unemployed'] else fmt(get(row,key),0 if key=='profile_clinics' else 1))
    st.subheader('How have shootings changed over time?')
    trend=D_window if row is None else D_window[D_window.ca_name==area]
    trend=trend.groupby('year',as_index=False)[['fatal','nonfatal']].sum()
    fig=px.bar(trend,x='year',y=['fatal','nonfatal'],barmode='stack',color_discrete_map={'fatal':RED,'nonfatal':TEAL},labels={'value':'Victims','variable':'Outcome'},height=440)
    fig.for_each_trace(lambda trace: trace.update(name={'fatal':'Fatal shooting victims','nonfatal':'Nonfatal shooting victims'}.get(trace.name,trace.name)))
    fig.update_layout(autosize=True,xaxis=dict(dtick=1,tickangle=-45,automargin=True),margin=dict(l=25,r=25,t=55,b=85),legend=dict(orientation='h',y=1.12,x=0));fig.update_yaxes(automargin=True);st.plotly_chart(fig,use_container_width=True)

with t2:
    st.subheader('How are neighborhood conditions related to gun violence?')
    st.write('Compare shooting victim rates with measures such as poverty, unemployment, income, housing costs, and evictions. Each point represents one Chicago community area.')
    indicators={'Poverty (%)':'pct_poverty','Unemployment (%)':'pct_unemployed','Income per person ($)':'per_capita_income','Rent burden (%)':'pct_rent_burdened','Eviction filing rate':'eviction_filing_rate'}
    factor=st.selectbox('Select a socioeconomic factor',list(indicators));col=indicators[factor]
    # Short, plain-English definitions; calculated rates come from the source data.
    # We do not recompute them without the original numerator, denominator, or source metadata.
    factor_explanations = {
        'Poverty (%)': (
            'Percentage of people living below the poverty threshold.',
            '(People below the poverty threshold ÷ people covered by the poverty measure) × 100.'
        ),
        'Unemployment (%)': (
            'Percentage of the labor force that is unemployed and looking for work.',
            '(Unemployed people ÷ total labor force) × 100.'
        ),
        'Income per person ($)': (
            'Average income per person, not per household.',
            'Total income of residents ÷ total population.'
        ),
        'Rent burden (%)': (
            'Percentage of renter households spending at least 30% of income on housing under the standard definition.',
            '(Rent-burdened renter households ÷ renter households assessed) × 100.'
        ),
        'Eviction filing rate': (
            'How frequently eviction cases are filed against renter households.',
            'Eviction filings ÷ renter households; the dataset source determines whether the result is per 100, per 1,000, or another unit.'
        ),
    }
    meaning, calculation = factor_explanations[factor]
    st.info(f'**{factor} — Meaning:** {meaning}  \n**How it is calculated:** {calculation}')
    st.caption('These are standard definitions. The exact definitions and reporting periods of some dataset indicators remain unverified.')
    with st.expander('See definitions and calculations for all five factors'):
        for factor_name, (short_meaning, short_formula) in factor_explanations.items():
            st.markdown(f'**{factor_name}:** {short_meaning}  \n*Calculation:* {short_formula}')
    valid=current.dropna(subset=[col,'shot_rate_per_10k'])
    fig=px.scatter(valid,x=col,y='shot_rate_per_10k',size='shot_total',hover_name='ca_name',color='shot_rate_per_10k',color_continuous_scale=[(0,'#DDF2F6'),(.5,TEAL),(1,RED)],labels={col:factor,'shot_rate_per_10k':'Shooting victims per 10,000 residents'},height=410)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Each point represents a community area. An association does not establish causation.')
    st.subheader('Statistical model (Poisson regression)')
    st.write('This analysis looks for patterns between poverty, unemployment, and shooting victim rates across Chicago. It accounts for differences in community population and reporting year. A relationship in the data does **not** mean one condition causes shootings.')
    predictors = ['pct_poverty', 'pct_unemployed']
    model_data = D[['ca_name', 'year', 'shot_total', 'population'] + predictors].dropna().copy()
    model_data = model_data[(model_data.population > 0) & (model_data.shot_total >= 0)]
    if len(model_data) < 10:
        st.info('There are insufficient complete records to fit the statistical model.')
    else:
        try:
            X = pd.concat([model_data[predictors].astype(float),
                pd.get_dummies(model_data.year.astype(str), prefix='year', drop_first=True, dtype=float)], axis=1)
            X = sm.add_constant(X)
            model = sm.GLM(model_data.shot_total.astype(float), X,
                family=sm.families.Poisson(),
                offset=np.log(model_data.population.astype(float))).fit(cov_type='HC1')
            dispersion = model.pearson_chi2 / model.df_resid if model.df_resid > 0 else np.nan
            c1, c2 = st.columns(2)
            c1.metric('Records analyzed', f'{len(model_data):,}', help='Records with complete data for a community area and reporting year.')
            c2.metric('Factors examined', '2', help='Poverty and unemployment.')
            results = []
            for key, label in [('pct_poverty', 'Poverty'), ('pct_unemployed', 'Unemployment')]:
                ratio = float(np.exp(model.params[key]))
                change = (ratio - 1) * 100
                direction = 'higher' if change > 0 else 'lower' if change < 0 else 'unchanged'
                results.append({'Factor': label,
                    'Estimated rate difference per percentage-point increase': f'{abs(change):.1f}% {direction}'})
            st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
            st.caption('These are model-based associations, not predictions or evidence that changing poverty or unemployment will change shootings. Other important factors may be missing.')
            st.subheader('How closely does the model match recorded shootings?')
            st.write('Each dot represents a Chicago community area. The horizontal axis shows the number of shooting victims estimated by the model, and the vertical axis shows the actual recorded number. Dots closer to the dashed line indicate a closer match.')
            # In-sample fitted values for the selected reporting year, not future predictions.
            model_data['model_estimate'] = model.predict(
                X, offset=np.log(model_data.population.astype(float)))
            comparison = model_data[model_data.year.eq(year)].copy()
            if not comparison.empty:
                fig_model = px.scatter(
                    comparison, x='model_estimate', y='shot_total',
                    hover_name='ca_name',
                    labels={'model_estimate': 'Model-estimated shooting victims',
                            'shot_total': 'Recorded shooting victims'},
                    height=410)
                fig_model.update_traces(marker=dict(color=TEAL, size=10, opacity=.8))
                limit = float(max(comparison['model_estimate'].max(),
                                  comparison['shot_total'].max(), 1)) * 1.05
                fig_model.add_shape(type='line', x0=0, y0=0, x1=limit, y1=limit,
                                    line=dict(color=RED, dash='dash', width=2))
                fig_model.update_xaxes(range=[0, limit])
                fig_model.update_yaxes(range=[0, limit])
                st.plotly_chart(fig_model, use_container_width=True)
                st.caption(f'Selected year: {year}. These are estimates for records used to fit the model, not forecasts or proof of cause and effect. The dashed line represents a perfect match.')
            else:
                st.info(f'No complete model records are available for {year}.')
            st.subheader('Explore hypothetical neighborhood conditions')
            st.write('Choose a community and adjust poverty and unemployment to explore how the model estimates shooting rates relative to the original conditions.')
            scenario_rows = model_data[model_data.year.eq(year)].sort_values('ca_name')
            if not scenario_rows.empty:
                scenario_areas = scenario_rows.ca_name.unique().tolist()
                chosen = st.selectbox('Select a community area for the scenario', scenario_areas,
                    index=scenario_areas.index(area) if area in scenario_areas else 0,
                    key=f'scenario_area_{year}')
                base = scenario_rows[scenario_rows.ca_name.eq(chosen)].iloc[0]
                bp, bu = float(base.pct_poverty), float(base.pct_unemployed)
                st.caption(f'Recorded conditions in {chosen}: poverty, {bp:.1f}%; unemployment, {bu:.1f}%.')
                sc1, sc2 = st.columns(2)
                p = sc1.slider('Poverty (%)', 0.0, 100.0, bp, 0.1, key=f'scenario_p_{year}_{chosen}')
                u = sc2.slider('Unemployment (%)', 0.0, 100.0, bu, 0.1, key=f'scenario_u_{year}_{chosen}')
                ratio = float(np.exp(model.params['pct_poverty']*(p-bp) + model.params['pct_unemployed']*(u-bu)))
                change = (ratio-1)*100
                st.metric('Estimated rate change', f'{change:+.1f}%')
                st.caption('Compared with the original model-estimated rate for this community. Population, year and other model inputs are held fixed. Extreme slider values may be outside the observed data. This is not a forecast or evidence of a causal effect.')
            else:
                st.info('No complete community records are available for this scenario in the selected year.')
            with st.expander('How was this calculated? (technical details)'):
                st.write('Exploratory Poisson regression for victim counts, adjusted for population size and reporting year, with robust standard errors. A one-percentage-point increase means, for example, poverty changing from 20% to 21%. Repeated observations within community areas and greater-than-expected variation may limit reliability. This analysis has not been independently validated.')
                if pd.notna(dispersion) and dispersion > 1.5:
                    st.caption('The observed counts vary more than the model expects, which may reduce the reliability of its estimates.')
        except Exception as exc:
            st.info(f'The statistical analysis is unavailable for the current dataset.')

with t3:
    st.subheader('Trauma center access')
    st.write('Explore trauma center locations and straight-line distances from Chicago community areas.')
    if row is not None:
        a,b,c=st.columns(3)
        a.metric('Nearest listed center',str(get(row,'profile_nearest_center')) if pd.notna(get(row,'profile_nearest_center')) else 'Data unavailable')
        b.metric('Estimated driving time to a trauma center (minutes)',fmt(get(row,'profile_trauma_drive_min'),1))
        c.metric('Estimated number of clinics',fmt(get(row,'profile_clinics')))
    render_trauma_options(current, G, H, row, year)
    st.divider()
    st.subheader("Cook County Health | John H. Stroger Jr. Hospital")
    st.caption("Featured hospital · Chicago, Illinois")
    k1, k2, k3 = st.columns(3)
    k1.metric("Trauma designation", "Adult Level I")
    k2.metric("Healthcare system", "Cook County Health")
    k3.metric("Location", "Chicago, IL")
    st.markdown(
        "**John H. Stroger Jr. Hospital of Cook County** is a public teaching "
        "hospital providing emergency and trauma services. "
        "**Address:** 1969 W. Ogden Avenue, Chicago, IL 60612. "
        "[Official Stroger Hospital page](https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/)"
    )
    st.subheader('Community area trauma care indicators')
    with st.expander('Explore community area trauma care indicators'):
        st.dataframe(current[['ca_name','shot_rate_per_10k','profile_nearest_center','profile_trauma_drive_min','profile_clinics']].sort_values('shot_rate_per_10k',ascending=False).rename(columns={'ca_name':'Community area','shot_rate_per_10k':'Victims per 10,000 residents','profile_nearest_center':'Nearest listed center','profile_trauma_drive_min':'Estimated drive (minutes)','profile_clinics':'Estimated clinics'}),hide_index=True,use_container_width=True)

with t4:
    st.subheader('Community area comparison')
    st.write('Compare shooting victim rates and neighborhood conditions across Chicago. Choose communities below, or leave the selection empty to see all areas.')
    measures = {
        'Shooting victims per 10,000 residents': 'shot_rate_per_10k',
        'Total shooting victims': 'shot_total',
        'Poverty (%)': 'pct_poverty',
        'Unemployment (%)': 'pct_unemployed',
        'Income per person ($)': 'per_capita_income',
    }
    filter_col, rank_col = st.columns([1.6, 1])
    with filter_col:
        neighborhoods = st.multiselect('Select community areas (leave blank to include all)', areas,
            default=[area] if area != 'All Chicago Community Areas' else [], key='compare_filter')
    with rank_col:
        measure_label = st.selectbox('Compare by', list(measures), key='compare_measure')
    measure = measures[measure_label]
    subset = current[current.ca_name.isin(neighborhoods)].copy() if neighborhoods else current.copy()
    subset = subset.sort_values(measure, ascending=False, na_position='last')
    if subset.empty:
        st.info('No community-area records are available for this selection.')
    else:
        chart_col, trend_col = st.columns(2, gap='large')
        with chart_col:
            st.markdown('**Community comparison**')
            top = subset.dropna(subset=[measure]).head(10).sort_values(measure)
            if top.empty:
                st.info('No values are available for this indicator.')
            else:
                bar = px.bar(top, x=measure, y='ca_name', orientation='h',
                    labels={measure: measure_label, 'ca_name': ''}, height=400)
                bar.update_traces(marker_color=TEAL)
                bar.update_layout(showlegend=False, margin=dict(l=0,r=8,t=5,b=0))
                bar.update_xaxes(tickprefix='$' if measure == 'per_capita_income' else '',
                    ticksuffix='%' if measure in ('pct_poverty','pct_unemployed') else '')
                st.plotly_chart(chart_style(bar), use_container_width=True)
        with trend_col:
            st.markdown('**Shooting victim rates over time**')
            history = D_window[D_window.ca_name.isin(subset.ca_name)].copy()
            history = history.dropna(subset=['shot_rate_per_10k'])
            if history.empty:
                st.info('Historical data are unavailable for this selection.')
            else:
                if not neighborhoods:
                    annual = history.dropna(subset=['shot_total','population']).groupby('year', as_index=False)[['shot_total','population']].sum()
                    annual = annual[annual.population.gt(0)].copy()
                    annual['rate'] = annual.shot_total / annual.population * 10000
                    trend = px.line(annual, x='year', y='rate', markers=True,
                        labels={'year':'Year','rate':'Victims per 10,000 residents'}, height=400)
                    st.caption('Citywide trend. Select individual community areas to compare their trends.')
                else:
                    trend = px.line(history, x='year', y='shot_rate_per_10k',
                        color='ca_name', markers=True,
                        labels={'year':'Year','shot_rate_per_10k':'Victims per 10,000 residents','ca_name':'Community area'},
                        height=400)
                    if len(neighborhoods) > 8:
                        st.caption('Select fewer community areas to make the trends easier to distinguish.')
                trend.update_layout(xaxis=dict(dtick=1), hovermode='x unified')
                st.plotly_chart(chart_style(trend), use_container_width=True)
        st.caption('Charts describe observed patterns; they do not establish causes. Socioeconomic data may refer to different source years.')
        st.markdown('**Community-area data**')
        display_cols = {
            'ca_name':'Community area',
            'shot_total':'Shooting victims',
            'shot_rate_per_10k':'Victims per 10,000 residents',
            'pct_poverty':'Poverty (%)',
            'pct_unemployed':'Unemployment (%)',
            'per_capita_income':'Income per person ($)',
        }
        full = subset[[c for c in display_cols if c in subset.columns]].rename(columns=display_cols)
        formats = {'Shooting victims':'{:,.0f}', 'Victims per 10,000 residents':'{:,.1f}',
            'Poverty (%)':'{:,.1f}%', 'Unemployment (%)':'{:,.1f}%',
            'Income per person ($)':'${:,.0f}'}
        st.dataframe(full.style.format({k:v for k,v in formats.items() if k in full}, na_rep='N/A'),
            hide_index=True, use_container_width=True)

with t6:
    st.subheader("Economic impact: hospital costs")
    st.warning("These figures describe hospital-wide financial activity, not the cost of treating firearm injuries. The original financial sources and reporting periods have not been fully verified.")
    finance_labels = {
        'hospital': 'Hospital', 'type': 'Hospital type',
        'fiscal_year_end': 'Fiscal year end',
        'total_costs': 'Total hospital costs ($)',
        'net_patient_revenue': 'Net patient revenue ($)',
        'charity_care_cost': 'Charity care costs ($)',
        'uncompensated_care_cost': 'Uncompensated care costs ($)',
    }
    money_fields = ['total_costs','net_patient_revenue','charity_care_cost','uncompensated_care_cost']
    def show_financial_table(frame, include_type=True):
        cols = [c for c in finance_labels if c in frame.columns and (include_type or c != 'type')]
        table = frame[cols].rename(columns=finance_labels)
        formats = {finance_labels[c]: '${:,.0f}' for c in money_fields if c in cols}
        st.dataframe(table.style.format(formats, na_rep='N/A'), hide_index=True, use_container_width=True)

    stroger_data = H[H['hospital'].astype(str).str.contains(r'Stroger|Cook County Hospital', case=False, regex=True, na=False)]
    if not stroger_data.empty:
        st.markdown('**Stroger Hospital: reported hospital-wide figures**')
        show_financial_table(stroger_data, include_type=False)
    else:
        st.info('No financial record for Stroger Hospital is available in the dataset.')

    st.subheader('Hospital costs and revenue')
    st.caption('Hospital-wide financial figures (USD). Reporting periods may differ.')
    sort_label = st.selectbox('Sort hospitals by', ['Hospital (A–Z)'] + [finance_labels[c] for c in money_fields], key='finance_sort')
    sort_field = next((c for c in money_fields if finance_labels[c] == sort_label), 'hospital')
    ordered = H.sort_values(sort_field, ascending=sort_field == 'hospital', na_position='last')
    show_financial_table(ordered)
    chart_label = st.selectbox('Financial measure to compare', [finance_labels[c] for c in money_fields], key='finance_chart')
    chart_field = next(c for c in money_fields if finance_labels[c] == chart_label)
    financial = H.dropna(subset=[chart_field]).sort_values(chart_field)
    if not financial.empty:
        fig = px.bar(financial, x=chart_field, y='hospital', orientation='h',
                     labels={chart_field:chart_label,'hospital':''}, height=max(400,45*len(financial)))
        fig.update_traces(marker_color=[NAVY if 'stroger' in str(n).lower() else TEAL for n in financial.hospital],
                          hovertemplate='%{y}<br>$%{x:,.0f}<extra></extra>')
        fig.update_xaxes(tickprefix='$', tickformat=',.0f')
        st.plotly_chart(chart_style(fig), use_container_width=True)
    else:
        st.info('No numerical values are available for this financial measure.')
    st.caption('Financial figures cover all hospital services, not only firearm injuries.')
    st.markdown('[CMS Hospital Provider Cost Report — data source reference](https://data.cms.gov/provider-compliance/cost-reports/hospital-provider-cost-report)')


# Client-ready PDF report using the same selection and currently loaded dataset.
def create_client_pdf(data, all_history, hospitals, reporting_year, community, start_year, end_year):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    navy=colors.HexColor(NAVY)
    teal=colors.HexColor(TEAL)
    out=BytesIO()
    pdf=SimpleDocTemplate(out,pagesize=(612,792),leftMargin=42,rightMargin=42,
                          topMargin=42,bottomMargin=42)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CCHTitle',parent=styles['Title'],textColor=navy,fontSize=20,leading=24,spaceAfter=12))
    styles.add(ParagraphStyle(name='CCHHeading',parent=styles['Heading2'],textColor=navy,spaceBefore=14,spaceAfter=7))
    styles.add(ParagraphStyle(name='CCHBody',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=7))
    story=[Paragraph('Cook County Health | Community Insights',styles['CCHTitle']),
           Paragraph(f'{html.escape(community)} | Reporting year {reporting_year} | Historical view {start_year}–{end_year}',styles['CCHBody'])]
    result=data if community=='All Chicago Community Areas' else data[data.ca_name.eq(community)]
    patients=result.shot_total.sum(min_count=len(result))
    fatal_n=result.fatal.sum(min_count=len(result))
    pop_sum=result.population.sum()
    patient_rate=patients/pop_sum*10000 if pd.notna(pop_sum) and pop_sum>0 else np.nan
    kpi_data=[['Shooting victims','Fatal shooting victims','Victims per 10,000 residents'],
              [fmt(patients),fmt(fatal_n),fmt(patient_rate,1)]]
    kpi_table=Table(kpi_data,colWidths=[175]*3,rowHeights=[27,35])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#EAF7FA')),
        ('TEXTCOLOR',(0,1),(-1,1),navy),('FONTSIZE',(0,0),(-1,-1),10),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.extend([kpi_table,Spacer(1,12),Paragraph('Selected-year observations',styles['CCHHeading'])])
    valid=data.dropna(subset=['shot_rate_per_10k'])
    if community=='All Chicago Community Areas' and not valid.empty:
        top=valid.sort_values('shot_rate_per_10k',ascending=False).iloc[0]
        story.append(Paragraph(f'Highest observed community rate: {html.escape(str(top.ca_name))} ({top.shot_rate_per_10k:.1f} victims per 10k).',styles['CCHBody']))
    elif community!='All Chicago Community Areas' and not result.empty:
        r=result.iloc[0]
        if pd.notna(r.get('profile_trauma_drive_min',np.nan)):
            story.append(Paragraph(f'Estimated trauma-center driving time: {r.profile_trauma_drive_min:.1f} minutes. Not EMS transport time.',styles['CCHBody']))

    history=all_history[all_history.year.between(start_year,end_year)]
    history=history if community=='All Chicago Community Areas' else history[history.ca_name.eq(community)]
    history=history.dropna(subset=['shot_total','population']).groupby('year',as_index=False)[['shot_total','population']].sum()
    history=history[history.population.gt(0)].copy()
    if not history.empty:
        history['victim_rate']=history.shot_total/history.population*10000
        fig,ax=plt.subplots(figsize=(7.1,2.5))
        ax.plot(history.year,history.victim_rate,marker='o',color=TEAL,linewidth=2)
        ax.set_ylabel('Victims per 10,000 residents'); ax.set_xlabel('Year');ax.set_title('Annual shooting victim rate')
        ax.grid(axis='y',alpha=.25);fig.tight_layout()
        img=BytesIO();fig.savefig(img,format='png',dpi=140,bbox_inches='tight');plt.close(fig);img.seek(0)
        story.extend([Paragraph('Historical trend',styles['CCHHeading']),Image(img,width=505,height=178)])
    if community=='All Chicago Community Areas':
        top=data.dropna(subset=['shot_rate_per_10k']).sort_values('shot_rate_per_10k',ascending=False).head(6)
        if not top.empty:
            fig,ax=plt.subplots(figsize=(7.0,2.8))
            ax.barh(top.ca_name.iloc[::-1],top.shot_rate_per_10k.iloc[::-1],color=TEAL)
            ax.set_xlabel('Victims per 10,000 residents');fig.tight_layout()
            img2=BytesIO();fig.savefig(img2,format='png',dpi=130,bbox_inches='tight');plt.close(fig);img2.seek(0)
            story.extend([Paragraph('Six highest community rates',styles['CCHHeading']),Image(img2,width=505,height=202)])
    story.extend([Spacer(1,9),Paragraph('Data definitions and limitations',styles['CCHHeading']),
        Paragraph('Public references: Chicago Police Department / City of Chicago Violence Reduction victimizations dataset; Law Center for Better Housing Chicago Evictions Data Portal; Chicago Health Atlas; U.S. Census Bureau ACS. Individual measures must be reconciled with their original releases.',styles['CCHBody']),
        Paragraph('Victims are counts of injured or killed people, not distinct incidents. Socioeconomic profile variables may repeat across years. Estimated driving times are not EMS response times. Associations are not causal predictions. Hospital financial figures (not shown here) are hospital-wide, not firearm-specific.',styles['CCHBody']),
        Paragraph('Research prototype | Local input values have not been reconciled with original source exports. Not medical, emergency-response, or validated policy guidance.',styles['CCHBody'])])
    pdf.build(story)
    out.seek(0)
    return out.getvalue()

st.divider()
with t5:
    st.markdown('### Data Sources')
    st.caption('Public datasets and geographic resources relevant to this dashboard. The exact source of each merged indicator is listed under Definitions when documented.')
    for source_id in ('cpd','evictions','atlas','acs','community_acs','boundaries','county_gis','cms','cdph_clinics'):
        agency, title, url = SOURCE_REGISTRY[source_id]
        st.markdown(f'**{agency}** — [{title}]({url})')
    st.markdown('### Publications')
    st.markdown('- [U.S. Census Bureau — Housing Cost Burden](https://www.census.gov/library/stories/2022/12/housing-costs-burden.html)')
    st.markdown('### References')
    for source_id in ('idph_trauma','idph_report','cch','osrm','osm'):
        agency, title, url = SOURCE_REGISTRY[source_id]
        st.markdown(f'**{agency}** — [{title}]({url})')
    st.markdown('- [U.S. Census Bureau — Poverty Measures](https://www.census.gov/topics/income-poverty/guidance/poverty-measures.html)')
    st.markdown('### Definitions')
    indicator_guide = {
        'shot_total': ('Total Shooting Victims', 'Number of recorded fatal and nonfatal shooting victims.'),
        'fatal': ('Fatal Shooting Victims', 'Number of recorded shooting victims who died from their injuries.'),
        'nonfatal': ('Nonfatal Shooting Victims', 'Number of recorded shooting victims who survived their injuries.'),
        'population': ('Community Population', 'Estimated number of residents in a community area.'),
        'shot_rate_per_10k': ('Shooting Victim Rate', 'Number of shooting victims per 10,000 residents.'),
        'pct_poverty': ('Poverty', 'Percentage of people living below the poverty threshold.'),
        'pct_unemployed': ('Unemployment', 'Percentage of the labor force that is unemployed.'),
        'per_capita_income': ('Income per Person', 'Average income per resident, in dollars.'),
        'pct_rent_burdened': ('Rent Burden', 'Percentage of renter households with high housing costs relative to income.'),
        'eviction_filing_rate': ('Eviction Filing Rate', 'Frequency of eviction court filings relative to renter households.'),
        'pct_on_medicaid': ('Medicaid Coverage', 'Percentage of the measured population covered by Medicaid.'),
        'hospital_locations': ('Hospital Locations', 'Geographic locations of listed hospitals.'),
        'hospital_financials': ('Hospital Costs', 'Costs incurred by a hospital across its services, not specifically the costs of treating patients with firearm injuries.'),
        'profile_clinics': ('Community Clinics', 'Reported number of clinics associated with a community area.'),
        'profile_nearest_center': ('Nearest Listed Trauma Center', 'Trauma center identified as nearest in the community profile.'),
        'profile_trauma_drive_min': ('Estimated Driving Time', 'Estimated driving time to a listed trauma center, in minutes.'),
        'hospital_coverage': ('Illustrative Hospital Coverage Areas', 'Illustrative geographic areas used to describe access to hospitals.'),
    }
    definitions = []
    for field, (indicator, definition) in indicator_guide.items():
        ids = PROVENANCE[field]['source_ids'] if documented(field) else []
        source = '; '.join(dict.fromkeys(SOURCE_REGISTRY[x][0] for x in ids)) if ids else 'Not confirmed'
        definitions.append({'Indicator': indicator, 'Definition': definition, 'Source': source})
    st.dataframe(pd.DataFrame(definitions), hide_index=True, use_container_width=True, height=650,
                 column_config={'Indicator': st.column_config.TextColumn('Indicator', width='medium'),
                                'Definition': st.column_config.TextColumn('Definition', width='large'),
                                'Source': st.column_config.TextColumn('Source', width='medium')})
    st.markdown('### Methods')
    st.markdown('''- **Data preparation:** Records are organized by community area and reporting year. Duplicate records, conflicting entries and invalid numerical values are screened before analysis.
- **Statistical analysis:** The Poisson model examines associations between shooting victim counts, poverty and unemployment while accounting for population and reporting year.
- **Trauma center access:** The map calculates straight-line distances from community area centers to listed trauma centers. Dotted circles show the selected distance radius.
- **Financial analysis:** The dashboard compares reported hospital financial indicators, not firearm injury treatment costs.''')
    st.markdown('### Limitations')
    st.markdown('''- **Eviction data:** The referenced eviction data release covers 2010–2019. The reporting period of the eviction values in the merged dataset remains unconfirmed.
- **Data coverage:** Source details, measurement units and reporting periods are not confirmed for all indicators. Socioeconomic estimates may span multiple years.
- **Geography:** Community area averages do not describe individual risk.
- **Trauma care access:** Distance and estimated driving time do not measure ambulance response, actual transport time or patient destinations.
- **Hospital costs:** Hospital financial records do not identify costs or losses attributable specifically to firearm injuries. Financial reporting periods may differ from shooting reporting years.
- **Statistical findings:** Associations do not establish causation. Model estimates are not validated forecasts.''')
    st.markdown('### Notes')
    st.markdown('''- Shooting victim counts represent victims rather than distinct incidents.
- The shooting victim rate equals the number of victims divided by population, multiplied by 10,000.
- Citywide rates use the combined victim count and population across community areas.''')

st.caption('Chicago community area research dashboard. For research purposes only; not intended for emergency response or clinical guidance.')


st.markdown('<div class="cch-footer"><strong>Project credits | IDS 406 · Fall 2026</strong><br>Academic project by Mohammed Ali, Abdoul Adoum, Arhaam Fazul Al Rahman and Praise Oyebanji.<br>Faculty supervisor: Professor Ranga Chandrasekaran · Teaching assistant: Vishaal Dayashanker · Project mentor: Dr. Kendra Outler.<br>Explore sources and limitations in Glossary &amp; Resources. <a href="https://cookcountyhealth.org/" target="_blank" rel="noopener noreferrer">Visit Cook County Health</a>.</div>', unsafe_allow_html=True)
