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

NAVY = '#06264A'
TEAL = '#079DB5'
TURQUOISE = '#20B5C6'
LAVENDER = '#8B83CF'
RED = '#C84459'  # Reserved for fatal victims and adverse rate changes
PALETTE = [NAVY, TEAL, LAVENDER]

st.set_page_config(page_title='Cook County Health | Chicago Community Insights', page_icon='🏥', layout='wide')
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
        '<div class="brand-note">Chicago Community Insights<br>Research project · Dashboard filters</div></div>',
        unsafe_allow_html=True)
else:
    st.sidebar.markdown('**[Cook County Health](https://cookcountyhealth.org/)**')
    st.sidebar.caption('Chicago Community Insights · Research project')
year=st.sidebar.selectbox('Reporting year',sorted(D.year.unique(),reverse=True))
years_available=sorted(D.year.unique().tolist())
range_start,range_end=st.sidebar.select_slider(
    'Historical comparison range', options=years_available,
    value=(years_available[0],years_available[-1]),
    help='Controls the historical trends and the PDF report. The reporting-year map and KPIs remain annual.',
)
D_window=D[D.year.between(range_start,range_end)].copy()

current=D[D.year==year].copy()
areas=sorted(current.ca_name.unique())
if 'selected_area' not in st.session_state: st.session_state.selected_area='All Chicago'
if st.session_state.selected_area not in ['All Chicago']+areas: st.session_state.selected_area='All Chicago'
area=st.sidebar.selectbox('Focus community area',['All Chicago']+areas,key='selected_area')
st.sidebar.caption(f'{D.ca_num.nunique()} Chicago community areas · {D.year.min()}–{D.year.max()} shooting data')
st.markdown(f"""
<section class="cch-hero">
  <div class="eyebrow">Community health research | Chicago, Illinois</div>
  <h1>Chicago Gun Violence &amp; Healthcare Insights</h1>
  <p>Explore neighborhood conditions, shooting victim trends and access to healthcare across Chicago.</p>
  <div class="meta"><span>{html.escape(str(area))}</span><span>Reporting year: {year}</span><span>{D.ca_num.nunique()} community areas</span><span>Research dashboard</span></div>
  <p style="margin-top:15px;margin-bottom:0"><a href="https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/" target="_blank" rel="noopener noreferrer">Cook County Health — Stroger Hospital</a></p>
</section>
""", unsafe_allow_html=True)
st.caption('Research project · Source documentation and verification limitations are available in Tab 5. Not an official patient portal.')
t1,t2,t3,t4,t5=st.tabs(['🗺️ Overview & Map','📊 Socioeconomic & Regression','🏥 Healthcare & Hospital Impact','🏘️ Neighborhood Comparison','📚 Glossary & Resources'])

row=current[current.ca_name==area].iloc[0] if area!='All Chicago' else None
selected=current if row is None else current[current.ca_name==area]
shot=selected.shot_total.sum(min_count=len(selected));fatal=selected.fatal.sum(min_count=len(selected));nonfatal=selected.nonfatal.sum(min_count=len(selected));pop=selected.population.sum(min_count=len(selected))
rate=shot/pop*10000 if pop>0 else np.nan

with t1:
    st.subheader(f'{area} · {year}')
    a,b,c,d=st.columns(4)
    a.metric('Shooting victims',fmt(shot));b.metric('Fatal victims',fmt(fatal));c.metric('Nonfatal victims',fmt(nonfatal));d.metric('Victims per 10k',fmt(rate,1))
    st.markdown('#### Community insights at a glance')
    valid_rate=current.dropna(subset=['shot_rate_per_10k'])
    if not valid_rate.empty:
        highest=valid_rate.loc[valid_rate.shot_rate_per_10k.idxmax()]
        insight1=(f"**Highest recorded victim rate:** {highest.ca_name} "
                  f"({highest.shot_rate_per_10k:.1f} per 10k in {year}).")
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
                  f"({best.rate_drop:+.1f} per 10k decrease vs. {year-1}).") if best.rate_drop>0 else                  f'**Year-over-year comparison:** No community area shows a lower rate than {year-1} in the comparable data.'
    else:
        insight2=f'**Year-over-year comparison:** Comparable {year-1} rates are unavailable.'
    valid_drive=current.dropna(subset=['profile_trauma_drive_min'])
    if not valid_drive.empty:
        longest=valid_drive.loc[valid_drive.profile_trauma_drive_min.idxmax()]
        insight3=(f"**Longest estimated trauma-center drive:** {longest.ca_name} "
                  f"({longest.profile_trauma_drive_min:.1f} min; not EMS response time).")
    else:
        insight3='**Estimated trauma-center drive:** No data available.'
    for insight in (insight1,insight2,insight3):
        st.markdown('- '+insight)
    st.caption('Automatically calculated from your selected year; descriptive observations, not validated public-health recommendations.')
    left,right=st.columns([1.55,1],gap='large')
    with left:
        st.subheader('Where are shooting victims concentrated?')
        choices={'Victims per 10k':'shot_rate_per_10k','Total shooting victims':'shot_total','Poverty (%)':'pct_poverty','Unemployment (%)':'pct_unemployed'}
        choice=st.radio('Map shading',list(choices),horizontal=True)
        field=choices[choice]
        values=current.set_index('ca_num')[field].to_dict()
        v=pd.Series(list(values.values())).dropna()
        vmin=float(v.min()) if len(v) else 0; vmax=float(v.max()) if len(v) else 1
        import branca.colormap as bcm
        cmap=bcm.LinearColormap(['#fff7bc','#fec44f','#f03b20','#800026'],vmin=vmin,vmax=max(vmax,vmin+.01),caption=choice)
        show_trauma_on_main=st.checkbox('Show listed trauma hospitals on this map',value=False,key='main_trauma_markers')
        show_county=st.checkbox('Show Cook County boundary', value=True, key='cook_county_boundary')
        mp=folium.Map(location=[41.84,-87.68],zoom_start=10,tiles='OpenStreetMap',control_scale=True)
        county_geo, county_error = (
        load_cook_county_boundary() if show_county else (None, None))
        if county_error:
            st.info('Cook County boundary temporarily unavailable. The Chicago community map still works. '
                    'You can also place cook_county_boundary.geojson in the data/ folder for offline use.')
        lookup=current.set_index('ca_num')
        def style(feature):
            cid=int(feature['properties']['ca_num']);val=values.get(cid,np.nan)
            return {'fillColor':cmap(val) if pd.notna(val) else '#cbd5e1','color':'#202c3b' if row is not None and int(row.ca_num)==cid else '#657181','weight':3 if row is not None and int(row.ca_num)==cid else .8,'fillOpacity':.78}
        for feature in G['features']:
            cid=int(feature['properties']['ca_num'])
            if cid not in lookup.index: continue
            r=lookup.loc[cid]
            fields=[('Shooting victims',fmt(r.shot_total)),('Fatal victims',fmt(r.fatal)),('Nonfatal victims',fmt(r.nonfatal)),('Victims per 10k',fmt(r.shot_rate_per_10k,1)),('Poverty (%)',pct(r.pct_poverty)),('Unemployment (%)',pct(r.pct_unemployed)),('Income per person', '$'+fmt(r.per_capita_income)),('Eviction filing rate',get(r,'eviction_reference')),('Estimated trauma drive (min)',fmt(get(r,'profile_trauma_drive_min'),1)),('Clinics (source estimate)',fmt(get(r,'profile_clinics'))),('Nearest listed trauma center',str(get(r,'profile_nearest_center')) if pd.notna(get(r,'profile_nearest_center')) else 'Data unavailable'),('Medicaid reliance (%)',pct(get(r,'pct_on_medicaid')))]
            details=''.join(f'<tr><td style="padding:3px 8px">{html.escape(k)}</td><td><b>{html.escape(str(val))}</b></td></tr>' for k,val in fields)
            popup=folium.Popup(f'<div style="font:13px Arial;width:315px"><h3>{html.escape(str(r.ca_name))} · {year}</h3><table>{details}</table><p style="font-size:11px">Source definitions and years require verification. Travel time is estimated, not EMS response time.</p></div>',max_width=380)
            folium.GeoJson(feature,style_function=style,highlight_function=lambda f:{'weight':3,'color':'#111827','fillOpacity':.88},tooltip=f'{r.ca_name}: {choice} {fmt(get(r,field),1)}',popup=popup).add_to(mp)
        # Draw the county outline on top of the Chicago polygons for visibility.
        if show_county and county_geo:
            folium.GeoJson(
                county_geo, name='Cook County boundary (outline)',
                style_function=lambda f: {'color':'#6D4AC6','weight':4,'opacity':1,
                                          'fillOpacity':0,'dashArray':'9,5'},
                tooltip='Cook County boundary'
            ).add_to(mp)

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
        st.caption('Click any Chicago community polygon for its detailed popup. The dashed purple boundary shows Cook County, including suburban areas not covered by these shooting datasets. Use the left selector to update all charts.')
    with right:
        st.subheader('Top 10 community areas')
        top=current.nlargest(10,'shot_total').sort_values('shot_total')
        fig=px.bar(top,x='shot_total',y='ca_name',orientation='h',labels={'shot_total':'Shooting victims','ca_name':''},height=490)
        fig.update_traces(marker_color=TEAL);fig.update_layout(margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    if row is not None:
        st.subheader(f'{area}: socioeconomic & healthcare snapshot')
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[('Poverty', 'pct_poverty'),('Unemployment','pct_unemployed'),('Estimated trauma drive (min)','profile_trauma_drive_min'),('Clinics (estimate)','profile_clinics')]):
            col.metric(label,pct(get(row,key)) if key in ['pct_poverty','pct_unemployed'] else fmt(get(row,key),0 if key=='profile_clinics' else 1))
    st.subheader('How have shootings changed over time?')
    trend=D_window if row is None else D_window[D_window.ca_name==area]
    trend=trend.groupby('year',as_index=False)[['fatal','nonfatal']].sum()
    fig=px.bar(trend,x='year',y=['fatal','nonfatal'],barmode='stack',color_discrete_map={'fatal':RED,'nonfatal':TEAL},labels={'value':'Victims','variable':'Outcome'},height=330)
    fig.update_layout(xaxis=dict(dtick=1),margin=dict(l=0,r=0,t=5,b=0));st.plotly_chart(fig,use_container_width=True)

with t2:
    st.subheader('How are neighborhood conditions related to gun violence?')
    st.write('Compare shooting victim rates with measures such as poverty, unemployment, income, housing costs, and evictions. Each point represents one Chicago community area.')
    indicators={'Poverty (%)':'pct_poverty','Unemployment (%)':'pct_unemployed','Income per person ($)':'per_capita_income','Rent burden (%)':'pct_rent_burdened','Eviction filing rate':'eviction_filing_rate'}
    factor=st.selectbox('Compare shooting rates with',list(indicators));col=indicators[factor]
    # Short, plain-English definitions; calculated rates come from the source data.
    # We do not recompute them without the original numerator, denominator, or source metadata.
    factor_explanations = {
        'Poverty (%)': (
            'People living below the poverty threshold.',
            '(People below the poverty threshold ÷ people covered by the poverty measure) × 100.'
        ),
        'Unemployment (%)': (
            'People in the labor force who are unemployed and looking for work.',
            '(Unemployed people ÷ total labor force) × 100.'
        ),
        'Income per person ($)': (
            'Average income per person, not per household.',
            'Total income of residents ÷ total population.'
        ),
        'Rent burden (%)': (
            'Renter households spending a large share of income on housing, usually 30% or more.',
            '(Rent-burdened renter households ÷ renter households assessed) × 100.'
        ),
        'Eviction filing rate': (
            'How frequently eviction cases are filed against renter households.',
            'Eviction filings ÷ renter households; the dataset source determines whether the result is per 100, per 1,000, or another unit.'
        ),
    }
    meaning, calculation = factor_explanations[factor]
    st.info(f'**{factor} — Meaning:** {meaning}  \n**How it is calculated:** {calculation}')
    st.caption('Definitions are general. Confirm each indicator’s original source, reference year and denominator before interpreting its reported values.')
    with st.expander('See definitions and calculations for all five factors'):
        for factor_name, (short_meaning, short_formula) in factor_explanations.items():
            st.markdown(f'**{factor_name}:** {short_meaning}  \n*Calculation:* {short_formula}')
    valid=current.dropna(subset=[col,'shot_rate_per_10k'])
    fig=px.scatter(valid,x=col,y='shot_rate_per_10k',size='shot_total',hover_name='ca_name',color='shot_rate_per_10k',color_continuous_scale=[(0,'#DDF2F6'),(.5,TEAL),(1,RED)],labels={col:factor,'shot_rate_per_10k':'Shooting victims per 10,000'},height=410)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Each bubble is one community area. Association does not establish that the factor causes shootings.')
    st.subheader('Poisson Regression Model — What do the numbers suggest?')
    st.write('This analysis looks for patterns between poverty, unemployment, and shooting victim rates across Chicago. It accounts for differences in community population and reporting year. A relationship in the data does **not** mean one condition causes shootings.')
    predictors = ['pct_poverty', 'pct_unemployed']
    model_data = D[['ca_name', 'year', 'shot_total', 'population'] + predictors].dropna().copy()
    model_data = model_data[(model_data.population > 0) & (model_data.shot_total >= 0)]
    if len(model_data) < 10:
        st.info('Not enough complete records to show the analysis.')
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
            c1.metric('Records analyzed', f'{len(model_data):,}', help='Community-area and year records with complete data.')
            c2.metric('Factors examined', '2', help='Poverty and unemployment.')
            results = []
            for key, label in [('pct_poverty', 'Poverty'), ('pct_unemployed', 'Unemployment')]:
                ratio = float(np.exp(model.params[key]))
                change = (ratio - 1) * 100
                direction = 'higher' if change > 0 else 'lower' if change < 0 else 'unchanged'
                results.append({'Factor': label,
                    'Estimated rate difference for a 1-point increase': f'{abs(change):.1f}% {direction}'})
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
            if pd.notna(dispersion) and dispersion > 1.5:
                st.warning('The data vary more than this simple model expects, so these estimates may be unreliable.')
            st.subheader('Explore hypothetical neighborhood conditions')
            st.write('Choose a community and adjust poverty and unemployment to explore how the model estimates shooting rates relative to the original conditions.')
            scenario_rows = model_data[model_data.year.eq(year)].sort_values('ca_name')
            if not scenario_rows.empty:
                scenario_areas = scenario_rows.ca_name.unique().tolist()
                chosen = st.selectbox('Community area for scenario', scenario_areas,
                    index=scenario_areas.index(area) if area in scenario_areas else 0,
                    key=f'scenario_area_{year}')
                base = scenario_rows[scenario_rows.ca_name.eq(chosen)].iloc[0]
                bp, bu = float(base.pct_poverty), float(base.pct_unemployed)
                st.caption(f'Original conditions in {chosen}: poverty {bp:.1f}%, unemployment {bu:.1f}%.')
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
                st.write('Exploratory Poisson regression for victim counts, adjusted for population size and reporting year, with robust standard errors. A one-point increase means, for example, poverty changing from 20% to 21%. Repeated community measures and extra variation can limit reliability. This analysis has not been independently validated.')
        except Exception as exc:
            st.info(f'The analysis is unavailable: {exc}')

with t3:
    st.caption('Healthcare access values and hospital locations are existing project estimates; original source periods and methods remain unconfirmed. Hospital financial records are displayed as supplied; original source reconciliation is pending. See Tab 5.')
    st.subheader('Healthcare access and hospital impact')
    st.write('Explore where listed hospitals are located, estimated driving-time coverage, and hospital-wide financial information.')
    st.caption('Travel times are estimates, not measured ambulance response or transport times. Hospital costs are hospital-wide, not shooting-specific.')
    if row is not None:
        a,b,c=st.columns(3)
        a.metric('Nearest listed center',str(get(row,'profile_nearest_center')) if pd.notna(get(row,'profile_nearest_center')) else 'Data unavailable')
        b.metric('Estimated trauma drive (min)',fmt(get(row,'profile_trauma_drive_min'),1))
        c.metric('Clinics (estimate)',fmt(get(row,'profile_clinics')))
    # Interactive hospital map: highlight Stroger without duplicating its CSV entry.
    st.markdown("#### Chicago trauma hospitals")
    show_other = st.checkbox("Show other listed hospitals", value=True, key="show_other_hospitals")
    hospital_map = folium.Map(location=[41.84, -87.68], zoom_start=10, tiles="OpenStreetMap", control_scale=True)

    show_catchments = st.checkbox(
        "Show illustrative hospital coverage areas",
        value=False,
        disabled=not bool(COVERAGE.get("features")),
        key="show_hospital_catchments"
    )

    coverage_minutes = st.slider(
        "Coverage threshold (minutes)",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
        disabled=not show_catchments
    )

    stroger_lat, stroger_lon = 41.8564, -87.6663  # Approximate map coordinates
    hospital_map = folium.Map(
        location=[41.84, -87.68],
        zoom_start=10,
        tiles="OpenStreetMap", control_scale=True
    )
    other_layer = folium.FeatureGroup(name="Other listed hospitals", show=show_other)
    stroger_layer = folium.FeatureGroup(name="Cook County / Stroger Hospital", show=True)

    # Use one dedicated marker for Stroger, even if the CSV already contains it.
    for _, hospital in H.iterrows():
        hospital_name = str(hospital.get("hospital", "Unknown hospital"))
        if any(alias in hospital_name.lower() for alias in
               ("stroger", "cook county hospital")):
            continue
        if not show_other:
            continue
        lat = pd.to_numeric(hospital.get("lat"), errors="coerce")
        lon = pd.to_numeric(hospital.get("lon"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        # Hospital profiles: use the local CSV first, with campus reference details
        # for the hospitals already listed in the project. Verify before publication.
        profiles = [
            ('loyola', 'Loyola Medicine', '2160 S. First Avenue, Maywood, IL 60153', 'https://www.loyolamedicine.org/'),
            ('mount sinai', 'Sinai Chicago', '1500 S. Fairfield Avenue, Chicago, IL 60608', 'https://www.sinaichicago.org/'),
            ('northwestern', 'Northwestern Medicine', '251 E. Huron Street, Chicago, IL 60611', 'https://www.nm.org/'),
            ('rush', 'Rush University System for Health', '1653 W. Congress Parkway, Chicago, IL 60612', 'https://www.rush.edu/'),
            ('lutheran general', 'Advocate Health Care', '1775 Dempster Street, Park Ridge, IL 60068', 'https://www.advocatehealth.com/'),
            ('illinois masonic', 'Advocate Health Care', '836 W. Wellington Avenue, Chicago, IL 60657', 'https://www.advocatehealth.com/'),
            ('uchicago', 'UChicago Medicine', '5841 S. Maryland Avenue, Chicago, IL 60637', 'https://www.uchicagomedicine.org/'),
            ('university of chicago', 'UChicago Medicine', '5841 S. Maryland Avenue, Chicago, IL 60637', 'https://www.uchicagomedicine.org/'),
            ('advocate christ', 'Advocate Health Care', '4440 W. 95th Street, Oak Lawn, IL 60453', 'https://www.advocatehealth.com/'),
        ]
        system, address, website = next(((sys, addr, url) for alias, sys, addr, url in profiles
                                         if alias in hospital_name.lower()), ('Listed hospital', '', ''))
        def profile_value(*keys):
            for key in keys:
                value = hospital.get(key)
                if pd.notna(value) and str(value).strip():
                    return str(value).strip()
            return ''
        hospital_type = profile_value('type') or 'Not specified'
        trauma = profile_value('adult_level1_trauma')
        trauma_label = ('Adult Level I Trauma Center (dataset)' if trauma.lower() in ('yes','true','1','1.0')
                        else 'Adult Level I designation: ' + (trauma or 'Not specified'))
        address = profile_value('address','street_address') or address
        website_link = (f'<a href="{html.escape(website, quote=True)}" target="_blank" rel="noopener noreferrer">Official healthcare system website</a>' if website else '')
        popup_html = (
            '<div style="font-family:Arial,sans-serif;min-width:275px;line-height:1.6;color:#06264A">'
            '<div style="background:#06264A;color:white;padding:10px;border-radius:7px 7px 0 0">'
            f'<strong>{html.escape(system.upper())}</strong><br>{html.escape(hospital_name)}</div>'
            '<div style="padding:10px">'
            f'<b>{html.escape(trauma_label)}</b><br>'
            f'{html.escape(address or "Address not available")}<br>'
            f'{html.escape(hospital_type)}<br>{website_link}'
            '<p style="font-size:11px;margin-bottom:0">Campus details and trauma designation require confirmation before publication.</p></div></div>'
        )
        folium.Marker(
            [float(lat), float(lon)], tooltip=hospital_name,
            popup=folium.Popup(popup_html, max_width=380),
            icon=folium.Icon(color="cadetblue", icon="plus-sign")
        ).add_to(other_layer)

    stroger_popup = """
    <div style="font-family:Arial,sans-serif;min-width:275px;line-height:1.6;color:#06264A">
      <div style="background:#06264A;color:white;padding:10px;border-radius:7px 7px 0 0">
        <strong>COOK COUNTY HEALTH</strong><br>
        John H. Stroger Jr. Hospital
      </div>
      <div style="padding:10px">
        <b>Adult Level I Trauma Center</b><br>
        1969 W. Ogden Avenue<br>Chicago, IL 60612<br>
        Public teaching hospital · Emergency and trauma care<br>
        <a href="https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/" target="_blank" rel="noopener noreferrer">
          Cook County Health — official website</a>
      </div>
    </div>
    """
    folium.Marker(
        [stroger_lat, stroger_lon],
        tooltip="★ Cook County Health — Stroger Hospital",
        popup=folium.Popup(stroger_popup, max_width=360),
        icon=folium.Icon(color="darkblue", icon="plus-sign")
    ).add_to(stroger_layer)
    
    
    if show_catchments:
        coverage_layer = folium.FeatureGroup(
            name=f"{coverage_minutes}-minute driving coverage",
            show=True
        )

        hospital_names = sorted({
            str(feature.get("properties", {}).get("hospital", "Hospital"))
            for feature in COVERAGE.get("features", [])
        })

        hospital_palette = [
            "#E63946",  # Red
            "#457B9D",  # Blue
            "#2A9D8F",  # Teal
            "#F4A261",  # Orange
            "#9B5DE5",  # Purple
            "#E9C46A",  # Gold
            "#F15BB5",  # Pink
            "#4361EE",  # Royal blue
            "#6A994E",  # Green
            "#BC6C25",  # Brown
        ]

        hospital_colors = {
            name: hospital_palette[i % len(hospital_palette)]
            for i, name in enumerate(hospital_names)
        }

        matching_areas = [
            feature for feature in COVERAGE.get("features", [])
            if int(feature.get("properties", {}).get("minutes", -1))
            == coverage_minutes
        ]

        for feature in matching_areas:
            hospital_name = str(
                feature.get("properties", {}).get("hospital", "Hospital")
            )

            folium.GeoJson(
                feature,
                
                style_function=lambda x, color=hospital_colors[hospital_name]: {
                    "color": color,
                    "weight": 2,
                    "fillColor": color,
                    "fillOpacity": 0.22
                },

                tooltip=folium.Tooltip(
                    f"{html.escape(hospital_name)}: "
                    f"{coverage_minutes}-minute driving area"
                )
            ).add_to(coverage_layer)

        coverage_layer.add_to(hospital_map)

        if not matching_areas:
            st.warning(
                f"No coverage data found for {coverage_minutes} minutes."
            )
        else:
            st.caption(
                f"Showing modeled {coverage_minutes}-minute driving-time "
                "coverage areas. Overlapping shaded areas indicate "
                "potential access to multiple hospitals. "
                "These are not actual patient catchment areas, "
                "ambulance response times, or traffic-aware estimates."
            )

    other_layer.add_to(hospital_map)
    stroger_layer.add_to(hospital_map)
    folium.LayerControl(collapsed=False).add_to(hospital_map)
    st_folium(
        hospital_map,
        height=470,
        use_container_width=True,
        key="hospital_map",
        returned_objects=[]
    )
    st.caption("Navy: Stroger Hospital · Turquoise-blue: other locally listed hospitals, if available. "
               "Map coordinates are approximate; confirm locations and trauma designations before publication.")

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
    st.info("The neighborhood shooting totals in this dashboard are not counts of "
            "patients treated at Stroger Hospital. Hospital-specific firearm injury "
            "counts have not been verified or added.")
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
        st.markdown('**Stroger Hospital financial summary**')
        show_financial_table(stroger_data, include_type=False)
    else:
        st.info('No Stroger financial record is available in the hospital CSV.')

    st.subheader('Hospital financial comparison')
    st.caption('Hospital-wide amounts from trauma_hospitals.csv, in US dollars. Fiscal periods may differ. Original source reconciliation is pending; see Tab 5.')
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
    st.caption('These figures are not firearm-specific treatment expenses, reimbursement or financial losses.')
    st.markdown('[CMS Hospital Provider Cost Report — verification reference](https://data.cms.gov/provider-compliance/cost-reports/hospital-provider-cost-report)')

    st.subheader('Estimated access compared with shooting rates')
    access_preview=current.dropna(subset=['profile_trauma_drive_min','shot_rate_per_10k'])
    if not access_preview.empty:
        hospital_scatter=px.scatter(access_preview,x='profile_trauma_drive_min',
            y='shot_rate_per_10k',hover_name='ca_name',
            labels={'profile_trauma_drive_min':'Estimated drive to listed trauma center (minutes)',
                    'shot_rate_per_10k':'Shooting victims per 10k'},
            color_discrete_sequence=[TEAL],height=360)
        st.plotly_chart(chart_style(hospital_scatter),use_container_width=True)
    st.caption('Driving minutes are dataset estimates, not ambulance transport or EMS response times.')
    st.subheader('Neighborhood healthcare comparison')
    with st.expander('Explore detailed neighborhood healthcare figures'):
        st.dataframe(current[['ca_name','shot_rate_per_10k','profile_nearest_center','profile_trauma_drive_min','profile_clinics']].sort_values('shot_rate_per_10k',ascending=False).rename(columns={'ca_name':'Community area','shot_rate_per_10k':'Victims per 10,000','profile_nearest_center':'Nearest listed center','profile_trauma_drive_min':'Estimated drive (minutes)','profile_clinics':'Estimated clinics'}),hide_index=True,use_container_width=True)

with t4:
    st.caption('Comparisons use the existing community-area dataset. Source details and verification status are available in Tab 5.')
    st.subheader('Neighborhood comparison')
    st.write('Compare shooting victim rates and neighborhood conditions across Chicago. Choose communities below, or leave the selection empty to see all areas.')
    measures = {
        'Shooting victims per 10,000': 'shot_rate_per_10k',
        'Total shooting victims': 'shot_total',
        'Poverty (%)': 'pct_poverty',
        'Unemployment (%)': 'pct_unemployed',
        'Income per person ($)': 'per_capita_income',
    }
    filter_col, rank_col = st.columns([1.6, 1])
    with filter_col:
        neighborhoods = st.multiselect('Community areas (empty means all)', areas,
            default=[area] if area != 'All Chicago' else [], key='compare_filter')
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
                st.info('No values available for this indicator.')
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
                        labels={'year':'Year','rate':'Victims per 10,000'}, height=400)
                    st.caption('Citywide trend. Select specific communities to compare their lines.')
                else:
                    trend = px.line(history, x='year', y='shot_rate_per_10k',
                        color='ca_name', markers=True,
                        labels={'year':'Year','shot_rate_per_10k':'Victims per 10,000','ca_name':'Community area'},
                        height=400)
                    if len(neighborhoods) > 8:
                        st.caption('Many selected communities can make the trend chart crowded; choose fewer for a clearer view.')
                trend.update_layout(xaxis=dict(dtick=1), hovermode='x unified')
                st.plotly_chart(chart_style(trend), use_container_width=True)
        st.caption('Charts describe observed patterns; they do not establish causes. Socioeconomic data may refer to different source years.')
        st.markdown('**Community-area data**')
        display_cols = {
            'ca_name':'Community area',
            'shot_total':'Shooting victims',
            'shot_rate_per_10k':'Victims per 10,000',
            'pct_poverty':'Poverty (%)',
            'pct_unemployed':'Unemployment (%)',
            'per_capita_income':'Income per person ($)',
        }
        full = subset[[c for c in display_cols if c in subset.columns]].rename(columns=display_cols)
        formats = {'Shooting victims':'{:,.0f}', 'Victims per 10,000':'{:,.1f}',
            'Poverty (%)':'{:,.1f}%', 'Unemployment (%)':'{:,.1f}%',
            'Income per person ($)':'${:,.0f}'}
        st.dataframe(full.style.format({k:v for k,v in formats.items() if k in full}, na_rep='N/A'),
            hide_index=True, use_container_width=True)
        st.download_button('Download comparison CSV',
            full.to_csv(index=False).encode('utf-8'),
            file_name=f'cook_county_neighborhood_comparison_{year}.csv',
            mime='text/csv', key='download_comparison')

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
    result=data if community=='All Chicago' else data[data.ca_name.eq(community)]
    patients=result.shot_total.sum(min_count=len(result))
    fatal_n=result.fatal.sum(min_count=len(result))
    pop_sum=result.population.sum()
    patient_rate=patients/pop_sum*10000 if pd.notna(pop_sum) and pop_sum>0 else np.nan
    kpi_data=[['Shooting victims','Fatal victims','Victims per 10k'],
              [fmt(patients),fmt(fatal_n),fmt(patient_rate,1)]]
    kpi_table=Table(kpi_data,colWidths=[175]*3,rowHeights=[27,35])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#EAF7FA')),
        ('TEXTCOLOR',(0,1),(-1,1),navy),('FONTSIZE',(0,0),(-1,-1),10),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.extend([kpi_table,Spacer(1,12),Paragraph('Selected-year observations',styles['CCHHeading'])])
    valid=data.dropna(subset=['shot_rate_per_10k'])
    if community=='All Chicago' and not valid.empty:
        top=valid.sort_values('shot_rate_per_10k',ascending=False).iloc[0]
        story.append(Paragraph(f'Highest observed community rate: {html.escape(str(top.ca_name))} ({top.shot_rate_per_10k:.1f} victims per 10k).',styles['CCHBody']))
    elif community!='All Chicago' and not result.empty:
        r=result.iloc[0]
        if pd.notna(r.get('profile_trauma_drive_min',np.nan)):
            story.append(Paragraph(f'Estimated trauma-center driving time: {r.profile_trauma_drive_min:.1f} minutes. Not EMS transport time.',styles['CCHBody']))

    history=all_history[all_history.year.between(start_year,end_year)]
    history=history if community=='All Chicago' else history[history.ca_name.eq(community)]
    history=history.dropna(subset=['shot_total','population']).groupby('year',as_index=False)[['shot_total','population']].sum()
    history=history[history.population.gt(0)].copy()
    if not history.empty:
        history['victim_rate']=history.shot_total/history.population*10000
        fig,ax=plt.subplots(figsize=(7.1,2.5))
        ax.plot(history.year,history.victim_rate,marker='o',color=TEAL,linewidth=2)
        ax.set_ylabel('Victims per 10k'); ax.set_xlabel('Year');ax.set_title('Annual shooting victim rate')
        ax.grid(axis='y',alpha=.25);fig.tight_layout()
        img=BytesIO();fig.savefig(img,format='png',dpi=140,bbox_inches='tight');plt.close(fig);img.seek(0)
        story.extend([Paragraph('Historical trend',styles['CCHHeading']),Image(img,width=505,height=178)])
    if community=='All Chicago':
        top=data.dropna(subset=['shot_rate_per_10k']).sort_values('shot_rate_per_10k',ascending=False).head(6)
        if not top.empty:
            fig,ax=plt.subplots(figsize=(7.0,2.8))
            ax.barh(top.ca_name.iloc[::-1],top.shot_rate_per_10k.iloc[::-1],color=TEAL)
            ax.set_xlabel('Victims per 10k');fig.tight_layout()
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
    st.markdown('**Start here:** The dashboard describes recorded shooting victims and compares community-level patterns. It does not identify why an individual shooting occurred or predict the effects of a policy.')
    st.markdown('### At a glance')
    st.markdown('''- **Shooting victims** counts people, not separate shooting incidents.
- **Victims per 10,000** makes comparisons fairer between communities of different sizes.
- **Estimated driving time** describes modeled access to listed hospitals, not ambulance response time.
- **Hospital costs** describe whole hospitals, not the costs of treating gunshot patients.
- **Model findings** show associations, not proof of causes.''')
    st.markdown('''This dashboard integrates Chicago community-area shooting-victim data, neighborhood indicators, hospital records, and geographic boundaries. The information supports **exploratory comparisons**, not causal conclusions, clinical decisions, or estimates of the cost of individual gunshot injuries.''')
    st.markdown('### 1. Data sources and original publications')
    st.write('Find the public organizations and datasets behind the maps, community indicators, and charts.')
    sources = [
        ('Chicago Police Department / City of Chicago',
         'Violence Reduction — Victims of Homicides and Non-Fatal Shootings',
         'https://data.cityofchicago.org/d/gumc-mgzr',
         'Recorded fatal and nonfatal shooting victimizations.'),
        ('Law Center for Better Housing (LCBH)',
         'Chicago Evictions Data Portal',
         'https://eviction.lcbh.org/',
         'Historical eviction filings and rates, with Release 2 covering 2010–2019.'),
        ('Chicago Health Atlas', 'Community health and social indicators',
         'https://chicagohealthatlas.org/',
         'Community-level health and social indicators.'),
        ('U.S. Census Bureau', 'American Community Survey (ACS)',
         'https://www.census.gov/programs-surveys/acs',
         'Population and socioeconomic estimates.'),
        ('City of Chicago', 'Community Area Boundaries',
         'https://data.cityofchicago.org/d/cauq-8yn6',
         'Geographic reference for the 77 Chicago community areas.'),
        ('Cook County GIS', 'Cook County GIS data portal',
         'https://hub-cookcountyil.opendata.arcgis.com/',
         'Geographic boundary information for the county outline shown on the map.')
    ]
    for agency, title, url, purpose in sources:
        st.markdown(f'**{agency} — [{title}]({url})**  \n{purpose}')
    st.markdown('### Additional official references (new)')
    purposes = {
        'cms':'Hospital-wide financial cost reports; candidate source, not confirmed CSV lineage.',
        'idph_trauma':'Official trauma designations and facility reference.',
        'idph_report':'Hospital utilization, financial and insurance-related measures; reference only unless reconciled.',
        'cch':'Official Stroger Hospital information.',
        'community_acs':'Community-area ACS estimates; confirm release and geography before use.',
        'osm':'Map background attribution only; not a routing or travel-time dataset.',
        'cdph_clinics':'Official directory of listed primary care community health centers; candidate source for replacement clinic counts, not yet imported.',
        'osrm':'Candidate method for modeled road travel times; not the confirmed source of existing project estimates.',
    }
    for source_id, purpose in purposes.items():
        agency,title,url=SOURCE_REGISTRY[source_id]
        st.markdown(f'**{agency} — [{title}]({url})**  \n{purpose}')
    st.markdown('### Data Sources, Definitions & Documentation')
    st.caption('Use this guide to understand each indicator, its meaning and the status of its supporting documentation. Definitions describe intended measures, not verified source metadata.')
    st.markdown('[Census poverty definition](https://www.census.gov/topics/income-poverty/guidance/poverty-measures.html) · [Census housing cost reference](https://www.census.gov/library/stories/2022/12/housing-costs-burden.html)')
    indicator_guide = {'eviction_filing_rate': ('Eviction Filing Rate',
                              'Eviction court filings relative to renter households; the rate unit is not yet '
                              'confirmed.'),
     'fatal': ('Fatal Shooting Victims', 'Recorded shooting victims who died from their injuries.'),
     'hospital_coverage': ('Illustrative Hospital Coverage Areas',
                           'Modeled areas reachable within a selected driving time; not actual patient catchment '
                           'areas.'),
     'hospital_financials': ('Hospital Financial Information',
                             'Hospital-wide costs, patient revenue and care costs; not specific to firearm '
                             'injuries.'),
     'hospital_locations': ('Hospital Locations',
                            'Geographic locations of listed hospitals; coordinates and designations require '
                            'verification.'),
     'nonfatal': ('Nonfatal Shooting Victims', 'Recorded shooting victims who survived their injuries.'),
     'pct_on_medicaid': ('Medicaid Coverage (Percentage)',
                         'Reported share covered by Medicaid; confirm the population denominator.'),
     'pct_poverty': ('Poverty (Percentage)',
                     'Share of people with income below the applicable poverty threshold; confirm the source '
                     'definition.'),
     'pct_rent_burdened': ('Rent Burden (Percentage)',
                           'Share of renter households spending a substantial portion of income on housing, '
                           'commonly 30% or more; confirm the threshold.'),
     'pct_unemployed': ('Unemployment (Percentage)',
                        'Share of the labor force without work and seeking employment; confirm the source '
                        'denominator.'),
     'per_capita_income': ('Income per Person ($)',
                           'Average income per resident; confirm the reference period and dollar basis.'),
     'population': ('Community Population', 'Estimated number of residents in a community area.'),
     'profile_clinics': ('Community Clinics',
                         'Reported number of clinics associated with a community; inclusion criteria require '
                         'confirmation.'),
     'profile_nearest_center': ('Nearest Listed Trauma Center',
                                'Hospital identified as nearest in the existing community profile; method '
                                'unconfirmed.'),
     'profile_trauma_drive_min': ('Estimated Travel Time to Trauma Center (Minutes)',
                                  'Estimated driving minutes, not measured ambulance response or transport '
                                  'time.'),
     'shot_rate_per_10k': ('Shooting Victim Rate (per 10,000 Residents)',
                           'Total shooting victims divided by population, multiplied by 10,000.'),
     'shot_total': ('Total Shooting Victims',
                    'Recorded fatal and nonfatal shooting victims combined; not a count of distinct incidents.')}
    st.warning('No original data exports were supplied with the code. The references below identify appropriate providers, not proof that existing data came from them. No new numerical dataset has been imported.')
    audit=[]
    for field, ids in FIELD_SOURCES.items():
        entry=PROVENANCE.get(field,{}) if isinstance(PROVENANCE.get(field,{}),dict) else {}
        audit.append({'Indicator':indicator_guide[field][0], 'What it means':indicator_guide[field][1], 'Reference providers':'; '.join(SOURCE_REGISTRY[x][0] for x in ids) or 'Not established',
                      'Reference period':entry.get('reference_period','Unknown'),
                      'Original file':entry.get('original_file','Not supplied'),
                      'Status':provenance_status(field)})
    audit_df=pd.DataFrame(audit)
    st.dataframe(audit_df,hide_index=True,use_container_width=True,height=420, column_config={
        'Indicator': st.column_config.TextColumn('Indicator', width='large'),
        'What it means': st.column_config.TextColumn('What it means', width='large'),
        'Reference providers': st.column_config.TextColumn('Reference providers', width='large'),
    })
    with st.expander('Read all indicator definitions', expanded=False):
        for indicator, definition in indicator_guide.values():
            st.markdown(f'**{indicator}** — {definition}')
    st.download_button('Download data dictionary and source documentation (CSV)',audit_df.assign(**{'Technical field':list(FIELD_SOURCES)}).to_csv(index=False).encode('utf-8'),file_name='source_audit.csv',mime='text/csv')
    st.caption('Source documentation file: data/source_provenance.json. Each field entry needs source_ids, reference_period, original_file and method. Add entries only after reviewing the original export and its transformation. Existing routing fields remain provisional until their source and method are documented.')
    st.caption('Current-year totals may be incomplete. Full-year year-over-year comparisons are disabled for the current calendar year.')
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Community areas',fmt(D.ca_num.nunique()))
    k2.metric('Years represented',f'{D.year.min()}–{D.year.max()}')
    k3.metric('Community-year rows',fmt(len(D)))
    k4.metric('Hospitals in file',fmt(len(H)))
    st.markdown('### 2. How are the numbers calculated?')
    st.write('The main comparison measure is shooting victims per 10,000 residents. For example, 20 victims among 10,000 residents is a rate of 20 per 10,000.')
    with st.expander('Definitions and data preparation',expanded=False):
        st.markdown('''**Unit of analysis:** One community area in one reporting year. The app removes exact duplicate rows, excludes conflicting community-area/year keys, validates numerical ranges and recalculates victim rates. These checks do not establish source accuracy. Some socioeconomic profile fields may repeat across years rather than represent new annual measurements.

**Shooting victims:** People recorded as fatally or nonfatally shot; this is not necessarily the number of distinct incidents.

**Victims per 10,000 residents:** `(shooting victims / population) × 10,000`. For all-Chicago summaries, victims and populations are summed before calculating the rate.

**Year-over-year change:** Current annual rate minus the previous annual rate for the same community. Percentage change is computed only when the prior rate is greater than zero.

**Socioeconomic variables:** Poverty, unemployment, income per person, rent burden, and eviction filing rate are displayed as provided in the merged file. Source reference periods, denominators and eviction rate units require validation.''')
    with st.expander('How the statistical model and sliders work',expanded=False):
        st.markdown('''**Current model:** Exploratory Poisson generalized linear model of shooting-victim counts with poverty and unemployment percentages, reporting-year indicators, and a log-population offset. The app reports heteroskedasticity-robust (HC1) standard errors, fitted rate ratios and a Pearson dispersion diagnostic. When dispersion is high, it offers an exploratory negative-binomial comparison.

**Observed versus estimated:** The scatterplot compares actual victim counts with the model's in-sample estimates, not a direct poverty-versus-shootings relationship.

**What-if controls:** Sliders change model inputs and display the relative fitted rate. They **do not** estimate the effect of a real intervention. Repeated area observations, time-varying measurement and unmeasured confounders limit inference. The model has not been independently validated for forecasting or policy use.''')
    with st.expander('Hospital access and financial methodology'):
        st.markdown('''**Access:** Existing travel times are displayed as unverified project estimates. Coverage polygons require documented routing methodology. They are not ambulance response times, patient transport times, or observed patient catchment areas.

**Finances:** The existing trauma_hospitals.csv amounts are displayed without changing their values. CMS and IDPH are verification references; original source reconciliation remains pending. Hospital records show hospital-wide financial indicators. They cannot establish the cost, reimbursement, uncompensated care, or net financial loss attributable specifically to gunshot patients. Financial reporting years may differ from shooting years.''')
    st.markdown('### 3. Understanding the findings')
    st.write('These limitations matter when comparing neighborhoods or using findings to guide further research.')
    limitations = [
        ('Recent eviction records','Community-area eviction filing records for 2020–2025 have been requested but are not available in the project inputs; historical eviction values should not be interpreted as recent annual observations.'),
        ('Indicator timing','Socioeconomic estimates may cover multi-year reference periods or be repeated across reporting years.'),
        ('Geographic interpretation','Community-area averages conceal variation within areas and do not identify individual risk.'),
        ('Healthcare access','Modeled travel reach excludes observed ambulance dispatch, traffic conditions, treatment routing and patient origin.'),
        ('Hospital economics','Hospital-wide financial data do not isolate gunshot-specific expenditures or losses.'),
        ('Statistical interpretation','Observed associations are not causal effects; model estimates are exploratory and not validated forecasts.')
    ]
    for heading, explanation in limitations:
        st.markdown(f'**{heading}.** {explanation}')
    st.markdown('### 4. Glossary and downloads')
    st.write('The data dictionary explains the column names used in the downloadable research files.')
    dictionary = pd.DataFrame([
        ('ca_num','Chicago community-area identifier'),('ca_name','Community-area name'),('year','Shooting reporting year'),
        ('shot_total','Total shooting victims'),('fatal','Fatal shooting victims'),('nonfatal','Nonfatal shooting victims'),
        ('population','Population denominator'),('shot_rate_per_10k','Shooting victims per 10,000 residents'),
        ('pct_poverty','Poverty percentage; source year needs confirmation'),('pct_unemployed','Unemployment percentage; source year needs confirmation'),
        ('per_capita_income','Income per person'),('pct_rent_burdened','Rent-burdened household percentage'),
        ('eviction_filing_rate','Eviction filing rate; unit and source years need confirmation'),
        ('profile_trauma_drive_min','Estimated driving minutes to listed trauma center'),('profile_clinics','Estimated clinic count')
    ],columns=['Field','Interpretation'])
    with st.expander('View technical field names and definitions'):
        st.dataframe(dictionary[dictionary.Field.isin(D.columns)],hide_index=True,use_container_width=True)
    with st.expander('Download research data (CSV)'):
        d1,d2=st.columns(2)
        d1.download_button('Download merged dataset',D.to_csv(index=False).encode('utf-8'),file_name='chicago_community_area_year_merged.csv',mime='text/csv',key='methods_merged')
        d2.download_button('Download selected-year data',current.to_csv(index=False).encode('utf-8'),file_name=f'chicago_{year}.csv',mime='text/csv',key='methods_year')

with st.sidebar.expander('Download dashboard report (PDF)',expanded=False):
    st.write(f'Generate a report for **{area}**, reporting year **{year}**, trend period **{range_start}–{range_end}**. Includes a KPI summary, charts, and limitations.')
    report_selection=(str(area),int(year),int(range_start),int(range_end))
    if st.button('Generate PDF report',key='generate_report'):
        try:
            st.session_state['pdf_report_bytes']=create_client_pdf(current,D,H,year,area,range_start,range_end)
            st.session_state['pdf_report_selection']=report_selection
        except ImportError:
            st.error('PDF export requires reportlab and matplotlib. Add both packages to requirements.txt.')
        except Exception as export_error:
            st.error(f'Could not create the PDF report for this selection: {export_error}')
    if (st.session_state.get('pdf_report_selection') == report_selection
            and st.session_state.get('pdf_report_bytes')):
        st.download_button('Download dashboard report',
            data=st.session_state['pdf_report_bytes'],
            file_name=f'cook_county_health_{area.lower().replace(" ","_")}_{year}.pdf',
            mime='application/pdf',key='download_pdf')
    else:
        st.caption('Click Generate PDF report first. Reports are rebuilt when you change the selection.')

st.caption('Chicago community-area research dashboard · Exploratory information, not emergency-response or clinical guidance')


st.markdown('<div class="cch-footer"><strong>Chicago Community Insights</strong> · Research project<br>Explore sources and limitations in Glossary &amp; Resources. <a href="https://cookcountyhealth.org/" target="_blank" rel="noopener noreferrer">Visit Cook County Health</a>.</div>', unsafe_allow_html=True)
