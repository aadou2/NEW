"""Trauma center options by community ("Level 2" access view).

For each Chicago community area: straight-line distance to every adult Level I
trauma center, how many are within a chosen distance, and whether Stroger is
one of them. Used by app.py in the Healthcare & Hospital Impact tab:

    from trauma_options import render_trauma_options
    render_trauma_options(current, G, H, row, year)
"""
import html

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


def fmt(x, dec=0):
    return 'N/A' if pd.isna(x) else f'{x:,.{dec}f}'

# ---------------------------------------------------------------------------
# Trauma center options by community ("Level 2" access view)
# For each community area: distance to every adult Level I trauma center,
# how many are within a chosen distance, and whether Stroger is one of them.
# ---------------------------------------------------------------------------

# Approximate campus coordinates. The CSV coordinates are rounded to two decimals
# (up to ~0.4 mile off), so these are used when a hospital name matches.
# Verify against IDPH / hospital addresses before publication.
CAMPUS_COORDS = {
    'stroger': (41.8730, -87.6740),          # 1969 W. Ogden Ave
    'mount sinai': (41.8606, -87.6944),      # 1500 S. Fairfield Ave
    'loyola': (41.8590, -87.8346),           # 2160 S. First Ave, Maywood
    'northwestern': (41.8950, -87.6205),     # 251 E. Huron St
    'lutheran general': (42.0397, -87.8483), # 1775 Dempster St, Park Ridge
    'illinois masonic': (41.9365, -87.6510), # 836 W. Wellington Ave
    'uchicago': (41.7886, -87.6047),         # 5841 S. Maryland Ave
    'advocate christ': (41.7205, -87.7330),  # 4440 W. 95th St, Oak Lawn
}
SHORT_NAMES = {
    'stroger': 'Stroger', 'mount sinai': 'Mount Sinai', 'loyola': 'Loyola',
    'northwestern': 'Northwestern', 'lutheran general': 'Lutheran General',
    'illinois masonic': 'Illinois Masonic', 'uchicago': 'UChicago', 'advocate christ': 'Advocate Christ',
}


def level1_centers(hospitals):
    """Adult Level I trauma centers only (drops e.g. Rush), with campus coordinates."""
    rows = []
    for _, h in hospitals.iterrows():
        flag = str(h.get('adult_level1_trauma', '')).strip().lower()
        if flag not in ('yes', 'true', '1', '1.0'):
            continue
        name = str(h.get('hospital', ''))
        key = next((k for k in CAMPUS_COORDS if k in name.lower()), None)
        lat, lon = CAMPUS_COORDS[key] if key else (pd.to_numeric(h.get('lat'), errors='coerce'),
                                                     pd.to_numeric(h.get('lon'), errors='coerce'))
        if pd.isna(lat) or pd.isna(lon):
            continue
        rows.append({'name': SHORT_NAMES.get(key, name), 'full_name': name, 'lat': float(lat), 'lon': float(lon),
                     'is_stroger': key == 'stroger'})
    return pd.DataFrame(rows)


def polygon_centroid(geometry):
    """Area-weighted centroid of the largest polygon (no shapely dependency)."""
    polys = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
    best = None
    for poly in polys:
        ring = np.asarray(poly[0], dtype=float)
        x, y = ring[:, 0], ring[:, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        area = cross.sum() / 2
        if area == 0:
            continue
        cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * area)
        cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * area)
        if best is None or abs(area) > best[0]:
            best = (abs(area), cy, cx)
    return (best[1], best[2]) if best else (np.nan, np.nan)


def miles_between(lat1, lon1, lat2, lon2):
    """Straight-line (great-circle) distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False)
def community_center_distances(geo, hospitals):
    """One row per community area x Level I center, with straight-line miles."""
    centers = level1_centers(hospitals)
    rows = []
    for feature in geo['features']:
        lat, lon = polygon_centroid(feature['geometry'])
        for _, c in centers.iterrows():
            rows.append({'ca_num': int(feature['properties']['ca_num']), 'c_lat': lat, 'c_lon': lon,
                         'center': c['name'], 'is_stroger': c['is_stroger'],
                         'miles': float(miles_between(lat, lon, c['lat'], c['lon']))})
    return pd.DataFrame(rows), centers


def hospital_details(name, hospitals):
    matches = hospitals[hospitals['hospital'].astype(str).eq(name)]
    hospital = matches.iloc[0] if not matches.empty else pd.Series(dtype=object)
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
    profiles.insert(0, ('stroger', 'Cook County Health', '1969 W. Ogden Avenue, Chicago, IL 60612', 'https://cookcountyhealth.org/locations/john-h-stroger-jr-hospital-of-cook-county/'))
    system, address, website = next(((sys, addr, url) for alias, sys, addr, url in profiles if alias in name.lower()), ('Listed hospital', '', ''))
    def value(*keys):
        for key in keys:
            v = hospital.get(key)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        return ''
    address = value('address', 'street_address') or address or 'Address not available'
    kind = value('type') or ('Public teaching hospital · Emergency and trauma care' if 'stroger' in name.lower() else 'Hospital type not specified')
    link = f'<br><a href="{html.escape(website, quote=True)}" target="_blank" rel="noopener noreferrer">Official healthcare system website</a>' if website else ''
    details = (f'<b>{html.escape(system)}</b><br>{html.escape(name)}<br>'
               f'<b>Adult Level I Trauma Center (dataset)</b><br>{html.escape(address)}<br>{html.escape(kind)}')
    tooltip = '<div style="font:13px Arial;color:#06264A;line-height:1.5">' + details + '</div>'
    popup = '<div style="font:14px Arial;color:#06264A;line-height:1.6;min-width:260px">' + details + link + '<br><small>Campus details and designation require confirmation before publication.</small></div>'
    return tooltip, popup


def render_trauma_options(current, geo, hospitals, focus_row, year):
    st.markdown('#### Trauma center options by community')
    st.write('For each community area: how far it is from every adult Level I trauma center, and how many '
             'centers are within the distance you choose. Where several centers are close, the question becomes '
             'which one patients are taken to, and why.')
    dist, centers = community_center_distances(geo, hospitals)
    if dist.empty:
        st.info('No adult Level I trauma centers with locations are available in the hospital file.')
        return

    threshold = st.slider('Distance threshold (miles)', 2.0, 10.0, 5.0, 0.5, key='options_miles',
                          help='5 miles is the distance used in the Chicago "trauma deserts" study '
                               '(Crandall et al., American Journal of Public Health, 2013).')

    # Summarise each community: nearest three centers, count within threshold, Stroger status.
    dist = dist.sort_values(['ca_num', 'miles'])
    summary = []
    for ca, g in dist.groupby('ca_num'):
        near = g.head(3)
        within = g[g.miles <= threshold]
        stroger = g[g.is_stroger]
        summary.append({
            'ca_num': ca, 'c_lat': g.c_lat.iloc[0], 'c_lon': g.c_lon.iloc[0],
            'options': len(within),
            'within_list': ', '.join(f'{r.center} ({r.miles:.1f} mi)' for r in within.itertuples()) or 'None',
            'nearest_three': ' · '.join(f'{r.center} {r.miles:.1f} mi' for r in near.itertuples()),
            'nearest_miles': near.miles.iloc[0],
            'stroger_miles': stroger.miles.iloc[0] if not stroger.empty else np.nan,
            'stroger_option': (not stroger.empty) and stroger.miles.iloc[0] <= threshold,
        })
    summary = pd.DataFrame(summary).merge(
        current[['ca_num', 'ca_name', 'shot_total', 'population']], on='ca_num', how='left')

    # Headline numbers: people first.
    def shot_in(mask):
        return int(summary.loc[mask, 'shot_total'].fillna(0).sum())
    desert = summary.options.eq(0)
    multi = summary.options.ge(2)
    stroger_shared = summary.stroger_option & multi
    m1, m2, m3 = st.columns(3)
    for col, label, mask in [(m1, f'No Level I center within {threshold:g} mi', desert),
                             (m2, 'Two or more centers within reach', multi),
                             (m3, 'Stroger is one of several options', stroger_shared)]:
        col.metric(label, f'{int(mask.sum())} areas')
        col.caption(f'{shot_in(mask):,} shooting victims recorded in these areas in {year}')

    # Map: shade by number of options; lines from the focus area to its nearest centers.
    option_colors = {0: '#C84459', 1: '#F2C46D', 2: '#7FC8D2', 3: '#079DB5'}
    def color_for(n): return option_colors[min(int(n), 3)]
    lookup = summary.set_index('ca_num')
    omap = folium.Map(location=[41.84, -87.70], zoom_start=10, tiles='OpenStreetMap', control_scale=True)
    omap.fit_bounds([[41.64, -87.94], [42.03, -87.52]])
    for feature in geo['features']:
        cid = int(feature['properties']['ca_num'])
        if cid not in lookup.index:
            continue
        r = lookup.loc[cid]
        is_focus = focus_row is not None and int(focus_row.ca_num) == cid
        tip = (f"<b>{html.escape(str(r.ca_name))}</b><br>"
               f"{int(r.options)} Level I center(s) within {threshold:g} mi: {html.escape(r.within_list)}<br>"
               f"Nearest: {html.escape(r.nearest_three)}<br>"
               f"People shot ({year}): {fmt(r.shot_total)}")
        folium.GeoJson(feature,
            style_function=lambda f, c=color_for(r.options), w=3 if is_focus else .7: {
                'fillColor': c, 'color': '#202c3b' if w > 1 else '#657181', 'weight': w, 'fillOpacity': .62},
            highlight_function=lambda f: {'weight': 3, 'color': '#111827', 'fillOpacity': .8},
            tooltip=folium.Tooltip(tip)).add_to(omap)
    for c in centers.itertuples():
        _, popup = hospital_details(c.full_name, hospitals)
        folium.CircleMarker([c.lat, c.lon], radius=8 if c.is_stroger else 6,
            color='#06264A', weight=2, fill=True, fill_color='#06264A' if c.is_stroger else '#FFFFFF',
            fill_opacity=1,
            popup=folium.Popup(popup, max_width=380)).add_to(omap)
        folium.Circle([c.lat, c.lon], radius=threshold * 1609.34, color='#06264A' if c.is_stroger else '#457B9D',
            weight=1.5 if c.is_stroger else 1, fill=False, dash_array=None if c.is_stroger else '4,6').add_to(omap)
    if focus_row is not None and int(focus_row.ca_num) in lookup.index:
        fr = lookup.loc[int(focus_row.ca_num)]
        near = dist[dist.ca_num.eq(int(focus_row.ca_num))].head(3)
        for n in near.itertuples():
            c = centers[centers.name.eq(n.center)].iloc[0]
            inside = n.miles <= threshold
            folium.PolyLine([[fr.c_lat, fr.c_lon], [c.lat, c.lon]], color='#06264A' if inside else '#C84459',
                weight=3, dash_array=None if inside else '6,6',
                tooltip=f'{fr.ca_name} → {n.center}: {n.miles:.1f} mi').add_to(omap)
            mid = [(fr.c_lat + c.lat) / 2, (fr.c_lon + c.lon) / 2]
            folium.Marker(mid, icon=folium.DivIcon(class_name='', icon_size=(160, 22), icon_anchor=(0, 11), html=(
                f'<span style="display:inline-block;font:600 12px Arial;color:#06264A;background:#fff;border:1px solid #06264A;'
                f'border-radius:4px;padding:1px 5px;white-space:nowrap">{n.center}: {n.miles:.1f} mi</span>'))).add_to(omap)
    legend = ''.join(f'<div><span style="display:inline-block;width:12px;height:12px;background:{option_colors[k]};'
                     f'margin-right:6px;border:1px solid #888"></span>{label}</div>'
                     for k, label in [(0, 'None (trauma desert)'), (1, 'One center'), (2, 'Two centers'), (3, 'Three or more')])
    omap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:22px;left:12px;z-index:9999;background:#fff;padding:8px 10px;'
        'border-radius:6px;font:12px Arial;color:#06264A;box-shadow:0 1px 4px rgba(0,0,0,.25)">'
        f'<b>Level I centers within {threshold:g} mi</b>{legend}'
        '<div style="margin-top:4px">Filled navy dot: Stroger</div></div>'))
    st_folium(omap, height=520, use_container_width=True, returned_objects=[],
              key=f'options_map_{year}_{threshold}_{focus_row.ca_num if focus_row is not None else "all"}')
    st.caption('Distances are straight-line miles from the center of each community area to each adult Level I '
               'trauma center (Rush is excluded; it is not Level I). Circles show the chosen distance around each '
               'center. Pick a community in the sidebar to draw lines to its three nearest centers. The map shows '
               'the options; it does not show which hospital patients were actually taken to. Click a hospital marker for details and its official website.')

    # Table: the comparison behind the map.
    table = summary.sort_values(['options', 'shot_total'], ascending=[True, False])[
        ['ca_name', 'shot_total', 'options', 'nearest_three', 'stroger_miles', 'stroger_option']].rename(columns={
        'ca_name': 'Community area', 'shot_total': f'People shot ({year})', 'options': f'Centers within {threshold:g} mi',
        'nearest_three': 'Nearest three centers', 'stroger_miles': 'Miles to Stroger', 'stroger_option': 'Stroger within reach'})
    table['Stroger within reach'] = table['Stroger within reach'].map({True: 'Yes', False: 'No'})
    with st.expander('See every community’s trauma center options'):
        st.dataframe(table.style.format({f'People shot ({year})': '{:,.0f}', 'Miles to Stroger': '{:.1f}'}, na_rep='N/A'),
                     hide_index=True, use_container_width=True)
        st.download_button('Download trauma options (CSV)', table.to_csv(index=False).encode('utf-8'),
                           file_name=f'trauma_options_{year}_{threshold:g}mi.csv', mime='text/csv', key='download_options')
