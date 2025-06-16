import streamlit as st
st.set_page_config(layout="wide")

# --- Streamlit App ---
import joblib
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import geopandas as gpd
import requests
import json

# Load the model and encoder
rf_model = joblib.load('model/rf_multi_model.pkl')
le_party = joblib.load('model/le_party.pkl')

# Load your preprocessed data
# df should contain all years and results, df_2024 should contain only 2024 prediction input
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_data.csv')
    df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)
    df_2024 = pd.read_csv('data/df_2024.csv')
    df_2024['county_fips'] = df_2024['county_fips'].astype(str).str.zfill(5)
    df_2024_encoded = pd.get_dummies(df_2024, columns=['state', 'county_name'], drop_first=True)
    return df, df_2024, df_2024_encoded

def load_geojson():
    with open("data/modified_counties_with_alaska.json", "r") as f:
        geojson = json.load(f)
    return geojson

counties_geo = load_geojson()

df, df_2024, df_2024_encoded = load_data()

# Predict for 2024
X_2024_encoded = df_2024_encoded.reindex(columns=rf_model.feature_names_in_, fill_value=0)
y_2024_pred = rf_model.predict(X_2024_encoded)
df_2024 = df_2024.copy()
df_2024['predicted_party'] = le_party.inverse_transform(y_2024_pred[:, 0])
df_2024['predicted_flipped'] = y_2024_pred[:, 1]

# --- Streamlit UI ---
st.title("Mapping Political Volatility: Swing Counties and Population Trends (2000-2020 + 2024)")

tabs = st.tabs(["🗺️ Time Slider Map", "🔮 2024 Predictions"])

# --- Tab 1: Slider Map ---
with tabs[0]:
    st.header("Party Control by County (2000–2020)")

    fig = px.choropleth(
        df,
        geojson=counties_geo,
        locations='county_fips',
        color='party',
        color_discrete_map={"DEMOCRAT": "blue", "REPUBLICAN": "red"},
        scope="usa",
        hover_name='county_name',
        labels={'party': 'Winning Party'},
        animation_frame='year',        # <- this adds the slider to the plot
        animation_group='county_fips'  # to track counties across frames
    )
    fig.update_layout(
        transition={'duration': 500},
        geo=dict(
        scope='usa',
        projection_scale=1,  # Adjusts zoom
        center={"lat": 37.0902, "lon": -95.7129},  # Centers over the US
        showcoastlines=False,
        showframe=False,
        ),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=600,
    )

    # Disable user interactions but keep hover
    config = {
        'scrollZoom': False,
        'displayModeBar': False,
        'editable': False,
        'doubleClick': False,
        'showAxisDragHandles': False,
        'showTips': True,           # keeps hover
        'staticPlot': False         # must be False to allow hover
    }
    #fig.update_layout(transition={'duration': 500})  

    st.plotly_chart(fig, config=config, use_container_width=True)

    st.markdown("""
    <div style="padding:10px; border-left: 4px solid #007ACC; margin-top:15px;">
    <strong>Note:</strong> Alaska’s election results are originally reported by congressional districts rather than counties. To provide a consistent county-level map, all Alaska districts have been combined into a single statewide unit. Thus, the results shown for Alaska represent the entire state, not individual counties or districts.
    </div>
    """, unsafe_allow_html=True)

# --- Tab 2: 2024 Prediction ---
with tabs[1]:
    st.header("Predicted Party Outcome for 2024")

    # 1. Map predicted party to sign
    party_sign_map = {"DEMOCRAT": -1, "REPUBLICAN": 1}
    df_2024['party_sign'] = df_2024['predicted_party'].map(party_sign_map)

    # 2. Compute (1 - flip_probability) so that:
    #    - higher flip probability = closer to party change
    #    - we want strong flip confidence to map to dark color
    df_2024['signed_flip_score'] = (1 - df_2024['flip_probability']) * df_2024['party_sign']

    # 3. Normalize signed score from [-1, 1] to [0, 1] for Plotly
    df_2024['color_code'] = (df_2024['signed_flip_score'] + 1) / 2

    # 4. Colorscale: blue to white to red (Dem to uncertain to Rep)
    colorscale = [
        [0.0, 'rgb(0, 0, 255)'],       # Strong Dem 
        [0.5, 'rgb(255, 255, 255)'],   # Likely to Flip
        [1.0, 'rgb(255, 0, 0)']        # Strong Rep 
    ]

    # 5. Hover text with percentage probability
    df_2024['hover_flip_prob'] = (df_2024['flip_probability'] * 100).round(1).astype(str) + '%'

    fig_2024 = go.Figure(go.Choropleth(
        geojson=counties_geo,
        locations=df_2024['county_fips'],
        z=df_2024['color_code'],
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        marker_line_width=0,
        showscale=True,
        colorbar=dict(
            title="Flip Confidence & Direction",
            titleside="top",
            tickvals=[0.0, 0.5, 1.0],
            ticktext=["Strong Dem", "Likely to Flip", "Strong Rep"],
            len=0.3,
            x=0.5,
            y=-0.1,
            xanchor='center',
            yanchor='top',
            orientation='h',
            ticks="outside",
            tickfont=dict(size=10)
        ),
        hovertext=df_2024['county_name'] +
                  "<br>Predicted Party: " + df_2024['predicted_party'] +
                  "<br>Flip Probability: " + df_2024['hover_flip_prob'],
        hoverinfo='text'
    ))
    fig_2024.update_layout(
        geo=dict(
        scope='usa',
        projection_scale=1,  # Adjusts zoom
        center={"lat": 37.0902, "lon": -95.7129},  # Centers over the US
        showcoastlines=False,
        showframe=False,
        ),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=600,
    )

    # Disable user interactions but keep hover
    config_2024 = {
        'scrollZoom': False,
        'displayModeBar': False,
        'editable': False,
        'doubleClick': False,
        'showAxisDragHandles': False,
        'showTips': True,           # keeps hover
        'staticPlot': False         # must be False to allow hover
    }

    st.plotly_chart(fig_2024, use_container_width=True, config=config_2024, key="flip_prediction")

    st.markdown("""
    <div style="padding:10px; border-left: 4px solid #007ACC; margin-top:15px;">
    <strong>Note:</strong> The 2024 predictions for Alaska are made at the statewide level because the original data is based on congressional districts, not counties. Therefore, the prediction shown for Alaska reflects the entire state rather than separate districts or counties.
    </div>
    """, unsafe_allow_html=True)