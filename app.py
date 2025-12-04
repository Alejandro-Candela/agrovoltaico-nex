import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from src.etl import WeatherProvider
from src.model import get_base_predictor
from src.indices import calculate_frost_risk, calculate_gdd

# --- Page Config ---
st.set_page_config(
    page_title="AgroVoltaico-Nex",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styles ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: transparent !important;
    }
    h3 {
        font-size: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Configuración")

# Initialize session state for location
if 'lat' not in st.session_state:
    st.session_state.lat = 42.4667
if 'lon' not in st.session_state:
    st.session_state.lon = -2.4500

# 1. Address Search
st.sidebar.subheader("📍 Ubicación")
address = st.sidebar.text_input("Buscar dirección:", placeholder="Ej. Logroño, La Rioja")

if address:
    try:
        geolocator = Nominatim(user_agent="agrovoltaico-nex")
        location = geolocator.geocode(address)
        if location:
            st.session_state.lat = location.latitude
            st.session_state.lon = location.longitude
            st.sidebar.success(f"📍 {location.address}")
        else:
            st.sidebar.error("Dirección no encontrada.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# 2. Manual Input
with st.sidebar.expander("Coordenadas Manuales", expanded=False):
    col_lat, col_lon = st.columns(2)
    with col_lat:
        new_lat = st.number_input("Lat", value=st.session_state.lat, format="%.4f", key="manual_lat")
    with col_lon:
        new_lon = st.number_input("Lon", value=st.session_state.lon, format="%.4f", key="manual_lon")
    
    # Update state if manual input changes (Streamlit handles this via key, but we need to sync)
    if new_lat != st.session_state.lat:
        st.session_state.lat = new_lat
    if new_lon != st.session_state.lon:
        st.session_state.lon = new_lon

# 3. Map Widget
with st.sidebar:
    st.markdown("Seleccionar en mapa:")
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=9)
    folium.Marker(
        [st.session_state.lat, st.session_state.lon], 
        popup="Ubicación", 
        tooltip="Ubicación Actual"
    ).add_to(m)

    # Capture map clicks
    st_data = st_folium(m, width=280, height=250, key="map_widget")

if st_data and st_data['last_clicked']:
    clicked_lat = st_data['last_clicked']['lat']
    clicked_lng = st_data['last_clicked']['lng']
    # Only update if different to avoid loops
    if abs(clicked_lat - st.session_state.lat) > 0.0001 or abs(clicked_lng - st.session_state.lon) > 0.0001:
        st.session_state.lat = clicked_lat
        st.session_state.lon = clicked_lng
        st.rerun()

capacity_kwp = st.sidebar.number_input("Capacidad Solar (kWp)", value=100.0, step=10.0)

st.sidebar.markdown("---")
st.sidebar.info("AgroVoltaico-Nex v1.2\n\nPowered by Open-Meteo & XGBoost")

# --- Main Logic ---
st.markdown("<h3>🍇 AgroVoltaico-Nex | Powered by Open-Meteo</h3>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_weather_data(latitude, longitude):
    provider = WeatherProvider()
    return provider.get_forecast(latitude, longitude)

@st.cache_resource(show_spinner="Entrenando modelo para tu ubicación...")
def get_trained_predictor(latitude, longitude):
    predictor = get_base_predictor()
    # Force training for this specific location
    predictor.load_model(latitude, longitude)
    return predictor

try:
    with st.spinner("Analizando datos satelitales y ejecutando modelos..."):
        # 1. Get Weather Data
        forecast_df = load_weather_data(st.session_state.lat, st.session_state.lon)
        
        # 2. Predict Solar Generation (Train on fly if needed)
        predictor = get_trained_predictor(st.session_state.lat, st.session_state.lon)
        final_df = predictor.predict(forecast_df)
        
        # Scale to user capacity
        reference_capacity_mw = 1.0 # Synthetic training data is normalized to ~1MW
        user_capacity_mw = capacity_kwp / 1000.0
        scaling_factor = user_capacity_mw / reference_capacity_mw
        
        final_df['user_solar_mw'] = final_df['pred_solar_mw'] * scaling_factor
        final_df['user_solar_mwh'] = final_df['user_solar_mw']
        
        # 3. Calculate Indices
        frost_risk = calculate_frost_risk(final_df)
        gdd = calculate_gdd(final_df)
        
        # --- Dashboard Layout ---
        
        # Row 1: KPIs
        col1, col2, col3 = st.columns(3)
        
        current_temp = final_df['temperature_2m'].iloc[0]
        total_energy_today = final_df[final_df['date'].dt.date == datetime.now().date()]['user_solar_mwh'].sum()
        
        with col1:
            st.metric("Temperatura Actual", f"{current_temp:.1f} °C", delta=None)
        
        with col2:
            st.metric("Generación Est. Hoy", f"{total_energy_today:.2f} MWh", delta=None)
            
        with col3:
            st.metric("Riesgo Helada (14 días)", frost_risk, delta_color="inverse" if frost_risk in ['CRÍTICO', 'ALTO'] else "normal")
            if frost_risk in ['CRÍTICO', 'ALTO']:
                st.error(f"⚠️ ALERTA: Riesgo de helada {frost_risk} detectado en los próximos 14 días.")

        st.markdown("---")

        # Row 2: Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🌡️ Temperatura y Riesgo de Helada")
            
            fig_temp = go.Figure()
            
            # Add temperature line
            fig_temp.add_trace(go.Scatter(
                x=final_df['date'], 
                y=final_df['temperature_2m'],
                mode='lines',
                name='Temperatura',
                line=dict(color='orange', width=2)
            ))
            
            # Calculate Daily Min/Max
            final_df['day_date'] = final_df['date'].dt.date
            daily_stats = final_df.groupby('day_date')['temperature_2m'].agg(['min', 'max']).reset_index()
            
            # Create stepped lines for Min/Max
            # We map the daily value back to the hourly timestamps for plotting
            final_df = final_df.merge(daily_stats, on='day_date', suffixes=('', '_daily'))
            
            fig_temp.add_trace(go.Scatter(
                x=final_df['date'],
                y=final_df['max'],
                mode='lines',
                name='Max Diaria',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            ))
            
            fig_temp.add_trace(go.Scatter(
                x=final_df['date'],
                y=final_df['min'],
                mode='lines',
                name='Min Diaria',
                line=dict(color='blue', width=1, dash='dash'),
                opacity=0.7
            ))

            # Add frost zone (below 0)
            fig_temp.add_hrect(
                y0=-10, y1=0, 
                fillcolor="blue", opacity=0.1, 
                layer="below", line_width=0,
                annotation_text="Zona de Helada", annotation_position="bottom right"
            )
            
            # Add critical frost zone (below -1)
            fig_temp.add_hrect(
                y0=-10, y1=-1, 
                fillcolor="red", opacity=0.1, 
                layer="below", line_width=0
            )
            
            fig_temp.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Temperatura (°C)",
                hovermode="x unified",
                height=400,
                xaxis=dict(
                    tickformat="%Y-%m-%d %H:%M:%S",
                    tickangle=-45
                ),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with c2:
            st.subheader("☀️ Generación Solar Prevista")
            
            fig_solar = go.Figure()
            
            fig_solar.add_trace(go.Bar(
                x=final_df['date'],
                y=final_df['user_solar_mw'],
                name='Generación (MW)',
                marker_color='#f4d03f'
            ))
            
            # Add line for radiation (secondary y)
            fig_solar.add_trace(go.Scatter(
                x=final_df['date'],
                y=final_df['shortwave_radiation'],
                name='Radiación (W/m²)',
                yaxis='y2',
                line=dict(color='red', width=1, dash='dot'),
                opacity=0.5
            ))
            
            fig_solar.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Potencia (MW)",
                yaxis2=dict(
                    title="Radiación (W/m²)",
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                hovermode="x unified",
                height=400,
                legend=dict(orientation="h", y=1.1),
                xaxis=dict(
                    tickformat="%Y-%m-%d %H:%M:%S",
                    tickangle=-45
                )
            )
            st.plotly_chart(fig_solar, use_container_width=True)

        # Row 3: Data Table
        st.markdown("### 📊 Datos Detallados (14 Días)")
        
        display_cols = ['date', 'temperature_2m', 'shortwave_radiation', 'precipitation', 'wind_speed_10m', 'user_solar_mw']
        column_config = {
            'date': st.column_config.DatetimeColumn('Fecha/Hora', format="YYYY-MM-DD HH:mm:ss"),
            'temperature_2m': st.column_config.NumberColumn('Temp (°C)', format="%.1f"),
            'shortwave_radiation': st.column_config.NumberColumn('Radiación (W/m²)', format="%.0f"),
            'precipitation': st.column_config.NumberColumn('Precipitación (mm)', format="%.1f"),
            'wind_speed_10m': st.column_config.NumberColumn('Viento (km/h)', format="%.1f"),
            'user_solar_mw': st.column_config.NumberColumn('Generación (MW)', format="%.3f")
        }
        
        st.dataframe(
            final_df[display_cols], 
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
        
        csv = final_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar CSV",
            csv,
            "agrovoltaico_forecast_14days.csv",
            "text/csv",
            key='download-csv'
        )

except Exception as e:
    st.error(f"Ocurrió un error al cargar los datos: {e}")
    # st.exception(e) # Uncomment for debugging
