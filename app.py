import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pytz

from src.etl import WeatherProvider
from src.model import get_predictor
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
    /* Dark mode adjustments if needed, but keeping simple for now */
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Configuración")

# Default: Logroño
default_lat = 42.4667
default_lon = -2.4500

lat = st.sidebar.number_input("Latitud", value=default_lat, format="%.4f")
lon = st.sidebar.number_input("Longitud", value=default_lon, format="%.4f")
capacity_kwp = st.sidebar.number_input("Capacidad Solar (kWp)", value=100.0, step=10.0)

st.sidebar.markdown("---")
st.sidebar.info("AgroVoltaico-Nex v1.0\n\nPowered by Open-Meteo & XGBoost")

# --- Main Logic ---
st.title("🍇 AgroVoltaico-Nex | Powered by Open-Meteo")

@st.cache_data(ttl=3600)
def load_data(latitude, longitude):
    provider = WeatherProvider()
    return provider.get_forecast(latitude, longitude)

try:
    with st.spinner("Conectando con satélites meteorológicos..."):
        # 1. Get Weather Data
        forecast_df = load_data(lat, lon)
        
        # 2. Predict Solar Generation
        predictor = get_predictor()
        # Scale prediction by capacity (model trained on ~1MW/5000m2, let's assume model output is per MW capacity unit roughly, 
        # but actually the model outputs MW directly for a specific synthetic plant. 
        # To make it dynamic, we should normalize. 
        # Synthetic plant: 5000m2 * 0.2 eff ~= 1MW peak (very rough). 
        # Let's assume model output is "MW per 1MW installed" (Normalized) for simplicity, 
        # OR just scale the synthetic output.
        # The synthetic data generation used: power_mw = ... 
        # Let's treat the model output as "Generation for a reference 1MW plant".
        # So user input kWp -> MW: capacity_kwp / 1000.
        
        # However, the synthetic data was generated with specific logic. 
        # Let's assume the model learns the relationship for THAT specific plant.
        # To make it usable for the user input, we'll scale it.
        # Reference plant peak in synthetic data is approx 1.0 MW (radiation 1000 * 5000 * 0.2 / 1e6 = 1.0).
        # So we can multiply prediction by (user_capacity_kwp / 1000).
        
        final_df = predictor.predict(forecast_df)
        
        # Scale to user capacity
        reference_capacity_mw = 1.0
        user_capacity_mw = capacity_kwp / 1000.0
        scaling_factor = user_capacity_mw / reference_capacity_mw
        
        final_df['user_solar_mw'] = final_df['pred_solar_mw'] * scaling_factor
        final_df['user_solar_mwh'] = final_df['user_solar_mw']  # Hourly data, so MW = MWh per hour
        
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
            risk_color = {
                'CRÍTICO': 'inverse', # Streamlit doesn't allow custom colors in metric easily, but we can use delta
                'ALTO': 'off',
                'MEDIO': 'normal',
                'BAJO': 'normal'
            }
            st.metric("Riesgo Helada (3 días)", frost_risk, delta_color="inverse" if frost_risk in ['CRÍTICO', 'ALTO'] else "normal")
            if frost_risk in ['CRÍTICO', 'ALTO']:
                st.error(f"⚠️ ALERTA: Riesgo de helada {frost_risk} detectado en los próximos 3 días.")

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
                xaxis_title="Hora",
                yaxis_title="Temperatura (°C)",
                hovermode="x unified",
                height=400
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
                xaxis_title="Hora",
                yaxis_title="Potencia (MW)",
                yaxis2=dict(
                    title="Radiación (W/m²)",
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                hovermode="x unified",
                height=400,
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_solar, use_container_width=True)

        # Row 3: Data Table
        st.markdown("### 📊 Datos Detallados")
        
        display_cols = ['date', 'temperature_2m', 'shortwave_radiation', 'precipitation', 'wind_speed_10m', 'user_solar_mw']
        column_config = {
            'date': 'Fecha/Hora',
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
            "agrovoltaico_forecast.csv",
            "text/csv",
            key='download-csv'
        )

except Exception as e:
    st.error(f"Ocurrió un error al cargar los datos: {e}")
    # st.exception(e) # Uncomment for debugging
