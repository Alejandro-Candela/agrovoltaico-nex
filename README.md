# AgroVoltaico-Nex

Sistema de Soporte a la Decisión (DSS) para la gestión agrovoltaica en viñedos.

## Descripción
Esta aplicación permite a los gestores de viñedos con instalaciones solares:
1.  **Predecir la generación solar** para los próximos 3 días.
2.  **Detectar riesgos de helada** para proteger los cultivos.

## Instalación

1.  Clonar el repositorio o descargar los archivos.
2.  Crear un entorno virtual (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```
3.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución

Para iniciar la aplicación:

```bash
streamlit run app.py
```

## Estructura del Proyecto

-   `app.py`: Punto de entrada de la aplicación Streamlit.
-   `src/etl.py`: Módulo de extracción y transformación de datos (Open-Meteo).
-   `src/model.py`: Modelo de predicción solar (XGBoost).
-   `src/indices.py`: Cálculos de índices agroclimáticos (Riesgo de helada, GDD).
-   `src/utils.py`: Funciones de utilidad.

## Notas Técnicas
-   La primera ejecución entrenará un modelo XGBoost con datos sintéticos y lo guardará en `model.json`.
-   Los datos meteorológicos se obtienen de la API gratuita de Open-Meteo.
