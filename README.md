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

## Arquitectura de Machine Learning

AgroVoltaico-Nex utiliza un enfoque híbrido de inteligencia artificial:

1.  **Predicción Meteorológica (Open-Meteo)**:
    -   Utiliza modelos numéricos globales (GFS, IFS) refinados con técnicas de **Machine Learning** para mejorar la resolución espacial a 1-11 km.
    -   Esto permite obtener datos hiperlocales para el viñedo específico.

2.  **Predicción de Energía Solar (XGBoost)**:
    -   Hemos implementado un modelo de **Gradient Boosting (XGBoost)**.
    -   **Entrenamiento**: El modelo se entrena con datos históricos (sintéticos en esta demo) aprendiendo la relación no lineal entre:
        -   Radiación de onda corta (W/m²).
        -   Temperatura ambiente (afecta la eficiencia de los paneles).
        -   Variables temporales cíclicas (Seno/Coseno de la hora) para capturar el ciclo diurno.
    -   **Inferencia**: El modelo toma el pronóstico meteorológico de 14 días y predice la curva de generación MW hora a hora.

3.  **Índices Agroclimáticos**:
    -   Algoritmos deterministas basados en umbrales biológicos (ej. temperatura < -1°C para helada crítica) aplicados sobre las predicciones de ML.
