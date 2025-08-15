# DataDevPP_T2

# API de Predicción de Temperatura - Estación Quinta Normal

Este proyecto implementa una API basada en FastAPI para predecir la temperatura en la estación meteorológica de Quinta Normal, utilizando un modelo de machine learning previamente entrenado y almacenado en formato joblib.

## Contexto académico y objetivo

Este proyecto corresponde a la **Tarea 2** de la asignatura "Desarrollo de Productos y Proyectos de Datos" del Magíster en Data Science UDD. El objetivo principal es diseñar, implementar y desplegar un producto de datos: una API capaz de predecir la temperatura en la estación meteorológica de Quinta Normal, utilizando técnicas de machine learning y buenas prácticas de desarrollo.

## Descripción

La API permite realizar predicciones de temperatura a partir de datos históricos, facilitando la consulta de la temperatura futura en base a las temperaturas registradas en las horas previas. El modelo utilizado fue entrenado con datos reales y considera los valores de temperatura de las últimas 1, 2, 3, 24 y 25 horas para realizar la predicción.

## Despliegue

La API está desplegada en Render y disponible públicamente en:

- https://prediccion-temperatura-estacion-quinta.onrender.com/

Puedes acceder a la documentación interactiva en:

- https://prediccion-temperatura-estacion-quinta.onrender.com/docs

## Funcionalidades principales

- **Predicción puntual**: Entrega la temperatura estimada para la próxima hora, dado un conjunto de temperaturas recientes.
- **Predicción de n horas**: Permite predecir la temperatura para varias horas futuras de manera secuencial.
- **Evaluación de desempeño**: Calcula métricas como RMSE, media y desviación estándar para comparar las predicciones del modelo con los valores reales.

## Uso

1. Instala las dependencias listadas en `requirements.txt`.
2. Ejecuta el servidor FastAPI:
   ```cmd
   uvicorn main:app --reload
   ```
3. Accede a la documentación interactiva en [http://localhost:8000/docs](http://localhost:8000/docs) para probar los endpoints y ver ejemplos de uso.

## Endpoints principales

- `/predict` (POST): Recibe las temperaturas de las últimas horas y retorna la predicción para la próxima hora.
- `/predict_n` (POST): Recibe una lista de temperaturas y predice la temperatura para las próximas n horas.
- `/model_performance` (POST): Evalúa el desempeño del modelo con una secuencia de datos históricos y entrega métricas de calidad.

## Estructura del proyecto

- `main.py`: Código principal de la API.
- `models/model_multiple.joblib`: Modelo de machine learning entrenado.
- `requirements.txt`: Dependencias necesarias.
- `notebooks/`: Notebooks de análisis y desarrollo.

## Requisitos
- Python 3.11+
- FastAPI
- joblib
- numpy
- pydantic

**Nota:** El archivo `requirements.txt` contiene algunas librerías comentadas. Estas librerías (`matplotlib`, `seaborn`, `pandas`) no son necesarias para desplegar la API, pero sí son requeridas para ejecutar y analizar los notebooks del proyecto. Si deseas trabajar con los notebooks, asegúrate de instalar también estas dependencias.

## Notebook de pruebas

En el directorio `notebooks/` se incluye el notebook `client.ipynb` con ejemplos de pruebas a la API desplegada en Render. Este notebook muestra cómo consultar los principales endpoints (`/predict`, `/predict_n`, `/model_performance`) utilizando la librería `requests`, enviar datos de ejemplo y visualizar las respuestas obtenidas desde la API pública.

Esto permite validar el funcionamiento del producto de datos en un entorno real y facilita la integración con otros sistemas o flujos de trabajo.

