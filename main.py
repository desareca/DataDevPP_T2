import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pydantic.types import conlist, conint
from typing import List, Dict, Any

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Este archivo implementa una API para predecir la temperatura en la estación meteorológica de Quinta Normal.
# Utiliza FastAPI y un modelo previamente entrenado y guardado con joblib.


# Cargar el modelo entrenado desde archivo
model = joblib.load("./models/model_multiple.joblib")


# Inicializar la aplicación FastAPI con metadatos
app = FastAPI(
    title="Temperatura Estación Quinta Normal",
    description="API para predecir temperatura de la estación meteorológica de Quinta Normal.",
    version="1.0.0")

## ----- Schemas -----
# Esquemas de datos para las solicitudes de la API

class Temperature(BaseModel):
    # Temperaturas de las horas previas requeridas para la predicción puntual
    Ts_Valor_1h: float = Field(...,
                               description="Temperatura hace 1 hora",
                               example=1.2)
    Ts_Valor_2h: float = Field(...,
                               description="Temperatura hace 2 horas",
                               example=1.2)
    Ts_Valor_3h: float = Field(...,
                               description="Temperatura hace 3 horas",
                               example=1.2)
    Ts_Valor_24h: float = Field(...,
                               description="Temperatura hace 24 horas",
                               example=1.2)
    Ts_Valor_25h: float = Field(...,
                               description="Temperatura hace 25 horas",
                               example=1.2)

class ModelPerformance(BaseModel):
    # Lista de temperaturas para evaluar el desempeño del modelo (mínimo 26 valores)
    data: conlist(float, min_length=26, max_length=24*30) = Field(..., # type: ignore
        description="Lista de temperaturas (min 26)",
        example=[1.2]*26)

class TemperatureN(BaseModel):
    # Lista de temperaturas para predicción de n horas (exactamente 25 valores)
    data: conlist(float, min_length=25, max_length=25) = Field(..., # type: ignore
        description="Lista de temperaturas (min 25)",
        example=[1.2]*25)
    hours: conint(gt=0) = Field(..., # type: ignore
        description="N° de horas a predecir",
        example=1)



# ----- Helpers -----
# Función auxiliar para realizar la predicción usando el modelo cargado
def _predict_vector(x_vec: np.ndarray) -> float:
    """
    Recibe un vector de características con shape (5,) en el orden correcto y devuelve la predicción como float nativo.
    """
    # El modelo espera un array 2D, por eso se hace reshape
    pred_value = model.predict(x_vec.reshape(1, -1))[0]
    # Convertir a tipo float estándar de Python para serialización
    return float(pred_value)


# ----- Endpoints -----
# Definición de los endpoints de la API

@app.get("/", response_class=HTMLResponse)
def home() -> str:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post(
    "/predict",
    summary="Realiza una predicción puntual",
    description="Recibe temperaturas de las horas anteriores y predice la temperatura de la próxima hora (hora 0)."
)
def prediction(temp: Temperature) -> Dict[str, Any]:
    """
    Recibe las temperaturas de las últimas horas y retorna la predicción para la próxima hora.
    """
    # Construir el vector de características en el mismo orden usado para entrenar el modelo
    x = np.array([
        temp.Ts_Valor_1h,
        temp.Ts_Valor_2h,
        temp.Ts_Valor_3h,
        temp.Ts_Valor_24h,
        temp.Ts_Valor_25h,
    ], dtype=float)
    yhat = _predict_vector(x)
    return {"predicted_temperature": yhat}


@app.post(
    "/predict_n",
    summary="Realiza una predicción de n horas",
    description="Recibe temperaturas de las últimas 25 horas anteriores y predice la temperatura de las próximas n horas."
)
def prediction_n(payload: TemperatureN) -> Dict[str, List[float]]:
    """
    Recibe una lista de temperaturas y predice la temperatura para las próximas n horas de forma secuencial.
    """
    data_ = payload.data
    n_hours = payload.hours
    preds: List[float] = []

    for i in range(n_hours):
        # Construir el vector de entrada para el modelo usando los últimos valores
        x = np.array([
            data_[24],  # 1h
            data_[23],  # 2h
            data_[22],  # 3h
            data_[1],   # 24h
            data_[0],   # 25h
        ], dtype=float)

        y_pred = _predict_vector(x)
        preds.append(y_pred)

        # Actualizar la lista de temperaturas para la siguiente predicción
        data_ = data_[1:]
        data_.append(y_pred)

    return {"predicted_temperature": preds}



@app.post(
    "/model_performance",
    summary="Evalua el comportamiento del modelo",
    description="Recibe una lista de temperaturas (mínimo 26 hrs), predice secuencialmente la próxima hora y entrega estadísticas útiles para evaluar rendimiento"
)
def model_performance(payload: ModelPerformance) -> Dict[str, List[float]]:
    """
    Evalúa el desempeño del modelo realizando predicciones secuenciales y calculando métricas como RMSE, media y desviación estándar.
    """
    data_ = payload.data
    n = len(data_)

    preds: List[float] = []
    reals: List[float] = []

    # Para predecir el valor en t = i+25, los lags son:
    # 1h:  data[i+24]
    # 2h:  data[i+23]
    # 3h:  data[i+22]
    # 24h: data[i+1]
    # 25h: data[i]
    # target real: data[i+25]
    for i in range(n - 25):
        x = np.array([
            data_[i + 24],  # 1h
            data_[i + 23],  # 2h
            data_[i + 22],  # 3h
            data_[i + 1],   # 24h
            data_[i + 0],   # 25h
        ], dtype=float)

        y_true = float(data_[i + 25])
        y_pred = _predict_vector(x)

        preds.append(y_pred)
        reals.append(y_true)

    # Calcular RMSE
    rmse = [float(np.sqrt(np.mean((np.array(reals) - np.array(preds)) ** 2)))]

    # Calcular medias
    mean_true = [float(np.mean(reals))]
    mean_pred = [float(np.mean(preds))]

    # Calcular desviaciones estándar
    std_true = [float(np.std(reals, ddof=0))]
    std_pred = [float(np.std(preds, ddof=0))]

    return {
        "predicted_temperature": preds,
        "real_temperature": reals,
        "rmse": rmse,
        "mean_true": mean_true,
        "mean_pred": mean_pred,
        "std_true": std_true,
        "std_pred": std_pred
    }