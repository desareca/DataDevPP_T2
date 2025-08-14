import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pydantic.types import conlist, conint
from typing import List, Dict, Any

# Cargar modelo
model = joblib.load("./models/model_multiple.joblib")

app = FastAPI(
    title="Temperatura Estación Quinta Normal",
    description="API para predecir temperatura de la estación meteorológica de Quinta Normal.",
    version="1.0.0")

# ----- Schemas -----
class Temperature(BaseModel):
    Ts_Valor_1h: float = Field(...,
                               description="Temperatura hace 1 hora",
                               example=1.2
                               )
    Ts_Valor_2h: float = Field(...,
                               description="Temperatura hace 2 horas",
                               example=1.2
                               )
    Ts_Valor_3h: float = Field(...,
                               description="Temperatura hace 3 horas",
                               example=1.2
                               )
    Ts_Valor_24h: float = Field(...,
                               description="Temperatura hace 24 horas",
                               example=1.2
                               )
    Ts_Valor_25h: float = Field(...,
                               description="Temperatura hace 25 horas",
                               example=1.2
                               )

class ModelPerformance(BaseModel):
    # Al menos 26 puntos para poder formar (25,24,3,2,1) -> target en +25
    data: conlist(float, min_length=26, max_length=24*30) = Field(...,
                                                                  description="Lista de temperaturas (min 26)",
                                                                  example=[1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 
                                                                           1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 
                                                                           1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
                                                                  )

class TemperatureN(BaseModel):
    # Al menos 26 puntos para poder formar (25,24,3,2,1)
    data: conlist(float, min_length=25, max_length=25) = Field(...,
                                                               description="Lista de temperaturas (min 25)",
                                                               example=[1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 
                                                                        1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 
                                                                        1.2, 1.2, 1.2, 1.2, 1.2]
                                                                        )
    hours: conint(gt=0) = Field(...,
                                description="N° de horas a predecir",
                                example=1
                                )


# ----- Helpers -----

def _predict_vector(x_vec: np.ndarray) -> float:
    """Recibe vector shape (5,) en el ORDEN correcto y devuelve float nativo."""
    # Usamos array 2D para predict
    pred_value = model.predict(x_vec.reshape(1, -1))[0]
    # Garantizar tipo Python serializable
    return float(pred_value)

# ----- Endpoints -----

@app.get("/")
def home() -> str:
    return "¡Felicitaciones! Tu API está funcionando según lo esperado. Anda ahora a http://localhost:8000/docs."

@app.post(
        "/predict",
        summary="Realiza una predicción puntual",
        description="Recibe temperaturas de las horas anteriores y predice la temperatura de la próxima hora (hora 0)."
        )
def prediction(temp: Temperature) -> Dict[str, Any]:
    # Construir feature vector en el MISMO orden usado para entrenar
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
        description="Recibe temperaturas de las últimas 25 horas anteriores y predece la temperatura de la próximas n horas."
        )
def prediction_n(payload: TemperatureN) -> Dict[str, List[float]]:
    data_ = payload.data
    n_hours = payload.hours
    preds: List[float] = []

    for i in range(n_hours):
        x = np.array([
            data_[24],  # 1h
            data_[23],  # 2h
            data_[22],  # 3h
            data_[1],   # 24h
            data_[0],   # 25h
        ], dtype=float)

        y_pred = _predict_vector(x)
        preds.append(y_pred)

        data_ = data_[1:]
        data_.append(y_pred)

    return {"predicted_temperature": preds}


@app.post(
        "/model_performance",
        summary="Evalua el comportamiento del modelo",
        description="Recibe una lista de temperaturas (minimo 26 hrs), predice secuencialmente la proxima hora y entrega estadisticas íutiles para evaluar rendimiento"
        )
def model_performance(payload: ModelPerformance) -> Dict[str, List[float]]:
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

    rmse = [float(np.sqrt(np.mean((np.array(reals) - np.array(preds)) ** 2)))]

    # Medias
    mean_true = [float(np.mean(reals))]
    mean_pred = [float(np.mean(preds))]

    # Desviaciones estándar
    std_true = [float(np.std(reals, ddof=0))]
    std_pred = [float(np.std(preds, ddof=0))]

    return {"predicted_temperature": preds, "real_temperature": reals,
            "rmse": rmse, "mean_true": mean_true, "mean_pred": mean_pred, "std_true": std_true, "std_pred": std_pred}