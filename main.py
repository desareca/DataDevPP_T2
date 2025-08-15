
import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field, field_validator
from pydantic.types import conlist, conint
from typing import List, Dict, Any
import os
import logging

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Este archivo implementa una API para predecir la temperatura en la estación meteorológica de Quinta Normal.
# Utiliza FastAPI y un modelo previamente entrenado y guardado con joblib.

# ===========================================================================================================================================
# ======================================================= Configuración de validación =======================================================
# ===========================================================================================================================================

MODEL_PATH = "./models/model_multiple.joblib"
TEMPERATURE_MIN = -20.0  # °C
TEMPERATURE_MAX = 60.0   # °C
MAX_FORECAST_HOURS = 168 # Hasta 7 días
LEN_FORECAST_DATA = 25 
MAX_PERFORMANCE_HOURS = 24*30 # Hasta 30 días

def _spanish_message(default_en: str) -> str:
    """
    Traducción mínima para errores comunes de Pydantic/FastAPI.
    Si no se reconoce, devuelve un mensaje genérico en español.
    """
    mapping = {
        "Input should be a valid list": "La entrada debe ser una lista válida.",
        "Input should be a valid number": "El valor debe ser numérico.",
        "Input should be a valid integer": "El valor debe ser un entero válido.",
        "Field required": "Campo obligatorio.",
        "List should have at least": "La lista debe tener al menos la cantidad mínima de elementos indicada.",
        "List should have at most": "La lista debe tener como máximo la cantidad de elementos indicada.",
        "Input should be greater than 0": "El valor debe ser mayor que 0.",
        "JSON decode error": "Error en decodificar el JSON"
    }
    for k, v in mapping.items():
        if k in default_en:
            return v
    return f"Dato inválido: {default_en}"

# ===========================================================================================================================================
# ============================================================= Cargar modelo ===============================================================
# ===========================================================================================================================================
model = None
model_disponible = False

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    model_disponible = True

# ===========================================================================================================================================
# ================================================================ FastAPI ==================================================================
# ===========================================================================================================================================
app = FastAPI(
    title="Temperatura Estación Quinta Normal",
    description="API para predecir temperatura de la estación meteorológica de Quinta Normal.",
    version="1.0.0"
)
logger.info("API FastAPI inicializada.")

# ===========================================================================================================================================
# ========================================================== Validación de Errores ==========================================================
# ===========================================================================================================================================
# --- Manejador de errores de validación (422) en español ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errores = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "Error de validación")
        errores.append({"campo": loc, "mensaje": _spanish_message(msg)})
    return JSONResponse(
        status_code=422,
        content={"detalle": "Error de validación de entrada.", "errores": errores}
    )

# --- Manejador de errores HTTP genéricos en español (404/405/400/401/403/...) ---
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    codigos = {
        404: "Ruta no encontrada.",
        405: "Método HTTP no permitido para esta ruta.",
        400: "Solicitud incorrecta.",
        401: "No autorizado.",
        403: "Prohibido acceder a este recurso.",
        429: "Demasiadas solicitudes. Inténtalo más tarde.",
        503: "Servicio no disponible.",
        599: "El modelo no está disponible para realizar predicciones."
    }
    mensaje = codigos.get(exc.status_code, "Error HTTP.")
    if exc.status_code==599:
        return JSONResponse(
            status_code=503,
            content={"detalle": mensaje}
        )
    else:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detalle": mensaje}
        )

# --- Fallback para cualquier excepción no controlada (500) ---
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detalle": "Error interno del servidor."}
    )

# ===========================================================================================================================================
# ================================================================ Schemas ==================================================================
# ===========================================================================================================================================
# --- Valida datos a nivel general
class _TemperatureValidatorsMixin(BaseModel):
    @staticmethod
    def _check_temp(value: float, nombre: str) -> float:
        if value is None:
            raise ValueError(f"{nombre}: es obligatorio.")
        if isinstance(value, (float, int)):
            v = float(value)
            if not np.isfinite(v):
                raise ValueError(f"{nombre}: no puede ser NaN o infinito.")
            if v < TEMPERATURE_MIN or v > TEMPERATURE_MAX:
                raise ValueError(
                    f"{nombre}: fuera de rango permitido [{TEMPERATURE_MIN}, {TEMPERATURE_MAX}] °C."
                )
            return v
        raise ValueError(f"{nombre}: debe ser numérico.")

# --- Temperaturas de las horas previas requeridas para la predicción puntual
class Temperature(_TemperatureValidatorsMixin):
    Ts_Valor_1h: float = Field(..., description="Temperatura hace 1 hora", example=1.2)
    Ts_Valor_2h: float = Field(..., description="Temperatura hace 2 horas", example=1.2)
    Ts_Valor_3h: float = Field(..., description="Temperatura hace 3 horas", example=1.2)
    Ts_Valor_24h: float = Field(..., description="Temperatura hace 24 horas", example=1.2)
    Ts_Valor_25h: float = Field(..., description="Temperatura hace 25 horas", example=1.2)

    @field_validator("Ts_Valor_1h", "Ts_Valor_2h", "Ts_Valor_3h", "Ts_Valor_24h", "Ts_Valor_25h")
    @classmethod
    def validar_rango_temperatura(cls, v, info):
        return cls._check_temp(v, info.field_name)

# --- Lista de temperaturas para evaluar el desempeño del modelo
class ModelPerformance(BaseModel):
    data: conlist(float, min_length=LEN_FORECAST_DATA+1, max_length=MAX_PERFORMANCE_HOURS) = Field(  # type: ignore
        ..., description=F"Lista de temperaturas (min {LEN_FORECAST_DATA+1})", example=[1.2]*(LEN_FORECAST_DATA+1)
    )

    @field_validator("data")
    @classmethod
    def validar_lista_temperaturas(cls, values: List[float]):
        if any(v is None for v in values):
            raise ValueError("La lista 'data' no puede contener valores nulos.")
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("La lista 'data' no puede contener NaN ni infinitos.")
        fuera = (arr < TEMPERATURE_MIN) | (arr > TEMPERATURE_MAX)
        if np.any(fuera):
            idxs = np.where(fuera)[0][:5].tolist()
            raise ValueError(
                f"Hay valores fuera de rango [{TEMPERATURE_MIN}, {TEMPERATURE_MAX}] °C en posiciones {idxs} (máx. se muestran 5)."
            )
        return values

# --- Lista de temperaturas para predicción de n horas
class TemperatureN(BaseModel):
    data: conlist(float, min_length=LEN_FORECAST_DATA, max_length=LEN_FORECAST_DATA) = Field(  # type: ignore
        ..., description=f"Lista de temperaturas de las últimas {LEN_FORECAST_DATA} horas", example=[1.2]*LEN_FORECAST_DATA
    )
    hours: conint(gt=0, le=MAX_FORECAST_HOURS) = Field(  # type: ignore
        ..., description=f"Número de horas a predecir (máx. {MAX_FORECAST_HOURS})", example=1
    )

    @field_validator("data")
    @classmethod
    def validar_lista_temperaturas(cls, values: List[float]):
        if any(v is None for v in values):
            raise ValueError("La lista 'data' no puede contener valores nulos.")
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("La lista 'data' no puede contener NaN ni infinitos.")
        fuera = (arr < TEMPERATURE_MIN) | (arr > TEMPERATURE_MAX)
        if np.any(fuera):
            idxs = np.where(fuera)[0][:5].tolist()
            raise ValueError(
                f"Hay valores fuera de rango [{TEMPERATURE_MIN}, {TEMPERATURE_MAX}] °C en posiciones {idxs} (máx. se muestran 5)."
            )
        return values

# ===========================================================================================================================================
# ================================================================ Helpers ==================================================================
# ===========================================================================================================================================
# --- Función auxiliar que chequea el modelo
def _check_model() -> None:
    if not model_disponible:
        raise StarletteHTTPException(
            status_code=599,
            detail="El modelo no está disponible para realizar predicciones."
        )

# --- Función auxiliar para realizar la predicción usando el modelo cargado
def _predict_vector(x_vec: np.ndarray) -> float:
    """
    Recibe un vector (5,) y devuelve la predicción como float.
    Orden: [1h, 2h, 3h, 24h, 25h]
    """
    _check_model()
    pred_value = model.predict(x_vec.reshape(1, -1))[0]
    return float(pred_value)

# ===========================================================================================================================================
# =============================================================== Endpoints =================================================================
# ===========================================================================================================================================
@app.get("/", response_class=HTMLResponse)
def home() -> str:
    html_path = "static/index.html"
    if not os.path.exists(html_path):
        logger.error(f"Archivo HTML no encontrado: {html_path}")
        return HTMLResponse(content="<h1>Archivo HTML no encontrado.</h1>", status_code=404)
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            logger.info(f"Archivo HTML servido correctamente: {html_path}")
            return f.read()
    except Exception as e:
        logger.error(f"Error al leer el archivo HTML: {e}")
        return HTMLResponse(content="<h1>Error al leer el archivo HTML.</h1>", status_code=500)

@app.post(
    "/predict",
    summary="Realiza una predicción puntual",
    description="Recibe temperaturas de las horas anteriores y predice la temperatura de la próxima hora (hora 0)."
)
def prediction(temp: Temperature) -> Dict[str, Any]:
    x = np.array([
        temp.Ts_Valor_1h,
        temp.Ts_Valor_2h,
        temp.Ts_Valor_3h,
        temp.Ts_Valor_24h,
        temp.Ts_Valor_25h,
    ], dtype=float)
    yhat = _predict_vector(x)
    logger.info(f"Predicción puntual realizada: {yhat}")
    return {"predicted_temperature": yhat}

@app.post(
    "/predict_n",
    summary="Realiza una predicción de n horas",
    description="Recibe temperaturas de las últimas 25 horas y predice la temperatura de las próximas n horas."
)
def prediction_n(payload: TemperatureN) -> Dict[str, List[float]]:
    data_ = list(payload.data)  # hacerla mutable
    n_hours = int(payload.hours)
    preds: List[float] = []

    for _ in range(n_hours):
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

    logger.info(f"Predicción de {n_hours} horas realizada.")
    return {"predicted_temperature": preds}

@app.post(
    "/model_performance",
    summary="Evalúa el comportamiento del modelo",
    description="Predice secuencialmente y entrega estadísticas para evaluar el rendimiento (RMSE, medias, desviaciones)."
)
def model_performance(payload: ModelPerformance) -> Dict[str, List[float]]:
    data_ = payload.data
    n = len(data_)

    preds: List[float] = []
    reals: List[float] = []

    for i in range(n - 25):
        x = np.array([
            data_[i + 24],
            data_[i + 23],
            data_[i + 22],
            data_[i + 1],
            data_[i + 0],
        ], dtype=float)

        y_true = float(data_[i + 25])
        y_pred = _predict_vector(x)

        preds.append(y_pred)
        reals.append(y_true)

    rmse = [float(np.sqrt(np.mean((np.array(reals) - np.array(preds)) ** 2)))]
    mean_true = [float(np.mean(reals))]
    mean_pred = [float(np.mean(preds))]
    std_true = [float(np.std(reals, ddof=0))]
    std_pred = [float(np.std(preds, ddof=0))]

    logger.info(f"Evaluación de desempeño del modelo realizada. RMSE: {rmse[0]:.4f}")
    return {
        "predicted_temperature": preds,
        "real_temperature": reals,
        "rmse": rmse,
        "mean_true": mean_true,
        "mean_pred": mean_pred,
        "std_true": std_true,
        "std_pred": std_pred
    }
