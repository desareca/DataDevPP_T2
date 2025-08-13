import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("./model/model_multiple.joblib")

def predict_temp(features_temp):
    """Recibe un vector de características de temperaturas anteriores y predice 
       la temperatura en la siguiente hora.

    Argumentos:
        features_temp: Lista de temperaturas de hace 1, 2, 3, 24 y 25 horas.
    """
    
    pred_value = model.predict(features_temp.reshape(1, -1))[0]
    return pred_value

# Asignamos una instancia de la clase FastAPI a la variable "app".
# Interacturaremos con la API usando este elemento.
app = FastAPI(title='Implementando un modelo de Machine Learning usando FastAPI')

# Creamos una clase para el vector de features de entrada
class Temperature(BaseModel):
    Ts_Valor_1h: float
    Ts_Valor_2h: float
    Ts_Valor_3h: float
    Ts_Valor_24h: float
    Ts_Valor_25h: float

# Usando @app.get("/") definimos un método GET para el endpoint / (que sería como el "home").
@app.get("/")
def home():
    return "¡Felicitaciones! Tu API está funcionando según lo esperado. Anda ahora a http://localhost:8000/docs."


# Este endpoint maneja la lógica necesaria para la regresión.
# Requiere como entrada el vector de temperaturas para la regresión.
@app.post("/predict") 
def prediction(temp: Temperature):
    # 1. Correr el modelo de clasificación
    features_temp = np.array([temp.Ts_Valor_1h, temp.Ts_Valor_2h, temp.Ts_Valor_3h, temp.Ts_Valor_24h, temp.Ts_Valor_25h])
    pred = predict_temp(features_temp)
    
    # 2. Transmitir la respuesta de vuelta al cliente
    # Retornar el resultado de la predicción
    return {'predicted_temperature': pred}