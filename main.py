import os
import csv
from datetime import datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Inicialización de la App FastAPI ---
app = FastAPI(
    title="API de Predicción de Riesgo Cardiovascular",
    description="Microservicio FastAPI para consumir el modelo Champion desplegado en Databricks Model Serving",
    version="3.0"
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuración del endpoint de Databricks ---
# Estas variables vienen estrictamente de variables de entorno - nunca se escriben en el código
DATABRICKS_URL = os.environ.get("DATABRICKS_URL", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

if not DATABRICKS_URL or not DATABRICKS_TOKEN:
    raise RuntimeError("Variables DATABRICKS_URL o DATABRICKS_TOKEN no configuradas")

LOG_FILE = "predicciones.csv"


# --- Función auxiliar para contar registros guardados ---
def _contar_logs() -> int:
    if not os.path.exists(LOG_FILE):
        return 0
    try:
        with open(LOG_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            return sum(1 for row in reader) if header else 0
    except Exception:
        return 0


# --- Schema de Entrada - Coincide exactamente con las 18 variables del modelo ---
class Paciente(BaseModel):
    Age: float = Field(..., example=54.0, description="Edad del paciente")
    RestingBP: float = Field(..., example=130.0, description="Presión arterial en reposo")
    Cholesterol: float = Field(..., example=283.0, description="Nivel de colesterol serum")
    FastingBS: float = Field(..., example=0.0, description="Glucemia en ayunas > 120 mg/dl (1 = Verdadero, 0 = Falso)")
    MaxHR: float = Field(..., example=162.0, description="Frecuencia cardíaca máxima alcanzada")
    Oldpeak: float = Field(..., example=1.0, description="Depresión del segmento ST inducida por ejercicio")
    frecuencia_reserva_ratio: float = Field(..., example=0.975, description="Ratio de FCR alcanzada vs esperada por edad")
    alto_riesgo_clinico: float = Field(..., example=0.0, description="Flag de Angina + Depresión ST > 1.5 (1 = Sí, 0 = No)")
    carga_vascular: float = Field(..., example=140.4, description="Índice de presión arterial en reposo ponderada por edad")
    Sex_M: float = Field(..., example=1.0, description="Sexo masculino (1 = Hombre, 0 = Mujer)")
    ChestPainType_ATA: float = Field(..., example=0.0, description="Tipo de dolor de pecho: Angina Atípica")
    ChestPainType_NAP: float = Field(..., example=1.0, description="Tipo de dolor de pecho: Dolor No Anginoso")
    ChestPainType_TA: float = Field(..., example=0.0, description="Tipo de dolor de pecho: Angina Típica")
    RestingECG_Normal: float = Field(..., example=1.0, description="Electrocardiograma en reposo: Normal")
    RestingECG_ST: float = Field(..., example=0.0, description="Electrocardiograma en reposo: Anomalía de onda ST-T")
    ExerciseAngina_Y: float = Field(..., example=0.0, description="Angina inducida por el ejercicio: Sí (1 = Sí, 0 = No)")
    ST_Slope_Flat: float = Field(..., example=0.0, description="Pendiente del segmento ST durante esfuerzo: Plana")
    ST_Slope_Up: float = Field(..., example=1.0, description="Pendiente del segmento ST durante esfuerzo: Ascendente")


# --- Endpoints ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0",
        "endpoint_databricks": DATABRICKS_URL,
        "registros_guardados": _contar_logs()
    }


@app.post("/predecir")
def predecir(paciente: Paciente):
    # 1. Reformatear al formato que exige Databricks: dataframe_records
    payload = {
        "dataframe_records": [
            {
                "Age": paciente.Age,
                "RestingBP": paciente.RestingBP,
                "Cholesterol": paciente.Cholesterol,
                "FastingBS": paciente.FastingBS,
                "MaxHR": paciente.MaxHR,
                "Oldpeak": paciente.Oldpeak,
                "frecuencia_reserva_ratio": paciente.frecuencia_reserva_ratio,
                "alto_riesgo_clinico": paciente.alto_riesgo_clinico,
                "carga_vascular": paciente.carga_vascular,
                "Sex_M": paciente.Sex_M,
                "ChestPainType_ATA": paciente.ChestPainType_ATA,
                "ChestPainType_NAP": paciente.ChestPainType_NAP,
                "ChestPainType_TA": paciente.ChestPainType_TA,
                "RestingECG_Normal": paciente.RestingECG_Normal,
                "RestingECG_ST": paciente.RestingECG_ST,
                "ExerciseAngina_Y": paciente.ExerciseAngina_Y,
                "ST_Slope_Flat": paciente.ST_Slope_Flat,
                "ST_Slope_Up": paciente.ST_Slope_Up,
            }
        ]
    }

    # 2. Configurar encabezados de autenticación desde variables de entorno
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    # 3. Llamar al endpoint de Databricks
    try:
        response = requests.post(
            DATABRICKS_URL, 
            json=payload, 
            headers=headers, 
            timeout=10
        )
        response.raise_for_status()
        resultado = response.json()

        prediccion_val = resultado.get("predictions", [None])[0]

        # 4. Registrar en el archivo de logs (predicciones.csv)
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "prediccion", "diagnostico"])
            diagnostico = "Alto Riesgo Cardiovascular" if prediccion_val == 1 else "Riesgo Normal / Bajo"
            writer.writerow([datetime.now().isoformat(), prediccion_val, diagnostico])

        return {
            "status": "éxito",
            "prediccion": prediccion_val,
            "diagnostico": "Alto Riesgo Cardiovascular" if prediccion_val == 1 else "Riesgo Normal / Bajo",
            "respuesta_databricks": resultado
        }

    except requests.exceptions.HTTPError as http_err:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Error HTTP en Databricks Endpoint: {response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno al conectar con Databricks: {str(e)}"
        )