import os
import csv
from datetime import datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Inicialización de la App FastAPI ---
app = FastAPI(
    title="API de Predicción de Rendimiento Estudiantil",
    description="Microservicio FastAPI para consumir el modelo desplegado en Databricks",
    version="4.0"
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


# --- Schema de Entrada - 30 variables exactas exigidas por MLflow ---
class Estudiante(BaseModel):
    school: float = Field(..., example=0.0, description="Escuela del estudiante")
    sex: float = Field(..., example=1.0, description="Sexo del estudiante")
    age: float = Field(..., example=15.0, description="Edad del estudiante")
    address: float = Field(..., example=0.0, description="Tipo de dirección (urbana/rural)")
    famsize: float = Field(..., example=1.0, description="Tamaño de la familia")
    Pstatus: float = Field(..., example=1.0, description="Estado de convivencia de los padres")
    Medu: float = Field(..., example=4.0, description="Educación de la madre")
    Fedu: float = Field(..., example=4.0, description="Educación del padre")
    Mjob: float = Field(..., example=2.0, description="Trabajo de la madre")
    Fjob: float = Field(..., example=3.0, description="Trabajo del padre")
    reason: float = Field(..., example=1.0, description="Razón para elegir la escuela")
    guardian: float = Field(..., example=1.0, description="Tutor del estudiante")
    traveltime: float = Field(..., example=1.0, description="Tiempo de viaje a la escuela")
    studytime: float = Field(..., example=2.0, description="Tiempo de estudio semanal")
    failures: float = Field(..., example=0.0, description="Número de fracasos en clases pasadas")
    schoolsup: float = Field(..., example=1.0, description="Apoyo educativo extra")
    famsup: float = Field(..., example=0.0, description="Apoyo educativo familiar")
    paid: float = Field(..., example=0.0, description="Clases pagadas extra")
    activities: float = Field(..., example=1.0, description="Actividades extracurriculares")
    nursery: float = Field(..., example=1.0, description="Asistió a guardería")
    higher: float = Field(..., example=1.0, description="Desea educación superior")
    internet: float = Field(..., example=1.0, description="Acceso a internet en casa")
    romantic: float = Field(..., example=0.0, description="En una relación romántica")
    famrel: float = Field(..., example=4.0, description="Calidad de las relaciones familiares")
    freetime: float = Field(..., example=3.0, description="Tiempo libre después de la escuela")
    goout: float = Field(..., example=2.0, description="Salidas con amigos")
    Dalc: float = Field(..., example=1.0, description="Consumo de alcohol en días laborales")
    Walc: float = Field(..., example=1.0, description="Consumo de alcohol en fines de semana")
    health: float = Field(..., example=3.0, description="Estado de salud actual")
    absences: float = Field(..., example=2.0, description="Número de ausencias escolares")


# --- Endpoints ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.0",
        "endpoint_databricks": DATABRICKS_URL,
        "registros_guardados": _contar_logs()
    }


@app.post("/predecir")
def predecir(estudiante: Estudiante):
    # 1. Calcular las nuevas características (Feature Engineering)
    total_alcohol = float(estudiante.Dalc + estudiante.Walc)
    study_to_freetime_ratio = float(estudiante.studytime / estudiante.freetime) if estudiante.freetime != 0 else 0.0
    en_riesgo_academico = 1.0 if (estudiante.failures > 0 or estudiante.absences > 10) else 0.0

    # 2. Reformatear al formato EXACTO de 16 variables que exige Databricks
    payload = {
        "dataframe_records": [
            {
                "age": estudiante.age,
                "Medu": estudiante.Medu,
                "Fedu": estudiante.Fedu,
                "traveltime": estudiante.traveltime,
                "studytime": estudiante.studytime,
                "failures": estudiante.failures,
                "famrel": estudiante.famrel,
                "freetime": estudiante.freetime,
                "goout": estudiante.goout,
                "Dalc": estudiante.Dalc,
                "Walc": estudiante.Walc,
                "health": estudiante.health,
                "absences": estudiante.absences,
                "total_alcohol": total_alcohol,
                "study_to_freetime_ratio": study_to_freetime_ratio,
                "en_riesgo_academico": en_riesgo_academico
            }
        ]
    }

    # 3. Configurar encabezados de autenticación
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    # 4. Llamar al endpoint de Databricks
    try:
        response = requests.post(
            DATABRICKS_URL, 
            json=payload, 
            headers=headers, 
            timeout=35
        )
        response.raise_for_status()
        resultado = response.json()

        prediccion_val = resultado.get("predictions", [None])[0]

        # 5. Registrar en el archivo de logs
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "prediccion", "diagnostico_academico"])
            
            estado = "Aprobado" if prediccion_val == 1 else "Reprobado"
            writer.writerow([datetime.now().isoformat(), prediccion_val, estado])

        return {
            "status": "éxito",
            "prediccion": prediccion_val,
            "diagnostico_academico": estado,
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