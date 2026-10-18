import os
import csv
from datetime import datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="API de Predicción de Rendimiento Estudiantil", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATABRICKS_URL = os.environ.get("DATABRICKS_URL", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
LOG_FILE = "predicciones.csv"

class Estudiante(BaseModel):
    school: float = Field(...)
    sex: float = Field(...)
    age: float = Field(...)
    address: float = Field(...)
    famsize: float = Field(...)
    Pstatus: float = Field(...)
    Medu: float = Field(...)
    Fedu: float = Field(...)
    Mjob: float = Field(...)
    Fjob: float = Field(...)
    reason: float = Field(...)
    guardian: float = Field(...)
    traveltime: float = Field(...)
    studytime: float = Field(...)
    failures: float = Field(...)
    schoolsup: float = Field(...)
    famsup: float = Field(...)
    paid: float = Field(...)
    activities: float = Field(...)
    nursery: float = Field(...)
    higher: float = Field(...)
    internet: float = Field(...)
    romantic: float = Field(...)
    famrel: float = Field(...)
    freetime: float = Field(...)
    goout: float = Field(...)
    Dalc: float = Field(...)
    Walc: float = Field(...)
    health: float = Field(...)
    absences: float = Field(...)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predecir")
def predecir(estudiante: Estudiante):
    total_alcohol = int(estudiante.Dalc + estudiante.Walc)
    study_to_freetime_ratio = float(estudiante.studytime / estudiante.freetime) if estudiante.freetime != 0 else 0.0
    en_riesgo_academico = 1 if (estudiante.failures > 0 or estudiante.absences > 10) else 0

    payload = {
        "dataframe_records": [{
            "age": int(estudiante.age),
            "Medu": int(estudiante.Medu),
            "Fedu": int(estudiante.Fedu),
            "traveltime": int(estudiante.traveltime),
            "studytime": int(estudiante.studytime),
            "failures": int(estudiante.failures),
            "famrel": int(estudiante.famrel),
            "freetime": int(estudiante.freetime),
            "goout": int(estudiante.goout),
            "Dalc": int(estudiante.Dalc),
            "Walc": int(estudiante.Walc),
            "health": int(estudiante.health),
            "absences": int(estudiante.absences),
            "total_alcohol": total_alcohol,
            "study_to_freetime_ratio": study_to_freetime_ratio,
            "en_riesgo_academico": en_riesgo_academico
        }]
    }

    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

    try:
        response = requests.post(DATABRICKS_URL, json=payload, headers=headers, timeout=35)
        response.raise_for_status()
        resultado = response.json()
        prediccion_val = resultado.get("predictions", [None])[0]
        return {"status": "éxito", "prediccion": prediccion_val}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error HTTP: {response.text}" if 'response' in locals() else str(e))