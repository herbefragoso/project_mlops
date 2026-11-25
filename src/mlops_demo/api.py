"""API de inferência (FastAPI) servindo o modelo campeão.

Local:   uvicorn mlops_demo.api:app --reload
Docker:  docker run -p 8000:8000 mlops-demo
"""

import json
import os
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlops_demo import __version__, config

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")

app = FastAPI(title="Filtro de spam (demo CI/CD/CT)", version=__version__)
_modelo = joblib.load(CAMINHO_MODELO)

# Todo atributo do Spambase é uma frequência (%) ou uma contagem de maiúsculas: nunca negativo.
# A barreira fica na entrada porque o log1p do pipeline transforma valores menores que -1 em NaN
# e o classificador quebra (HTTP 500); entre -1 e 0 seria pior: uma previsão sem sentido e sem
# erro. NaN e infinito também são recusados. Tudo isso vira HTTP 422, com o índice do atributo.
AtributoDoEmail = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Email(BaseModel):
    """Os 57 atributos do e-mail, na mesma ordem do treino (config.FEATURES)."""

    atributos: list[AtributoDoEmail] = Field(
        min_length=len(config.FEATURES),
        max_length=len(config.FEATURES),
        description="Frequências de palavras e caracteres (%) e contagens de maiúsculas, >= 0.",
    )


class Predicao(BaseModel):
    classe: str  # "spam" ou "legitimo"
    probabilidade_spam: float
    versao_codigo: str


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {"status": "ok", "versao_codigo": __version__, "metricas_do_campeao": metricas}


@app.post("/predict", response_model=Predicao)
def prever(email: Email) -> Predicao:
    entrada = pd.DataFrame([email.atributos], columns=config.FEATURES)
    previsto = int(_modelo.predict(entrada)[0])
    # a coluna da classe positiva em predict_proba segue a ordem de classes_
    coluna = list(_modelo.classes_).index(config.CLASSE_POSITIVA)
    p_spam = float(_modelo.predict_proba(entrada)[0][coluna])
    return Predicao(
        classe="spam" if previsto == config.CLASSE_POSITIVA else "legitimo",
        probabilidade_spam=round(p_spam, 4),
        versao_codigo=__version__,
    )
