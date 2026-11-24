"""Métricas de avaliação. A classe de interesse é a CLASSE_POSITIVA do config (spam, rótulo 1)."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from mlops_demo.config import CLASSE_POSITIVA
from mlops_demo.features import separar_xy


def calcular_metricas(y_real, y_previsto) -> dict:
    """Precisão (métrica principal) e recall da classe positiva, mais a acurácia como referência."""
    return {
        "precisao_spam": float(
            precision_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA, zero_division=0)
        ),
        "recall_spam": float(recall_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA)),
        "acuracia": float(accuracy_score(y_real, y_previsto)),
        "n_amostras": int(len(y_real)),
    }


def avaliar(modelo, df: pd.DataFrame) -> dict:
    """Avalia um modelo treinado em um DataFrame com as colunas do projeto."""
    X, y = separar_xy(df)
    return calcular_metricas(y, modelo.predict(X))
