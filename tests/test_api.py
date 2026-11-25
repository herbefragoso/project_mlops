"""Testes de contrato da API: formato de entrada e de saída do /predict."""

import pytest
from fastapi.testclient import TestClient

from mlops_demo import config
from mlops_demo.api import app

pytestmark = pytest.mark.api

cliente = TestClient(app)


def test_health_responde_ok():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_predict_rejeita_quantidade_errada_de_atributos():
    resposta = cliente.post("/predict", json={"atributos": [1.0, 2.0]})
    assert resposta.status_code == 422


def test_predict_devolve_o_formato_combinado(base_holdout):
    email = base_holdout[config.FEATURES].iloc[0].tolist()
    resposta = cliente.post("/predict", json={"atributos": email})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_spam", "versao_codigo"}
    assert corpo["classe"] in {"spam", "legitimo"}
    assert 0.0 <= corpo["probabilidade_spam"] <= 1.0


def test_predict_acerta_um_spam_e_um_legitimo_claros(base_holdout, campeao):
    """Escolhe os e-mails mais claros (maior confiança do campeão) de cada classe."""
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_POSITIVA)
    p_spam = campeao.predict_proba(X)[:, coluna]
    i_spam = int(p_spam.argmax())
    i_legitimo = int(p_spam.argmin())
    for indice, esperado in ((i_spam, "spam"), (i_legitimo, "legitimo")):
        resposta = cliente.post("/predict", json={"atributos": X.iloc[indice].tolist()})
        assert resposta.json()["classe"] == esperado


def test_predict_recusa_atributo_negativo_e_aponta_qual(base_holdout):
    """Frequência ou contagem negativa é impossível: 422, com o índice do atributo culpado."""
    email = base_holdout[config.FEATURES].iloc[0].tolist()
    indice = config.FEATURES.index("char_freq_exclamation")
    email[indice] = -5.0
    resposta = cliente.post("/predict", json={"atributos": email})
    assert resposta.status_code == 422
    erros = resposta.json()["detail"]
    assert [erro["loc"] for erro in erros] == [["body", "atributos", indice]]
