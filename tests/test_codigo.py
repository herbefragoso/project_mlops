"""Testes de código: funções e pipeline se comportam como o contrato diz."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mlops_demo import config
from mlops_demo.evaluate import calcular_metricas
from mlops_demo.features import construir_pipeline, separar_xy
from mlops_demo.gate import decidir

pytestmark = pytest.mark.codigo


def test_separar_xy_devolve_os_57_atributos_e_o_alvo(base_treino):
    X, y = separar_xy(base_treino)
    assert X.shape == (len(base_treino), 57)
    assert len(y) == len(base_treino)


def test_separar_xy_impoe_a_ordem_das_colunas(base_treino):
    embaralhado = base_treino[list(reversed(base_treino.columns))]
    X, _ = separar_xy(embaralhado)
    assert list(X.columns) == config.FEATURES


def test_separar_xy_falha_quando_falta_coluna(base_treino):
    sem_coluna = base_treino.drop(columns=["word_freq_make"])
    with pytest.raises(ValueError, match="Colunas ausentes"):
        separar_xy(sem_coluna)


def test_separar_xy_nao_altera_o_dataframe_original(base_treino):
    antes = base_treino.copy()
    separar_xy(base_treino)
    pd.testing.assert_frame_equal(base_treino, antes)


def test_pipeline_tem_log_padronizacao_e_classificador(params):
    pipeline = construir_pipeline(params)
    assert isinstance(pipeline, Pipeline)
    assert [nome for nome, _ in pipeline.steps] == ["log", "padronizar", "classificador"]


def test_metricas_usam_o_spam_como_classe_positiva():
    # filtro que marca tudo como spam (1): recall 1.0 e precisão = proporção real de spam
    y_real = [0, 0, 1, 1]
    metricas = calcular_metricas(y_real, [1, 1, 1, 1])
    assert metricas["recall_spam"] == 1.0
    assert metricas["precisao_spam"] == pytest.approx(0.5)


# --- o próprio portão de qualidade precisa ser testado ---------------------------------------


def _metricas(recall, precisao):
    return {"recall_spam": recall, "precisao_spam": precisao, "n_amostras": 100}


def test_gate_aprova_candidato_melhor_e_dentro_dos_limiares():
    assert decidir(_metricas(0.97, 0.95), _metricas(0.95, 0.95))["aprovado"]


def test_gate_reprova_recall_abaixo_do_limiar():
    abaixo = config.RECALL_MINIMO - 0.01  # relativo ao contrato: não quebra se o limiar mudar
    assert not decidir(_metricas(abaixo, 0.99), _metricas(abaixo - 0.05, 0.99))["aprovado"]


def test_gate_reprova_precisao_abaixo_do_limiar():
    # recall perfeito não compensa uma precisão ruim
    assert not decidir(_metricas(1.0, 0.70), _metricas(0.95, 0.95))["aprovado"]


def test_gate_reprova_precisao_pior_que_a_do_campeao():
    # o recall melhora, mas a precisão cai em relação ao campeão: o portão olha a precisão
    assert not decidir(_metricas(0.97, 0.93), _metricas(0.90, 0.95))["aprovado"]


def test_gate_aprova_recall_menor_que_o_do_campeao_se_a_precisao_nao_caiu():
    # perder recall (ainda acima do limiar) não barra mais o candidato
    assert decidir(_metricas(0.88, 0.95), _metricas(0.97, 0.95))["aprovado"]


def test_pipeline_quebra_com_atributo_negativo_por_isso_a_api_barra(base_treino, params):
    """Sem a barreira da API, um negativo chega ao log1p, vira NaN e o classificador quebra."""
    X, y = separar_xy(base_treino)
    pipeline = construir_pipeline(params).fit(X, y)
    email = X.iloc[[0]].copy()
    email["capital_run_length_total"] = -5.0  # log1p(-5) = NaN
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="NaN"):
        pipeline.predict(email)
