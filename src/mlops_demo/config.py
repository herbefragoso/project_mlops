"""Configuração central do projeto: caminhos, nomes de colunas e limiares de qualidade."""

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]

# dados
DADOS_FONTE = RAIZ / "data" / "fonte" / "spambase.data"  # arquivo original da UCI, sem alteração
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_LOTES = DADOS_BRUTOS / "lotes"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
BASE_TREINO = DADOS_BRUTOS / "base_treino.csv"
BASE_HOLDOUT = DADOS_BRUTOS / "base_holdout.csv"
TREINO_PROCESSADO = DADOS_PROCESSADOS / "treino.csv"
AVALIACAO_PROCESSADA = DADOS_PROCESSADOS / "avaliacao.csv"

# modelos e relatórios
MODELO_CAMPEAO = RAIZ / "models" / "champion" / "model.joblib"
METRICAS_CAMPEAO = RAIZ / "models" / "champion" / "metrics.json"
MODELO_CANDIDATO = RAIZ / "models" / "candidate" / "model.joblib"
METRICAS_CANDIDATO = RAIZ / "reports" / "candidate_metrics.json"
RELATORIO_GATE = RAIZ / "reports" / "gate.json"
RESUMO_GATE = RAIZ / "reports" / "gate.md"
RELATORIO_DRIFT = RAIZ / "reports" / "drift.md"
PARAMS = RAIZ / "params.yaml"

# esquema dos dados (Spambase, UCI): 57 atributos numéricos extraídos de cada e-mail
_PALAVRAS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail",
    "receive", "will", "people", "report", "addresses", "free", "business", "email", "you",
    "credit", "your", "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs",
    "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
    "meeting", "original", "project", "re", "edu", "table", "conference",
]  # fmt: skip
# na UCI estas colunas se chamam char_freq_; char_freq_( char_freq_[ char_freq_! char_freq_$ e
# char_freq_#. Os símbolos foram trocados por nomes para servirem de identificador na API.
_CARACTERES = ["semicolon", "parenthesis", "bracket", "exclamation", "dollar", "hash"]
FEATURES = (
    [f"word_freq_{palavra}" for palavra in _PALAVRAS]  # % das palavras do e-mail iguais a WORD
    + [f"char_freq_{nome}" for nome in _CARACTERES]  # % dos caracteres do e-mail iguais a CHAR
    + [
        "capital_run_length_average",  # tamanho médio das sequências de letras maiúsculas
        "capital_run_length_longest",  # tamanho da maior sequência de maiúsculas
        "capital_run_length_total",  # total de letras maiúsculas no e-mail
    ]
)
ALVO = "spam"
CLASSE_POSITIVA = 1  # 1 = spam (a classe que o filtro precisa encontrar), 0 = e-mail legítimo
COLUNAS_TAMANHO = [c for c in FEATURES if c.startswith("capital_run_length")]

# limiares de qualidade (o "contrato" do modelo). Mudar aqui exige revisão em PR.
# O erro mais caro é o falso positivo (e-mail legítimo na quarentena), então a precisão manda.
PRECISAO_MINIMA = 0.91
RECALL_MINIMO = 0.85  # sem ele, um filtro que quase nunca marca spam teria precisão alta
TOLERANCIA_CAMPEAO = 0.0  # o candidato não pode ter recall menor que o campeão
PISO_SANIDADE = 0.80  # o campeão não pode colapsar na base de referência (holdout fixo)
LIMITE_DRIFT_P = 0.01  # p-valor do teste KS
MIN_FEATURES_DRIFT = 3  # gatilho do CT: quantos atributos com drift disparam o retreino


def carregar_params() -> dict:
    """Lê o params.yaml (hiperparâmetros e semente)."""
    with open(PARAMS, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
