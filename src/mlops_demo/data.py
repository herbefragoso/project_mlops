"""Dados: criação da base, validação de esquema, geração de lotes e preparação.

Uso na linha de comando:
    python -m mlops_demo.data base      # cria base_treino.csv e base_holdout.csv (uma vez)
    python -m mlops_demo.data preparar  # junta base + lotes e gera data/processed/
    python -m mlops_demo.data validar   # valida todos os CSVs de data/raw/

Fonte: Spambase, UCI Machine Learning Repository (Hopkins, Reeber, Forman e Suermondt, 1999),
https://archive.ics.uci.edu/dataset/94/spambase, licença CC BY 4.0. O arquivo original está em
data/fonte/spambase.data e não é alterado: a limpeza é refeita aqui, sempre do mesmo jeito.
"""

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_demo import config

SEMENTE_DADOS = 7  # divisão fixa da base: não mude, ou o campeão perde a referência
N_TREINO_BASE = 2500
# o que sobra (804 linhas) é a reserva de onde saem os lotes novos
N_HOLDOUT_BASE = 900

# colunas usadas pelas regras de domínio do validador
COLUNAS_PORCENTAGEM = [c for c in config.FEATURES if c.startswith(("word_freq_", "char_freq_"))]
COLUNAS_CONTAGEM = ["capital_run_length_longest", "capital_run_length_total"]


def salvar_csv(df: pd.DataFrame, caminho) -> None:
    """Grava o CSV sempre com fim de linha LF.

    O padrão do pandas é o do sistema (CRLF no Windows). Com LF fixo, o arquivo tem os mesmos
    bytes em qualquer máquina e o DVC calcula o mesmo hash no Windows e no GitHub Actions.
    """
    df.to_csv(caminho, index=False, lineterminator="\n")


def carregar_dataset_completo() -> pd.DataFrame:
    """Spambase sem linhas repetidas, com a coluna do alvo.

    O arquivo original tem 4.601 e-mails, mas muitos repetem os mesmos 57 atributos: há
    duplicatas exatas e alguns grupos com o mesmo vetor rotulado ora como spam, ora como legítimo.
    Os grupos contraditórios saem inteiros e as duplicatas ficam com uma cópia só. Sem isso, a
    mesma linha poderia cair no treino e no holdout (vazamento). Restam 4.204 linhas.
    """
    df = pd.read_csv(config.DADOS_FONTE, header=None, names=[*config.FEATURES, config.ALVO])
    rotulos_por_vetor = df.groupby(config.FEATURES)[config.ALVO].transform("nunique")
    df = df[rotulos_por_vetor == 1].drop_duplicates(subset=config.FEATURES)
    return df.reset_index(drop=True)


def dividir_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide o dataset em base de treino, holdout fixo e reserva (origem dos lotes novos)."""
    df = carregar_dataset_completo()
    resto, base_treino = train_test_split(
        df, test_size=N_TREINO_BASE, stratify=df[config.ALVO], random_state=SEMENTE_DADOS
    )
    reserva, base_holdout = train_test_split(
        resto, test_size=N_HOLDOUT_BASE, stratify=resto[config.ALVO], random_state=SEMENTE_DADOS
    )
    return (
        base_treino.reset_index(drop=True),
        base_holdout.reset_index(drop=True),
        reserva.reset_index(drop=True),
    )


def criar_base() -> None:
    """Grava a base de treino e o holdout fixo em data/raw/."""
    base_treino, base_holdout, _ = dividir_dataset()
    config.DADOS_BRUTOS.mkdir(parents=True, exist_ok=True)
    config.DADOS_LOTES.mkdir(parents=True, exist_ok=True)
    salvar_csv(base_treino, config.BASE_TREINO)
    salvar_csv(base_holdout, config.BASE_HOLDOUT)
    print(f"base_treino: {len(base_treino)} linhas | base_holdout: {len(base_holdout)} linhas")


def aplicar_drift(df: pd.DataFrame, fator: float) -> pd.DataFrame:
    """Simula mudança no perfil dos e-mails: sequências de maiúsculas multiplicadas pelo fator."""
    saida = df.copy()
    for coluna in config.COLUNAS_TAMANHO:
        if coluna in COLUNAS_CONTAGEM:
            saida[coluna] = (saida[coluna] * fator).round().astype(int)
        else:
            saida[coluna] = saida[coluna] * fator
    return saida


def gerar_lote(nome: str, n: int = 120, fator_drift: float = 1.0, semente: int = 1) -> str:
    """Cria data/raw/lotes/<nome>.csv a partir da reserva, opcionalmente com drift."""
    _, _, reserva = dividir_dataset()
    lote = reserva.sample(n=min(n, len(reserva)), random_state=semente).reset_index(drop=True)
    if fator_drift != 1.0:
        lote = aplicar_drift(lote, fator_drift)
    config.DADOS_LOTES.mkdir(parents=True, exist_ok=True)
    caminho = config.DADOS_LOTES / f"{nome}.csv"
    salvar_csv(lote, caminho)
    return str(caminho)


def validar_dataframe(df: pd.DataFrame) -> list[str]:
    """Devolve a lista de problemas encontrados (lista vazia = dados válidos)."""
    problemas = []
    esperadas = [*config.FEATURES, config.ALVO]
    if list(df.columns) != esperadas:
        faltando = sorted(set(esperadas) - set(df.columns))
        extras = sorted(set(df.columns) - set(esperadas))
        problemas.append(f"esquema diferente do esperado (faltando={faltando}, extras={extras})")
        return problemas
    nulos = int(df.isna().sum().sum())
    if nulos:
        problemas.append(f"{nulos} valores nulos")
    if not df[config.ALVO].isin([0, 1]).all():
        problemas.append("alvo com valores fora de {0, 1}")
    negativos = [c for c in config.FEATURES if (df[c] < 0).any()]
    if negativos:
        problemas.append(f"valores negativos em {negativos[:3]}")
    # regras do Spambase: porcentagens até 100, contagens de maiúsculas inteiras e coerentes
    acima_de_100 = [c for c in COLUNAS_PORCENTAGEM if (df[c] > 100).any()]
    if acima_de_100:
        problemas.append(f"porcentagens acima de 100 em {acima_de_100[:3]}")
    menores_que_1 = [c for c in config.COLUNAS_TAMANHO if (df[c] < 1).any()]
    if menores_que_1:
        problemas.append(f"sequências de maiúsculas menores que 1 em {menores_que_1}")
    nao_inteiras = [c for c in COLUNAS_CONTAGEM if (df[c] % 1 > 0).any()]
    if nao_inteiras:
        problemas.append(f"contagens não inteiras em {nao_inteiras}")
    media = df["capital_run_length_average"]
    maior = df["capital_run_length_longest"]
    fora_de_ordem = int(((media > maior) | (maior > df["capital_run_length_total"])).sum())
    if fora_de_ordem:
        problemas.append(f"{fora_de_ordem} linhas fora da ordem average <= longest <= total")
    return problemas


def listar_lotes() -> list:
    """Arquivos CSV de lotes novos, em ordem alfabética."""
    if not config.DADOS_LOTES.exists():
        return []
    return sorted(config.DADOS_LOTES.glob("*.csv"))


def preparar() -> None:
    """Junta a base com os lotes. Metade de cada lote treina, a outra metade avalia.

    O holdout fixo e as metades de avaliação dos lotes formam a base de avaliação. Assim campeão
    e candidato são comparados no MESMO conjunto, e o campeão nunca viu esses dados.
    """
    treino = [pd.read_csv(config.BASE_TREINO)]
    avaliacao = [pd.read_csv(config.BASE_HOLDOUT)]
    for caminho in listar_lotes():
        lote = pd.read_csv(caminho)
        metade_treino, metade_aval = train_test_split(
            lote, test_size=0.5, stratify=lote[config.ALVO], random_state=SEMENTE_DADOS
        )
        treino.append(metade_treino)
        avaliacao.append(metade_aval)
    config.DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    salvar_csv(pd.concat(treino, ignore_index=True), config.TREINO_PROCESSADO)
    salvar_csv(pd.concat(avaliacao, ignore_index=True), config.AVALIACAO_PROCESSADA)
    print(f"lotes usados: {len(listar_lotes())}")
    n_treino = sum(len(t) for t in treino)
    n_avaliacao = sum(len(a) for a in avaliacao)
    print(f"treino: {n_treino} linhas | avaliação: {n_avaliacao} linhas")


def validar_tudo() -> int:
    """Valida todos os CSVs de data/raw/. Devolve o código de saída (0 = tudo certo)."""
    arquivos = [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]
    codigo = 0
    for caminho in arquivos:
        problemas = validar_dataframe(pd.read_csv(caminho))
        status = "OK" if not problemas else "FALHOU: " + "; ".join(problemas)
        print(f"{caminho.name}: {status}")
        codigo |= 1 if problemas else 0
    return codigo


if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "base":
        criar_base()
    elif comando == "preparar":
        preparar()
    elif comando == "validar":
        sys.exit(validar_tudo())
    else:
        print(__doc__)
        sys.exit(2)
