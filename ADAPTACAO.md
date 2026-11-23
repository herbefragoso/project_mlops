# Adaptação: filtro de spam (Spambase)

## 1. Problema e dados

O modelo classifica e-mails em spam (1) ou legítimo (0), para a equipe de TI filtrar a caixa de entrada.
Fonte: Spambase, UCI Machine Learning Repository (Hopkins, Reeber, Forman e Suermondt, 1999),
<https://archive.ics.uci.edu/dataset/94/spambase>. Licença: CC BY 4.0.
Tamanho: 4.601 e-mails e 57 atributos numéricos. Removemos 391 duplicatas exatas e 3 grupos com o
mesmo vetor rotulado de formas diferentes: restam 4.204 linhas (39,9% de spam), divididas com
semente fixa em treino (2.500), holdout (900) e reserva para os lotes (804), sem sobreposição.

## 2. Contrato de qualidade

Classe positiva: spam. Erro mais caro: falso positivo (e-mail legítimo na quarentena), que a própria
documentação do dataset chama de "muito indesejável". Por isso a métrica principal é a precisão:

- precisão do spam >= 0,91 e recall do spam >= 0,85 (sem o piso, um filtro que quase nunca marca
  spam teria precisão alta);
- o candidato não pode ter precisão menor que a do campeão, no mesmo conjunto.

Sem log1p e com class_weight balanced, a precisão no holdout era 0,900. O campeão atual tem
precisão 0,919 e recall 0,916.

## 3. O que mudou, arquivo por arquivo

- [PR #1](https://github.com/herbefragoso/project_mlops/pull/1) (herbefragoso): `data/fonte/spambase.data`,
  `data/raw/*.csv`, `data.py`, `config.py`, `features.py` (log1p), `params.yaml` (class_weight null), primeiro campeão
- [PR do Ruan]: `validar_dataframe` com as regras do domínio + teste novo de dados
- [PR da Erica]: métricas do spam, portão pela precisão + teste novo de modelo
- [PR do Roger]: `aplicar_drift` e cenários 1 a 5
- [PR do Rafael]: API, smoke test + testes novos de API e de código
- [PR da Amabilly]: `README.md`, `.github/PULL_REQUEST_TEMPLATE.md` e este relatório

## 4. Evidências

- E1, CI verde na main: [link]
- E2, CI vermelho (cenário 2): [link do PR] [link da execução]
- E3, CT com drift e portão aprovando (cenário 4): [link do CT no PR] [link do PR do robô]
- E4, CT com portão reprovando (cenário 5): [link]
- E5, CD com a tag v1.0.0: [link da execução] [link do pacote]

## 5. Limites e riscos

O maior limite está nos próprios dados. Os e-mails legítimos do Spambase vieram de uma única pessoa, então
palavras como "george" e "650" (o nome dele e o prefixo do telefone) indicam "legítimo" só neste dataset. No
holdout, 14,3% dos e-mails têm "george", e zerar "george" e "650" muda 1,0% das previsões. O portão mede
precisão e recall no holdout, que tem o mesmo viés, e por isso não enxerga esse atalho. Além disso, os dados
são de 1999: o vocabulário do spam mudou desde então, e um holdout fixo de 900 e-mails não mostra essa perda.
O portão também é sensível a ruído. Sem tolerância, uma diferença de 0,008 na precisão (cerca de 3 e-mails)
decide a promoção: 16 de 36 simulações de drift reprovaram só por isso.

Em produção, monitoraríamos os e-mails resgatados da quarentena (os falsos positivos reais), a proporção de
e-mails marcados como spam por dia, o drift dos atributos e a precisão numa amostra rotulada por pessoas.
Esses sinais mostram o que o holdout fixo não mostra: se o filtro está errando com os e-mails de hoje.

## 6. Contribuição

- Herbe Fragoso: troca dos dados, contrato de qualidade, pré-processamento e primeiro campeão ([PR #1](https://github.com/herbefragoso/project_mlops/pull/1))
- Ruan: regras do domínio no `validar_dataframe` e teste novo de dados ([PR])
- Erica: métricas do spam, portão pela precisão e teste novo de modelo ([PR])
- Roger: simulação de drift e cenários 1 a 5 ([PR])
- Rafael: API, teste de fumaça e testes novos de API e de código ([PR])
- Amabilly Riva Dutra: README, modelo de PR, este relatório e as evidências ([PR])
