## O que mudou

<!-- Código, dados, hiperparâmetros, limiares ou documentação? Em uma frase. -->

## Checklist de ML

- [ ] O CI está verde (lint, testes de código, dados, modelo e API)
- [ ] Se mudei `params.yaml` ou o código de treino, conferi a precisão (>= 0,91) e o recall (>= 0,85) do spam
- [ ] Se mudei os limiares em `config.py`, expliquei o motivo abaixo e atualizei o `ADAPTACAO.md`
- [ ] Se adicionei dados ou lotes, rodei `python -m mlops_demo.data validar`
- [ ] Se adicionei um lote em `data/raw/lotes/`, conferi o relatório de drift e a tabela do portão no Summary do CT

## Impacto no modelo

<!-- Cole a tabela do portão de qualidade (reports/gate.md) quando o PR mexer em dados ou treino.
     Lembre: o erro mais caro é o falso positivo (e-mail legítimo na quarentena). -->
