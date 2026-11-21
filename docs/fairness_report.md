# Fairness report

Generato automaticamente da `python -m credit_xai.fairness` sul test set (200 richieste). Gli attributi sensibili **non** sono feature del modello.

Lettura: `approval_rate` = quota approvata; `false_rejection_rate` = buoni pagatori rifiutati; `false_approval_rate` = cattivi pagatori approvati. Un `disparate_impact_ratio` sotto 0,8 viola la regola dei 4/5.

## sex

| sex   |   approval_rate |   false_rejection_rate |   false_approval_rate |   n |    auc |
|:------|----------------:|-----------------------:|----------------------:|----:|-------:|
| F     |          0.2787 |                 0.6341 |                   0.1 |  61 | 0.778  |
| M     |          0.3741 |                 0.5152 |                   0.1 | 139 | 0.8129 |

- Differenza di demographic parity: **0.0954**
- Differenza di equalized odds: **0.119**
- Disparate impact ratio: **0.745**

## age_group

| age_group   |   approval_rate |   false_rejection_rate |   false_approval_rate |   n |    auc |
|:------------|----------------:|-----------------------:|----------------------:|----:|-------:|
| <=25        |          0.2045 |                 0.75   |                0.15   |  44 | 0.725  |
| 26-35       |          0.3485 |                 0.5111 |                0.0476 |  66 | 0.836  |
| 36-50       |          0.4426 |                 0.4468 |                0.0714 |  61 | 0.8511 |
| >50         |          0.3448 |                 0.625  |                0.2    |  29 | 0.7417 |

- Differenza di demographic parity: **0.2381**
- Differenza di equalized odds: **0.3032**
- Disparate impact ratio: **0.4621**

## foreign_worker

| foreign_worker   |   approval_rate |   false_rejection_rate |   false_approval_rate |   n |   auc |
|:-----------------|----------------:|-----------------------:|----------------------:|----:|------:|
| no               |          0.5    |                 0.4    |                0      |   6 | 1     |
| sì               |          0.3402 |                 0.5556 |                0.1017 | 194 | 0.798 |

- Differenza di demographic parity: **0.1598**
- Differenza di equalized odds: **0.1556**
- Disparate impact ratio: **0.6804**

## Mitigazione: ThresholdOptimizer (equalized_odds) su `age_group`

Prima = modello con soglia unica a costo minimo; dopo = soglie per gruppo scelte per l'accuratezza bilanciata sotto il vincolo di equità. Le due accuratezze non vanno lette come un confronto puro: guardare soprattutto le differenze tra gruppi.

| Metrica | Prima | Dopo |
| --- | --- | --- |
| Accuratezza | 0.585 | 0.79 |
| Equalized odds diff. | 0.3032 | 0.2643 |
| Demographic parity diff. | 0.2381 | 0.1846 |

## Limiti

- Gruppi piccoli: con poche decine di richieste per gruppo le differenze possono essere rumore. Riportare sempre la colonna `n`.
- La mitigazione usa l'attributo sensibile al momento della decisione: valutarne la liceità prima di un uso reale.
- Escludere un attributo non basta: altre variabili possono farne da proxy.
