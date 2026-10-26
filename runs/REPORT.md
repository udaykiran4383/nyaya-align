# Nyaya: results

## Accuracy by condition and question type (exact section match)

| condition | n | overall | forward | reverse | stale | heading | temporal_pre | temporal_post | temporal_straddle | trap rate* |
|---|---|---|---|---|---|---|---|---|---|---|
| base_closed | 100 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | - | - | - | 37.9% |
| aligned_closed | 2048 | 0.9% | 0.9% | 0.4% | 0.9% | 0.6% | 2.2% | 0.9% | 0.9% | 15.6% |
| base_rag | 100 | 63.0% | 50.0% | 0.0% | 64.7% | 80.7% | - | - | - | 13.8% |
| aligned_rag | 572 | 38.5% | 34.9% | 0.6% | 69.2% | 70.2% | - | - | - | 23.8% |
| symbolic | 2054 | 77.1% | 85.8% | 80.6% | 56.0% | 95.3% | 70.2% | 56.1% | 60.1% | 0.3% |

*share of questions with a trap answered with the trap: the repealed number where the new law applies, or the new number where the old law is saved (temporal_pre / straddle).

## Held-out vs taught mappings

| condition | train acc | test acc |
|---|---|---|
| base_closed | - (0) | 0.0% (43) |
| aligned_closed | 1.1% (1098) | 0.7% (276) |
| base_rag | - (0) | 39.5% (43) |
| aligned_rag | 35.1% (402) | 34.5% (113) |
| symbolic | 68.7% (1104) | 66.3% (276) |

## Is the confidence score informative?

AUROC = probability a correct answer carries a higher confidence than a wrong one (0.5 = no signal).

| condition | AUROC | acc @ conf>=80 | acc @ 50-80 | acc @ <50 | coverage @ >=80 |
|---|---|---|---|---|---|
| aligned_closed | 0.357 | 0.6% | 0.9% | - | 8.6% |
| aligned_rag | 0.592 | 47.4% | 25.2% | - | 59.8% |

## Certified selective answering (conformal risk control)

Threshold learned on the train split, so that with 90% probability the error rate on answered questions is at most 10%. Held-out test results at that threshold:

| condition | threshold | test coverage | test error when answering | test error without gate |
|---|---|---|---|---|
| aligned_closed | none certifiable | - | - | 99.1% |
| aligned_rag | none certifiable | - | - | 53.5% |

## Retrieval

Gold section in the reranked top-4: 64.3% (n=672).


## Token usage by condition

| tag | calls | prompt/input tokens | completion tokens |
|---|---|---|---|
| aligned_closed | 2054 | 424,253 | 17,134 |
| aligned_rag | 1455 | 677,223 | 5,904 |
| base_closed | 100 | 20,420 | 3,060 |
| base_rag | 200 | 88,514 | 910 |
| concord-dev | 7 | 0 | 0 |
| debug | 110 | 8,545 | 0 |
| index | 47 | 253,367 | 0 |
| probe | 4 | 786 | 207 |
| smoke | 3 | 75 | 15 |
