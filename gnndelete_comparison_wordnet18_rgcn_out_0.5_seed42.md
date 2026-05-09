# GNNDelete Comparison: WordNet18 R-GCN

## Experiment

| Item | Value |
|---|---|
| Dataset | WordNet18 |
| Model | R-GCN |
| Unlearning method | GNNDelete |
| Deleted set | out |
| Deleted ratio | 0.5% |
| Deleted edges | 707 / 141442 |
| Random seed | 42 |

## Retained Test Performance

| Metric | Before deletion: Original | After deletion: GNNDelete | Change |
|---|---:|---:|---:|
| test_loss / dt_loss | 1.0001 | 1.0520 | +0.0519 |
| test_dt_auc | 0.9073 | 0.9027 | -0.0046 |
| test_dt_aup | 0.9172 | 0.9131 | -0.0041 |

## Deleted Edge Scores

| Metric | Before deletion: Original | After deletion: GNNDelete |
|---|---:|---:|
| Mean deleted-edge probability | 0.9399 | 0.5513 |
| Median deleted-edge probability | 1.0000 | 0.6178 |
| Deleted-edge probability std | 0.2362 | 0.3648 |
| Deleted edges with probability < 0.5 | 43 / 707 | 317 / 707 |

## After-Deletion Deleted Set Metrics

| Metric | Value |
|---|---:|
| df_auc | 0.6772 |
| df_aup | 0.7434 |
| df_logit_mean | 0.5513 |
| df_logit_std | 0.3648 |

## Output Paths

Original checkpoint:

```text
checkpoint/WordNet18/rgcn/original/42/
```

GNNDelete checkpoint:

```text
checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected/out-0.5-42/
```

Offline W&B run:

```text
wandb/offline-run-20260427_125125-jyi97cn6/
```

## Summary

GNNDelete preserved the retained test performance well. The retained test AUC changed from 0.9073 to 0.9027, a decrease of 0.0046.

The deleted edges were substantially weakened. Their mean prediction probability dropped from 0.9399 before deletion to 0.5513 after deletion, and the number of deleted edges with probability below 0.5 increased from 43 to 317.

