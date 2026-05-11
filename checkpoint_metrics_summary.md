# Checkpoint Metrics Summary

Generated: 2026-05-11 20:14:45

Metrics are read from every `checkpoint/**/trainer_log.json`. `df_logit_mean` and `df_logit_std` are computed from the saved `df_logit` list when present.

Total runs: **24**

| dataset | gnn | method | df | df_size | seed | exp | dt_auc | dt_aup | df_auc | df_aup | df_logit_mean | df_logit_std | best_epoch | best_metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBLP | rgcn | original | none | 0.5000 | 42 | original | 0.8107 | 0.7986 | nan | nan | nan | nan | 1999 | 0.7934 |
| WordNet18 | rgat | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.0-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.8909 | 0.8834 | 0.5426 | 0.4827 | 0.6488 | 0.2733 |  |  |
| WordNet18 | rgat | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.1-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.8907 | 0.8833 | 0.6266 | 0.5495 | 0.5953 | 0.2596 |  |  |
| WordNet18 | rgat | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.25-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.8900 | 0.8825 | 0.7386 | 0.6675 | 0.5054 | 0.2502 |  |  |
| WordNet18 | rgat | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.5-non_connected/out-0.5-42 | 0.8889 | 0.8806 | 0.6198 | 0.5495 | 0.5903 | 0.2459 |  |  |
| WordNet18 | rgat | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.5-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.8876 | 0.8799 | 0.8503 | 0.8173 | 0.3912 | 0.2267 |  |  |
| WordNet18 | rgat | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.75-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.8819 | 0.8743 | 0.9007 | 0.8883 | 0.3021 | 0.2027 |  |  |
| WordNet18 | rgat | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.9-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.8692 | 0.8638 | 0.9095 | 0.9135 | 0.2277 | 0.1939 |  |  |
| WordNet18 | rgat | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-1.0-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.7492 | 0.7605 | 0.6606 | 0.7882 | 0.5000 | 0.0000 |  |  |
| WordNet18 | rgat | original | none | 0.5000 | 13 | original | 0.8297 | 0.8235 | nan | nan | nan | nan | 1999 | 0.8290 |
| WordNet18 | rgat | original | none | 0.5000 | 21 | original | 0.8603 | 0.8550 | nan | nan | nan | nan | 1999 | 0.8536 |
| WordNet18 | rgat | original | none | 0.5000 | 42 | original | 0.8902 | 0.8823 | nan | nan | nan | nan | 1999 | 0.8884 |
| WordNet18 | rgat | retrain | out | 0.5000 | 42 | out-0.5-42 | 0.8308 | 0.8200 | 0.4299 | 0.4298 | 0.6635 | 0.3201 | 1999 | 1.2584 |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.0-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.9071 | 0.9172 | 0.6762 | 0.7185 | 0.5543 | 0.3984 |  |  |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.1-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.9071 | 0.9172 | 0.6806 | 0.7244 | 0.5502 | 0.3974 |  |  |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.25-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.9070 | 0.9171 | 0.6890 | 0.7349 | 0.5405 | 0.3956 |  |  |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.5-non_connected/out-0.5-42 | 0.9027 | 0.9131 | 0.6772 | 0.7434 | 0.5513 | 0.3648 |  |  |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.5-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.9067 | 0.9168 | 0.7098 | 0.7588 | 0.5127 | 0.3897 |  |  |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.5-non_connected-scgu-16-random/out-0.5-42 | 0.9065 | 0.9165 | 0.6771 | 0.7267 | 0.5596 | 0.3852 |  |  |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.75-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.9060 | 0.9160 | 0.7599 | 0.8082 | 0.4191 | 0.3645 |  |  |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-0.9-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.9024 | 0.9129 | 0.8616 | 0.8907 | 0.1727 | 0.2537 |  |  |
| WordNet18 | rgcn | gnndelete | out | 0.5000 | 42 | mse_mean-both_layerwise-1.0-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42 | 0.7375 | 0.7639 | 0.6185 | 0.7836 | 0.5000 | 0.0000 |  |  |
| WordNet18 | rgcn | original | none | 0.5000 | 42 | original | 0.9073 | 0.9172 | nan | nan | nan | nan | 1999 | 0.9113 |
| WordNet18 | rgcn | retrain | out | 0.5000 | 42 | out-0.5-42 | 0.9068 | 0.9162 | 0.5970 | 0.5968 | 0.6216 | 0.4189 | 1499 | 1.4992 |

## Paths

| # | path |
| --- | --- |
| 1 | `checkpoint/DBLP/rgcn/original/42` |
| 2 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.0-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 3 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.1-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 4 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.25-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 5 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.5-non_connected/out-0.5-42` |
| 6 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.5-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 7 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.75-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 8 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.9-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 9 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-1.0-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 10 | `checkpoint/WordNet18/rgat/original/13` |
| 11 | `checkpoint/WordNet18/rgat/original/21` |
| 12 | `checkpoint/WordNet18/rgat/original/42` |
| 13 | `checkpoint/WordNet18/rgat/retrain/out-0.5-42` |
| 14 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.0-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 15 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.1-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 16 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.25-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 17 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected/out-0.5-42` |
| 18 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 19 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-scgu-16-random/out-0.5-42` |
| 20 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.75-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 21 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.9-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 22 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-1.0-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 23 | `checkpoint/WordNet18/rgcn/original/42` |
| 24 | `checkpoint/WordNet18/rgcn/retrain/out-0.5-42` |
