# 实验指标汇总与比较

生成时间：2026-05-08 UTC

共发现 `25` 个 `trainer_log.json`。指标来自 `checkpoint/**/trainer_log.json`，配置来自同目录 `training_args.json`。

## 字段说明

- **dt_auc / dt_aup**：保留/测试集相关 AUC、AUP，通常越高越好。
- **df_auc / df_aup**：删除集相关 AUC、AUP；若目标是遗忘，通常希望删除集表现降低，但具体方向取决于你的评价定义。
- **df_logit_mean**：删除边 logit/概率均值，越低通常表示模型越不倾向保留删除边。
- **dt_loss**：测试/保留集 loss；越低越好。
- **final_train_loss / min_train_loss**：训练日志中最后一次和最小训练 loss。

## 总表

| # | dataset | gnn | model | df | df_size | alpha | loss_r_type | operator | gate | dt_auc | dt_aup | df_auc | df_aup | df_logit_mean | dt_loss | final_train_loss | min_train_loss | df_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DBLP | rgcn | original | none | 0.5000 | 0.5000 | embedding_mse | original | soft | 0.8107 | 0.7986 | - | - | - | 0.5990 | 0.1512 | 0.1512 | 0 |
| 2 | WordNet18 | rgat | gnndelete | out | 0.5000 | 0.2000 | bce_rank_l2 | relation_lora_moe | soft | 0.8904 | 0.8830 | 0.7084 | 0.6308 | 0.5319 | 0.4439 | 1.9318 | 1.4494 | 707 |
| 3 | WordNet18 | rgat | gnndelete | out | 0.5000 | 0.5000 | embedding_mse | original | soft | 0.8889 | 0.8806 | 0.6198 | 0.5495 | 0.5903 | 0.4482 | 7.8219 | 4.8779 | 707 |
| 4 | WordNet18 | rgat | gnndelete | out | 0.5000 | 0.5000 | embedding_mse | relation_lora_moe | soft | 0.8907 | 0.8832 | 0.6041 | 0.5364 | 0.6174 | 0.4431 | 5.0852 | 4.1776 | 707 |
| 5 | WordNet18 | rgat | gnndelete | out | 0.5000 | 0.5000 | bce_rank_l2 | relation_lora_moe | soft | 0.8876 | 0.8799 | 0.8498 | 0.8172 | 0.3919 | 0.4486 | 3.9149 | 2.5801 | 707 |
| 6 | WordNet18 | rgat | gnndelete | out | 0.5000 | 0.7500 | bce_rank_l2 | relation_lora_moe | soft | 0.8814 | 0.8738 | 0.9004 | 0.8845 | 0.2972 | 0.4577 | 5.2144 | 3.3680 | 707 |
| 7 | WordNet18 | rgat | gnndelete | out | 0.5000 | 0.7500 | bce_l2 | lora_moe | soft | 0.8849 | 0.8772 | 0.8892 | 0.8681 | 0.3320 | 0.4526 | 4.5400 | 2.9913 | 707 |
| 8 | WordNet18 | rgat | gnndelete | out | 0.5000 | 0.7500 | rank_l2 | lora_moe | soft | 0.8872 | 0.8793 | 0.8513 | 0.8233 | 0.3966 | 0.4496 | 4.0864 | 2.6575 | 707 |
| 9 | WordNet18 | rgat | gnndelete | random | 0.5000 | 0.7500 | bce_l2 | lora_moe | soft | 0.8080 | 0.8084 | 0.7852 | 0.7914 | 0.3147 | 0.5689 | 3.2831 | 2.7430 | 707 |
| 10 | WordNet18 | rgat | original | none | 0.5000 | 0.5000 | embedding_mse | original | soft | 0.8297 | 0.8235 | - | - | - | 0.7035 | 0.1560 | 0.1560 | 0 |
| 11 | WordNet18 | rgat | original | none | 0.5000 | 0.5000 | embedding_mse | original | soft | 0.8603 | 0.8550 | - | - | - | 0.4925 | 0.1476 | 0.1476 | 0 |
| 12 | WordNet18 | rgat | original | none | 0.5000 | 0.5000 | embedding_mse | original | soft | 0.8902 | 0.8823 | - | - | - | 0.4427 | 0.1473 | 0.1473 | 0 |
| 13 | WordNet18 | rgcn | gnndelete | in | 0.5000 | 0.5000 | embedding_mse | relation_lora_moe | hard | 0.9028 | 0.9136 | 0.7190 | 0.7422 | 0.4717 | 1.0185 | 6.8938 | 5.2207 | 707 |
| 14 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.2000 | bce_rank_l2 | relation_lora_moe | soft | 0.9071 | 0.9171 | 0.6859 | 0.7310 | 0.5442 | 1.0065 | 1.4575 | 1.4575 | 707 |
| 15 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.5000 | - | - | - | 0.9027 | 0.9131 | 0.6772 | 0.7434 | 0.5513 | 1.0520 | 5.9470 | 4.4628 | 707 |
| 16 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.5000 | embedding_mse | relation_lora_moe | soft | 0.9068 | 0.9168 | 0.6817 | 0.7394 | 0.5644 | 1.0172 | 4.4637 | 4.3252 | 707 |
| 17 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.5000 | embedding_mse | scgu | soft | 0.9065 | 0.9165 | 0.6771 | 0.7267 | 0.5596 | 1.0178 | 5.1761 | 1.4725 | 707 |
| 18 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.5000 | bce_rank_l2 | - | - | 0.9033 | 0.9136 | 0.7120 | 0.7639 | 0.4859 | 1.0425 | 4.2158 | 3.7624 | 707 |
| 19 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.5000 | bce_rank_l2 | relation_lora_moe | hard | 0.9067 | 0.9168 | 0.7075 | 0.7569 | 0.5152 | 1.0107 | 3.4068 | 3.4068 | 707 |
| 20 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.5000 | bce_rank_l2 | relation_lora_moe | soft | 0.9067 | 0.9168 | 0.7098 | 0.7588 | 0.5127 | 1.0111 | 3.4075 | 3.4075 | 707 |
| 21 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.5000 | bce_l2 | - | - | 0.9034 | 0.9138 | 0.7011 | 0.7519 | 0.5013 | 1.0423 | 3.9003 | 3.3952 | 707 |
| 22 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.5000 | rank_l2 | - | - | 0.9035 | 0.9139 | 0.6901 | 0.7409 | 0.5179 | 1.0410 | 3.0102 | 2.7672 | 707 |
| 23 | WordNet18 | rgcn | gnndelete | out | 0.5000 | 0.7500 | bce_rank_l2 | relation_lora_moe | soft | 0.9060 | 0.9160 | 0.7599 | 0.8082 | 0.4191 | 1.0195 | 4.9593 | 4.9593 | 707 |
| 24 | WordNet18 | rgcn | original | none | 0.5000 | 0.5000 | embedding_mse | original | soft | 0.9073 | 0.9172 | - | - | - | 1.0001 | 0.1322 | 0.1322 | 0 |
| 25 | WordNet18 | rgcn | retrain | out | 0.5000 | 0.5000 | embedding_mse | original | soft | 0.9068 | 0.9162 | 0.5970 | 0.5968 | 0.6216 | 1.0472 | 0.1264 | 0.1264 | 707 |

## WordNet18 GNNDelete 对比摘要

### dt_auc 最高（保留/测试性能最好）

| rank | gnn | df | alpha | loss_r_type | operator | gate | value | dt_auc | df_auc | df_aup | df_logit_mean | dt_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rgcn | out | 0.2000 | bce_rank_l2 | relation_lora_moe | soft | 0.9071 | 0.9071 | 0.6859 | 0.7310 | 0.5442 | 1.0065 |
| 2 | rgcn | out | 0.5000 | embedding_mse | relation_lora_moe | soft | 0.9068 | 0.9068 | 0.6817 | 0.7394 | 0.5644 | 1.0172 |
| 3 | rgcn | out | 0.5000 | bce_rank_l2 | relation_lora_moe | soft | 0.9067 | 0.9067 | 0.7098 | 0.7588 | 0.5127 | 1.0111 |
| 4 | rgcn | out | 0.5000 | bce_rank_l2 | relation_lora_moe | hard | 0.9067 | 0.9067 | 0.7075 | 0.7569 | 0.5152 | 1.0107 |
| 5 | rgcn | out | 0.5000 | embedding_mse | scgu | soft | 0.9065 | 0.9065 | 0.6771 | 0.7267 | 0.5596 | 1.0178 |

### dt_loss 最低（保留/测试 loss 最低）

| rank | gnn | df | alpha | loss_r_type | operator | gate | value | dt_auc | df_auc | df_aup | df_logit_mean | dt_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rgat | out | 0.5000 | embedding_mse | relation_lora_moe | soft | 0.4431 | 0.8907 | 0.6041 | 0.5364 | 0.6174 | 0.4431 |
| 2 | rgat | out | 0.2000 | bce_rank_l2 | relation_lora_moe | soft | 0.4439 | 0.8904 | 0.7084 | 0.6308 | 0.5319 | 0.4439 |
| 3 | rgat | out | 0.5000 | embedding_mse | original | soft | 0.4482 | 0.8889 | 0.6198 | 0.5495 | 0.5903 | 0.4482 |
| 4 | rgat | out | 0.5000 | bce_rank_l2 | relation_lora_moe | soft | 0.4486 | 0.8876 | 0.8498 | 0.8172 | 0.3919 | 0.4486 |
| 5 | rgat | out | 0.7500 | rank_l2 | lora_moe | soft | 0.4496 | 0.8872 | 0.8513 | 0.8233 | 0.3966 | 0.4496 |

### df_logit_mean 最低（删除边平均 logit 最低）

| rank | gnn | df | alpha | loss_r_type | operator | gate | value | dt_auc | df_auc | df_aup | df_logit_mean | dt_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rgat | out | 0.7500 | bce_rank_l2 | relation_lora_moe | soft | 0.2972 | 0.8814 | 0.9004 | 0.8845 | 0.2972 | 0.4577 |
| 2 | rgat | random | 0.7500 | bce_l2 | lora_moe | soft | 0.3147 | 0.8080 | 0.7852 | 0.7914 | 0.3147 | 0.5689 |
| 3 | rgat | out | 0.7500 | bce_l2 | lora_moe | soft | 0.3320 | 0.8849 | 0.8892 | 0.8681 | 0.3320 | 0.4526 |
| 4 | rgat | out | 0.5000 | bce_rank_l2 | relation_lora_moe | soft | 0.3919 | 0.8876 | 0.8498 | 0.8172 | 0.3919 | 0.4486 |
| 5 | rgat | out | 0.7500 | rank_l2 | lora_moe | soft | 0.3966 | 0.8872 | 0.8513 | 0.8233 | 0.3966 | 0.4496 |

### df_auc 最低（删除集 AUC 最低）

| rank | gnn | df | alpha | loss_r_type | operator | gate | value | dt_auc | df_auc | df_aup | df_logit_mean | dt_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rgat | out | 0.5000 | embedding_mse | relation_lora_moe | soft | 0.6041 | 0.8907 | 0.6041 | 0.5364 | 0.6174 | 0.4431 |
| 2 | rgat | out | 0.5000 | embedding_mse | original | soft | 0.6198 | 0.8889 | 0.6198 | 0.5495 | 0.5903 | 0.4482 |
| 3 | rgcn | out | 0.5000 | embedding_mse | scgu | soft | 0.6771 | 0.9065 | 0.6771 | 0.7267 | 0.5596 | 1.0178 |
| 4 | rgcn | out | 0.5000 | - | - | - | 0.6772 | 0.9027 | 0.6772 | 0.7434 | 0.5513 | 1.0520 |
| 5 | rgcn | out | 0.5000 | embedding_mse | relation_lora_moe | soft | 0.6817 | 0.9068 | 0.6817 | 0.7394 | 0.5644 | 1.0172 |

### df_aup 最低（删除集 AUP 最低）

| rank | gnn | df | alpha | loss_r_type | operator | gate | value | dt_auc | df_auc | df_aup | df_logit_mean | dt_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rgat | out | 0.5000 | embedding_mse | relation_lora_moe | soft | 0.5364 | 0.8907 | 0.6041 | 0.5364 | 0.6174 | 0.4431 |
| 2 | rgat | out | 0.5000 | embedding_mse | original | soft | 0.5495 | 0.8889 | 0.6198 | 0.5495 | 0.5903 | 0.4482 |
| 3 | rgat | out | 0.2000 | bce_rank_l2 | relation_lora_moe | soft | 0.6308 | 0.8904 | 0.7084 | 0.6308 | 0.5319 | 0.4439 |
| 4 | rgcn | out | 0.5000 | embedding_mse | scgu | soft | 0.7267 | 0.9065 | 0.6771 | 0.7267 | 0.5596 | 1.0178 |
| 5 | rgcn | out | 0.2000 | bce_rank_l2 | relation_lora_moe | soft | 0.7310 | 0.9071 | 0.6859 | 0.7310 | 0.5442 | 1.0065 |

## 简单综合排序（参考）

评分 = 0.5×dt_auc归一化 + 0.3×低df_logit_mean归一化 + 0.2×低dt_loss归一化。仅作快速筛选参考。

| rank | score | gnn | df | alpha | loss_r_type | operator | gate | dt_auc | df_auc | df_aup | df_logit_mean | dt_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8654 | rgat | out | 0.7500 | bce_rank_l2 | relation_lora_moe | soft | 0.8814 | 0.9004 | 0.8845 | 0.2972 | 0.4577 |
| 2 | 0.8521 | rgat | out | 0.7500 | bce_l2 | lora_moe | soft | 0.8849 | 0.8892 | 0.8681 | 0.3320 | 0.4526 |
| 3 | 0.8111 | rgat | out | 0.5000 | bce_rank_l2 | relation_lora_moe | soft | 0.8876 | 0.8498 | 0.8172 | 0.3919 | 0.4486 |
| 4 | 0.8044 | rgat | out | 0.7500 | rank_l2 | lora_moe | soft | 0.8872 | 0.8513 | 0.8233 | 0.3966 | 0.4496 |
| 5 | 0.6957 | rgat | out | 0.2000 | bce_rank_l2 | relation_lora_moe | soft | 0.8904 | 0.7084 | 0.6308 | 0.5319 | 0.4439 |
| 6 | 0.6911 | rgcn | out | 0.7500 | bce_rank_l2 | relation_lora_moe | soft | 0.9060 | 0.7599 | 0.8082 | 0.4191 | 1.0195 |
| 7 | 0.6320 | rgat | out | 0.5000 | embedding_mse | original | soft | 0.8889 | 0.6198 | 0.5495 | 0.5903 | 0.4482 |
| 8 | 0.6260 | rgcn | in | 0.5000 | embedding_mse | relation_lora_moe | hard | 0.9028 | 0.7190 | 0.7422 | 0.4717 | 1.0185 |
| 9 | 0.6176 | rgat | out | 0.5000 | embedding_mse | relation_lora_moe | soft | 0.8907 | 0.6041 | 0.5364 | 0.6174 | 0.4431 |
| 10 | 0.6099 | rgcn | out | 0.5000 | bce_rank_l2 | relation_lora_moe | soft | 0.9067 | 0.7098 | 0.7588 | 0.5127 | 1.0111 |

## 日志路径索引

| # | path |
| --- | --- |
| 1 | `checkpoint/DBLP/rgcn/original/42` |
| 2 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.2-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 3 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.5-non_connected/out-0.5-42` |
| 4 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.5-non_connected-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 5 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.5-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 6 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.75-non_connected-bce_rank_l2-0.5-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 7 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.75-non_connected-bce_l2-0.0-1.0-lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 8 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.75-non_connected-rank_l2-0.0-1.0-lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 9 | `checkpoint/WordNet18/rgat/gnndelete/mse_mean-both_layerwise-0.75-non_connected-bce_l2-0.0-1.0-lora_moe-4-8-16-1.0-0.0-soft-1.0/random-0.5-42` |
| 10 | `checkpoint/WordNet18/rgat/original/13` |
| 11 | `checkpoint/WordNet18/rgat/original/21` |
| 12 | `checkpoint/WordNet18/rgat/original/42` |
| 13 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-relation_lora_moe-4-8-16-1.0-0.0-hard-1.0/in-0.5-42` |
| 14 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.2-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 15 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected/out-0.5-42` |
| 16 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 17 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-scgu-16-random/out-0.5-42` |
| 18 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-bce_rank_l2-0.0-1.0/out-0.5-42` |
| 19 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-hard-1.0/out-0.5-42` |
| 20 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 21 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-bce_l2-0.0-1.0/out-0.5-42` |
| 22 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.5-non_connected-rank_l2-0.0-1.0/out-0.5-42` |
| 23 | `checkpoint/WordNet18/rgcn/gnndelete/mse_mean-both_layerwise-0.75-non_connected-bce_rank_l2-0.0-1.0-relation_lora_moe-4-8-16-1.0-0.0-soft-1.0/out-0.5-42` |
| 24 | `checkpoint/WordNet18/rgcn/original/42` |
| 25 | `checkpoint/WordNet18/rgcn/retrain/out-0.5-42` |

python delete_gnn.py \
    --dataset WordNet18 \
    --gnn rgcn \
    --unlearning_model gnndelete \
    --df out \
    --df_size 0.5 \
    --random_seed 42 \
    --lr 1e-3 \
    --epochs 50 \
    --valid_freq 2 \
    --loss_fct mse_mean \
    --loss_type both_layerwise \
    --alpha 0.0 \
    --neg_sample_random non_connected \
    --loss_r_type bce_rank_l2 \
    --loss_r_margin 0.0 \
    --loss_r_beta 1.0 \
    --deletion_operator relation_lora_moe \
    --del_moe_num_experts 4 \
    --del_lora_rank 8 \
    --del_gate_emb_dim 16 \
    --del_lora_alpha 1.0 \
    --del_lora_dropout 0.0 \
    --del_gate_mode soft \
    --del_gate_temperature 1.0


export WANDB_MODE=offline
  export WANDB_PROJECT=zitniklab-gnn-unlearning

  python delete_gnn.py \
    --dataset WordNet18 \
    --gnn rgat \
    --unlearning_model retrain \
    --df out \
    --df_size 0.5 \
    --random_seed 42 \
    --lr 1e-3 \
    --epochs 2000 \
    --valid_freq 500