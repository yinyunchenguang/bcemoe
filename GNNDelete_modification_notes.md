# GNNDelete 改动讨论记录

本文档用于持续维护我们关于 GNNDelete 的改动、实验观察和后续方向。后续讨论和方案沉淀优先更新到这里。

## 1. 当前目标

原始 GNNDelete 在 KG/异构图场景中主要使用一个共享的 deletion operator，对不同关系类型、不同删除边的结构差异区分不足。我们当前的目标是：

1. 将原来的全局共享删除算子扩展为关系感知或子空间约束的算子；
2. 改进遗忘侧 loss，使被删除三元组的得分更接近负样本或显式下降；
3. 保持局部因果一致性，即非删除邻域节点的表示不要被过度破坏；
4. 在 `WordNet18 + RGCN/RGAT` 上验证不同 deletion operator 和 loss 组合的效果。

---

## 2. 全局 deletion operator 改动

相关代码主要在：

- `framework/models/deletion.py`
- `framework/training_args.py`
- `delete_gnn.py`
- `framework/trainer/gnndelete_nodeemb.py`

### 2.1 原始 GNNDelete 算子

原始 KG deletion layer 是一个共享矩阵：

```python
h' = h W_del
```

其中 `W_del` 初始化为接近 0 的小矩阵：

```python
delection_weight = ones(dim, dim) / 1000
```

问题：

- 所有关系类型共享同一个删除矩阵；
- 初始映射不是 identity，而是近似把局部 embedding 压到 0；
- `loss_l` 需要从接近 0 的表示重新学回原始表示，收敛慢。

### 2.2 SCGU / Low-rank deletion operator

已加入关系感知低秩形式：

```python
W_r = I + A B_r
h' = h W_r
```

特点：

- `A` 是共享低秩子空间；
- `B_r` 是关系相关参数；
- 删除边包含多个 relation 时，对对应 `B_r` 做频率加权平均；
- 初始结构包含 identity，因此更利于保持 locality。

命令参数：

```bash
--deletion_operator scgu
--scgu_rank 16
--scgu_init random
```

### 2.3 Relation LoRA-MoE deletion operator

已加入关系驱动的 LoRA-MoE 删除层：

```python
DEL_r(h) = h W_base + scale * sum_k g_k(r) * h A^T B_k^T
```

其中：

- `A` 为共享 LoRA down projection；
- `B_k` 为 expert-specific up projection；
- relation embedding 经过 gate 得到专家权重；
- 支持 soft gate 和 straight-through hard gate。

命令参数：

```bash
--deletion_operator relation_lora_moe
--del_moe_num_experts 4
--del_lora_rank 8
--del_gate_emb_dim 16
--del_lora_alpha 1.0
--del_lora_dropout 0.0
--del_gate_mode soft
--del_gate_temperature 1.0
```

### 2.4 残差 LoRA-MoE 尝试与回退

我们曾尝试将 Relation LoRA-MoE 改成残差形式，以缓解 `loss_l` 收敛慢的问题。动机是旧版 LoRA-MoE 不是 residual adapter，而是用一个接近 0 的 base projection 加 LoRA 去重建原 embedding：

```python
base = h @ W_base
h' = base + scale * LoRA(h)
```

由于：

- `W_base` 初始化为 `0.001`；
- `LoRA_B` 初始化为 0；
- `scale = del_lora_alpha / rank`，默认是 `1 / 8`；

训练初期近似：

```python
h' ≈ h @ 0.001
```

这会让 locality target 和当前表示差距过大，理论上可能导致 `loss_l` 慢。

尝试过的残差形式是：

```python
base_delta = h @ W_base
h' = h + base_delta + scale * LoRA(h)
```

但实验发现该残差版本效果不理想，因此目前已回退到原 LoRA-MoE 写法：

```python
base = h @ W_base
h' = base + scale * LoRA(h)
```

当前结论：

- `loss_l` 慢不一定只由非残差结构导致；
- 残差 identity 可能让局部保持更容易，但也可能削弱删除算子的遗忘强度；
- 后续若继续研究残差方向，应作为单独消融，而不是默认方案；
- 当前主线先回到非残差 LoRA-MoE，优先从 loss 权重、LoRA scaling、gate、诊断日志等方向排查。

---

## 3. Loss 改动

### 3.1 原始 GNNDelete loss

整体结构：

```python
loss = alpha * loss_r + (1 - alpha) * loss_l
```

其中：

- `loss_r`：遗忘/随机性目标，使 deleted edge 表示接近负样本；
- `loss_l`：局部保持目标，使非删除局部节点表示接近原模型。

### 3.2 Locality loss：`loss_l`

KG node embedding 分支中目前使用：

```python
loss_l1 = loss_fct(z1[local_non_df], z1_ori[local_non_df])
loss_l2 = loss_fct(z2[local_non_df], z2_ori[local_non_df])
loss_l = loss_l1 + loss_l2
```

其中 `local_non_df` 排除了删除边端点，避免 deleted nodes 直接约束回原始表示。

当前主要问题：

- 如果 deletion operator 初始不是 identity，`loss_l` 会非常大；
- 当 `alpha` 较大时，例如 `alpha=0.75`，`loss_l` 权重只有 0.25，收敛会更慢；
- `loss_r` 与 `loss_l` 存在天然冲突：删除边相关节点需要改变，但邻域保持又限制改变。

### 3.3 遗忘侧 loss：`loss_r_type`

已加入多种 KG 遗忘侧目标：

```bash
--loss_r_type embedding_mse
--loss_r_type preference
--loss_r_type preference_l2
--loss_r_type bce_l2
--loss_r_type rank_l2
--loss_r_type bce_rank_l2
--loss_r_type dist_l2
--loss_r_type sorted_dist_l2
```

重点关注：

#### bce_l2

对 deleted triples 的 DistMult logit 做 BCE，目标为负类：

```python
BCEWithLogits(deleted_logits, zeros)
```

优点：直接压低删除三元组得分。  
风险：如果权重过大，可能破坏局部表示，使 `loss_l` 上升。

#### rank_l2

让 deleted triples 的得分低于 retained triples 或负样本得分：

```python
softplus(beta * (df_logit - dr_logit + margin))
```

优点：更符合排序指标。  
风险：单独使用时可能没有 BCE 稳定。

#### bce_rank_l2

组合 BCE 和 ranking：

```python
loss_r2 = bce_loss + rank_loss
```

这是目前较值得继续调参的方向。

相关超参：

```bash
--loss_r_margin 0.0
--loss_r_beta 1.0
```

---

## 4. 当前实验观察

基于已有日志，LoRA-MoE 在非残差版本下，`loss_l` 初期较高，例如可达到 20 左右，之后下降但速度受 `alpha` 和 operator 初始化影响明显。

观察到的规律：

1. `alpha` 越大，越偏向遗忘侧，`loss_l` 越慢；
2. 原 LoRA-MoE 因为初始近似零映射，`loss_l` 被迫先恢复 identity；
3. hard gate 与 soft gate 在早期差异不明显，说明 gate 可能不是当前最主要瓶颈；
4. `bce_rank_l2` 能更直接推动删除边得分下降，但也更容易和 locality 目标冲突。

---

## 5. 建议的下一步修改方向

### 5.1 回到非残差 LoRA-MoE 主线

残差 LoRA-MoE 已尝试但效果不理想，当前代码已回退。下一步建议先在非残差版本上继续做系统排查。

推荐命令方向：

```bash
python delete_gnn.py \
  --dataset WordNet18 \
  --gnn rgcn \
  --unlearning_model gnndelete \
  --df out \
  --df_size 0.5 \
  --loss_type both_layerwise \
  --loss_fct mse_mean \
  --loss_r_type bce_rank_l2 \
  --alpha 0.5 \
  --deletion_operator relation_lora_moe \
  --del_lora_rank 8 \
  --del_lora_alpha 8 \
  --del_gate_mode soft
```

重点看：

- `loss_l` 和 `loss_r` 是否同时下降；
- `loss_r2_bce` 与 `loss_r2_rank` 哪个主导；
- `df_auc` 是否真正下降；
- `dt_auc` 是否被明显破坏；
- LoRA delta 是否过小或过大。


### 5.2 调整 LoRA scaling

当前默认：

```python
scale = del_lora_alpha / rank = 1 / 8
```

建议尝试：

```bash
--del_lora_alpha 8
--del_lora_alpha 16
```

如果 LoRA 更新过弱，可以增大 `del_lora_alpha`；如果训练震荡，则降低学习率或减小 `del_lora_alpha`。

### 5.3 调整 loss 权重

建议按以下顺序：

```bash
--alpha 0.2
--alpha 0.5
--alpha 0.75
```

解释：

- `alpha=0.2`：优先 locality，适合检查保持能力上限；
- `alpha=0.5`：遗忘和保持均衡；
- `alpha=0.75`：优先遗忘，可能牺牲 `loss_l`。

### 5.4 残差 LoRA 作为后续消融，不作为当前默认

残差版本已经尝试并回退。后续如果还要研究，可以做更严格的消融，例如：

```python
h' = h + scale * LoRA(h)
```

或引入可学习门控：

```python
h' = h + gamma * scale * LoRA(h)
```

但当前不建议把残差形式作为主线，除非能同时证明：

- `loss_l` 改善；
- `df_auc` 不变差；
- 删除边 logit 确实下降；
- `dt_auc` 不被破坏。


### 5.5 对 gate 做正则或温度调节

如果 expert 使用不充分，可以尝试：

```bash
--del_gate_temperature 0.5
--del_gate_temperature 2.0
--del_gate_mode hard
```

可能的后续正则：

- expert load balancing；
- gate entropy regularization；
- relation-wise expert diversity loss。

### 5.6 增加诊断日志

建议在训练日志中额外记录：

- `loss_l1`, `loss_l2`；
- `loss_r1`, `loss_r2`；
- `loss_r2_bce`, `loss_r2_rank`；
- LoRA delta norm：`||scale * delta||`；
- base delta norm：`||h @ W_base||`；
- gate 分布和 expert 使用率。

这样可以区分：

- 是 LoRA 更新太小；
- 还是 loss 权重冲突；
- 还是 gate 没学到关系差异；
- 还是 deletion subgraph 太大导致 locality 约束过强。

---

## 6. 当前待办清单

- [x] 添加 SCGU / low-rank relation-aware deletion operator；
- [x] 添加 Relation LoRA-MoE deletion operator；
- [x] 添加 KG 遗忘侧 `bce_l2`、`rank_l2`、`bce_rank_l2` 等 loss；
- [x] 尝试将 Relation LoRA-MoE 改为残差形式；
- [x] 因效果不理想，已将残差 LoRA-MoE 回退为原非残差形式；
- [ ] 比较 `del_lora_alpha=1/8/16`；
- [ ] 比较 `alpha=0.2/0.5/0.75`；
- [ ] 如有必要，再做纯 LoRA residual 消融，移除 `base_delta`；
- [ ] 增加 LoRA/gate/loss 分量的诊断日志；
- [ ] 汇总 RGCN 与 RGAT 上的 `dt_auc/df_auc/loss_l/loss_r` 表格。

