 python delete_gnn.py \
    --unlearning_model gnndelete_nodeemb \
    --gnn rgcn \
    --dataset DBLP \
    --deletion_operator scgu_moe \
    --loss_r_type bce_rank_l2 \
    --scgu_rank 16 \
    --scgu_init random \
    --del_moe_num_experts 4 \
    --del_gate_emb_dim 16 \
    --del_gate_mode soft \
    --alpha 0.5 \
    --df in \
    --df_size 0.5 \
    --rank_same_relation\
    --random_seed 42 \
    --epochs 50

     数据集    │ out 候选数 │ in 候选数 │                 建议                  │
  ├─────────────┼────────────┼───────────┼───────────────────────────────────────┤
  │ WordNet18   │ 4,631      │ 136,811   │ 两者都可，out 候选少注意 df_size 别超 │
  ├─────────────┼────────────┼───────────┼───────────────────────────────────────┤
  │ WordNet18RR │ 12,769     │ 74,066    │ 两者都可                              │
  ├─────────────┼────────────┼───────────┼───────────────────────────────────────┤
  │ FB15k-237   │ 1,254      │ 270,861   │ out 候选极少，建议用 in 或 random     │
  ├─────────────┼────────────┼───────────┼───────────────────────────────────────┤
  │ YAGO3-10    │ 245,544    │ 833,496   │ 两者都可                              │
  ├─────────────┼────────────┼───────────┼───────────────────────────────────────┤
  │ codex-s     │ 1,765      │ 31,123    │ out 候选少，建议用 in 或 random       │
  ├─────────────┼────────────┼───────────┼───────────────────────────────────────┤
  │ codex-m     │ 8,360      │ 177,224   │ 两者都可                              │
  ├─────────────┼────────────┼───────────┼───────────────────────────────────────┤
  │ codex-l     │ 38,686     │ 512,507   │ 两者都可           
wordnet18
dblp


  python delete_gnn.py --unlearning_model gnndelete_nodeemb --gnn rgat --dataset FB15k-237 --deletion_operator scgu --scgu_rank 16 --scgu_init random --df in --df_size 0.5 --random_seed 42 --alpha 0.95

  python delete_gnn.py --dataset WordNet18  --gnn rgcn --unlearning_model gnndelete \
    --df in --df_size 0.5 --alpha 0.5 --random_seed 42 \
    --loss_fct mse_mean --loss_type both_layerwise --neg_sample_random non_connected \
    --batch_size 4096 --num_steps 4