"""
Evaluate MI attack metrics on an already-trained unlearning checkpoint.
Loads model_best.pt + attack models, runs test(), saves updated trainer_log.json.

Usage (same args as delete_gnn.py):
  python eval_mi.py --dataset WordNet18 --gnn rgcn --unlearning_model gnndelete \
      --df out --df_size 0.5 --alpha 0.5 --random_seed 42
"""

import os
import copy
import json
import pickle
import torch
from torch_geometric.utils import to_undirected, k_hop_subgraph, is_undirected
from torch_geometric.seed import seed_everything

from framework import get_model, get_trainer
from framework.training_args import parse_args

try:
    from train_mi import MLPAttacker
except ImportError:
    MLPAttacker = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    seed_everything(args.random_seed)

    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    attack_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_all', str(args.random_seed))
    attack_path_sub = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_sub', str(args.random_seed))

    # Build checkpoint_dir (same logic as delete_gnn.py)
    if 'gnndelete' in args.unlearning_model:
        loss_tag = [args.loss_fct, args.loss_type, args.alpha, args.neg_sample_random]
        if getattr(args, 'loss_r_type', 'embedding_mse') != 'embedding_mse':
            loss_tag.extend([args.loss_r_type, args.loss_r_margin, args.loss_r_beta])
        if getattr(args, 'deletion_operator', 'original') != 'original':
            loss_tag.append(args.deletion_operator)
            if args.deletion_operator in ['scgu', 'scgu_lowrank', 'lowrank']:
                loss_tag.extend([args.scgu_rank, args.scgu_init])
            elif args.deletion_operator == 'scgu_moe':
                loss_tag.extend([args.scgu_rank, args.scgu_init, args.del_moe_num_experts, args.del_gate_emb_dim, args.del_gate_mode])
            else:
                loss_tag.extend([args.del_moe_num_experts, args.del_lora_rank, args.del_gate_emb_dim,
                                  args.del_lora_alpha, args.del_lora_dropout, args.del_gate_mode, args.del_gate_temperature])
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model,
            '-'.join([str(i) for i in loss_tag]),
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    else:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model,
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))

    model_ckpt_path = os.path.join(args.checkpoint_dir, 'model_best.pt')
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f'No checkpoint found at {model_ckpt_path}')
    print(f'Checkpoint: {args.checkpoint_dir}')

    # Load dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)

    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = dataset.num_features

    # Rebuild df/dr masks (must match what was used during training)
    assert args.df != 'none'
    if args.df_size >= 100:
        df_size = int(args.df_size)
    else:
        df_size = int(args.df_size / 100 * data.train_pos_edge_index.shape[1])

    df_masks = torch.load(os.path.join(args.data_dir, args.dataset, f'df_{args.random_seed}.pt'))
    if args.df in df_masks:
        df_mask_all = df_masks[args.df]
    elif args.df == 'random':
        df_mask_all = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    else:
        raise KeyError(f"Unknown --df '{args.df}'")

    df_nonzero = df_mask_all.nonzero().squeeze()
    torch.manual_seed(args.random_seed)
    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False
    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    if args.gnn in ['rgcn', 'rgat']:
        data.directed_df_edge_type = data.train_edge_type[df_mask]

    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        2, data.train_pos_edge_index, num_nodes=data.num_nodes)
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        1, data.train_pos_edge_index, num_nodes=data.num_nodes)

    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop

    if args.gnn in ['rgcn', 'rgat']:
        r, c = data.train_pos_edge_index
        rev_edge_index = torch.stack([c, r], dim=0)
        rev_edge_type = data.train_edge_type + args.num_edge_type
        data.edge_index = torch.cat((data.train_pos_edge_index, rev_edge_index), dim=1)
        data.edge_type = torch.cat([data.train_edge_type, rev_edge_type], dim=0)
        two_hop_mask = two_hop_mask.repeat(2).view(-1)
        df_mask = df_mask.repeat(2).view(-1)
        dr_mask = dr_mask.repeat(2).view(-1)
    else:
        train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(
            data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
        two_hop_mask = two_hop_mask.bool()
        df_mask = df_mask.bool()
        dr_mask = ~df_mask
        data.train_pos_edge_index = train_pos_edge_index
        data.edge_index = train_pos_edge_index

    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask

    # Load model
    model = get_model(args, sdf_node_1hop, sdf_node_2hop,
                      num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    ckpt = torch.load(model_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model = model.to(device)

    # Load retrain model if available
    retrain_path = os.path.join(
        'checkpoint', args.dataset, args.gnn, 'retrain',
        '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]),
        'model_best.pt')
    if os.path.exists(retrain_path):
        retrain_args = copy.deepcopy(args)
        retrain_args.unlearning_model = 'retrain'
        retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
        retrain.load_state_dict(torch.load(retrain_path, map_location=device)['model_state'])
        retrain = retrain.to(device).eval()
    else:
        retrain = None

    # Load attack models
    attack_model_all = attack_model_sub = None
    if MLPAttacker is not None:
        ckpt_all = os.path.join(attack_path_all, 'attack_model_best.pt')
        ckpt_sub = os.path.join(attack_path_sub, 'attack_model_best.pt')
        if os.path.exists(ckpt_all):
            attack_model_all = MLPAttacker()
            attack_model_all.load_state_dict(torch.load(ckpt_all, map_location=device)['model_state'])
            attack_model_all = attack_model_all.to(device)
            print(f'Loaded attack model (all) from {ckpt_all}')
        if os.path.exists(ckpt_sub):
            attack_model_sub = MLPAttacker()
            attack_model_sub.load_state_dict(torch.load(ckpt_sub, map_location=device)['model_state'])
            attack_model_sub = attack_model_sub.to(device)
            print(f'Loaded attack model (sub) from {ckpt_sub}')

    if attack_model_all is None and attack_model_sub is None:
        print('No attack models found. Run train_mi.py first.')
        return

    # Load existing trainer_log to preserve training metrics
    log_path = os.path.join(args.checkpoint_dir, 'trainer_log.json')
    trainer = get_trainer(args)
    if os.path.exists(log_path):
        with open(log_path) as f:
            trainer.trainer_log = json.load(f)

    # Compute "before" MI logits from the original (pre-unlearning) model
    # if they're not already stored in the trainer_log
    from framework.evaluation import member_infer_attack
    if 'mi_logit_all_before' not in trainer.trainer_log:
        original_model_path = os.path.join(original_path, 'model_best.pt')
        if os.path.exists(original_model_path):
            orig_args = copy.deepcopy(args)
            orig_args.unlearning_model = 'original'
            original_model = get_model(orig_args, sdf_node_1hop, sdf_node_2hop,
                                       num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
            original_model.load_state_dict(
                torch.load(original_model_path, map_location=device)['model_state'], strict=False)
            original_model = original_model.to(device).eval()
            print(f'Computing before-unlearning MI logits from {original_model_path}')
            if attack_model_all is not None:
                mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(
                    original_model, attack_model_all, data)
                trainer.trainer_log['mi_logit_all_before'] = mi_logit_all_before
                trainer.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
                print(f'  mi_sucrate_all_before: {mi_sucrate_all_before:.4f}')
            if attack_model_sub is not None:
                mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(
                    original_model, attack_model_sub, data)
                trainer.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
                trainer.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before
                print(f'  mi_sucrate_sub_before: {mi_sucrate_sub_before:.4f}')
            del original_model
        else:
            print(f'Warning: original model not found at {original_model_path}, skipping before logits')

    # Run test with attack models
    test_results = trainer.test(model, data, model_retrain=retrain,
                                attack_model_all=attack_model_all,
                                attack_model_sub=attack_model_sub)

    # Print MI results
    log = trainer.trainer_log
    print('\n=== MI Attack Results ===')
    for k in ['mi_sucrate_all_before', 'mi_sucrate_all_after', 'mi_ratio_all',
              'mi_sucrate_sub_before', 'mi_sucrate_sub_after', 'mi_ratio_sub']:
        v = log.get(k)
        if v is not None:
            print(f'  {k}: {v:.4f}')
    print(f'  dt_auc: {log.get("dt_auc", float("nan")):.4f}')
    print(f'  df_auc: {log.get("df_auc", float("nan")):.4f}')

    trainer.save_log()
    print(f'\nSaved updated trainer_log to {log_path}')


if __name__ == '__main__':
    main()
