#!/usr/bin/env python3
"""Bridge-ceiling probe: is a hard-AND-safe real negative even reachable?

Two independent questions, both cheap (paired M0 + cached flow, no training):

  (1) CEILING (flow-independent): for each user, over all real items, what is the
      highest win-rate W achievable by a SAFE item (not a train/test positive)?
      And do hard items (W>0.5) tend to be false negatives (test positives)?
      -> tells us whether the target the bridge is supposed to hit exists at all.

  (2) BRIDGE EFFICIENCY (flow-dependent): given the flow's continuous x_gen with
      win-rate W_cont, how much survives each mapping strategy, and at what FN cost?
      cosine-nearest / dot-nearest / score_topk / boundary_topk.

Usage:
  python scripts/probe_bridge.py [--experiment S1_flow_vae] [--dataset mind]
      [--ceiling-users 300] [--bridge-users 2000] [--pos-cap 20]
"""
import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pilot_runner import (
    _build_recbole_config_dict,
    _extract_flow_config,
    _load_experiments,
    _load_m0_checkpoint_path,
    _resolve_experiment,
)
from src.flowns_trainer import FLOW_CONFIG_DEFAULTS
from src.flow_model import ConditionalFlowModel
from src.neg_sampling import EmbeddingToItemMapper
from src.recbole_utils import (
    build_trainer,
    get_embeddings,
    get_user_positive_items,
    setup_recbole,
)
from src.reward import BoundaryAwareReward
from src.sde_sampler import SDESampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('probe_bridge')


def _recbole_overrides(merged):
    """Same control-key filtering pilot_runner uses to split flow vs recbole keys."""
    shared = {'eval_step', 'stopping_step'}
    control = (set(FLOW_CONFIG_DEFAULTS) - shared) | {
        'description', 'experiment', 'phases', 'use_paired_m0', 'evaluate_test',
        'quality_check', 'quality_sample_users', 'diagnostic_sample_users',
        'save_rec_checkpoint_pointer', 'save_flow_model',
    }
    return {k: v for k, v in merged.items() if k not in control}


def _win_rate(cand_scores, pos_scores):
    """cand_scores (..., M), pos_scores (P,) -> W (..., M) averaged over positives."""
    return torch.sigmoid(cand_scores.unsqueeze(-1) - pos_scores).mean(dim=-1)


def _quantiles(x):
    x = x.float()
    q = torch.tensor([0.5, 0.9, 0.99], device=x.device)
    vals = torch.quantile(x, q)
    return float(x.mean()), float(vals[0]), float(vals[1]), float(vals[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', default='S1_flow_vae')
    ap.add_argument('--dataset', default='mind')
    ap.add_argument('--ceiling-users', type=int, default=300)
    ap.add_argument('--bridge-users', type=int, default=2000)
    ap.add_argument('--pos-cap', type=int, default=20)
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--seed', type=int, default=2020)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    recbole_base, _, _ = _load_experiments()
    merged = _resolve_experiment(args.experiment)
    model_name = merged.get('model', 'LightGCN')
    config_dict = _build_recbole_config_dict(
        recbole_base, args.dataset, _recbole_overrides(merged), seed=args.seed,
    )
    config_dict['model'] = model_name

    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        model_name, args.dataset, config_dict=config_dict,
    )
    device = config['device']

    # Load paired M0 the same way FlowNSTrainer.phase1 does.
    flow_config = _extract_flow_config(merged, {})
    ckpt_path = _load_m0_checkpoint_path(args.dataset, model_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.load_other_parameter(ckpt.get('other_parameter'))
    model.eval()
    logger.info('Loaded paired M0: %s', ckpt_path)

    user_emb, item_emb = get_embeddings(model)
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)
    n_items, d = item_emb.shape
    logger.info('Scoring space: n_items=%d, emb_dim=%d', n_items, d)

    # Cached pretrained flow (no GRPO; the ceiling part doesn't use it anyway).
    flow = ConditionalFlowModel(
        emb_dim=d, hidden_dim=merged.get('flow_hidden_dim', 256),
        n_layers=merged.get('flow_n_layers', 3), device=str(device),
    )
    flow_path = flow_config.get('flow_checkpoint_path')
    import os
    proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    flow_path = flow_path if os.path.isabs(flow_path) else os.path.join(proj, flow_path)
    flow.velocity_net.load_state_dict(
        torch.load(flow_path, map_location=device, weights_only=False),
        strict=False,  # tolerate the CFG null-condition key being absent/present
    )
    flow.velocity_net.eval()
    logger.info('Loaded cached flow: %s', flow_path)
    sde = SDESampler(
        flow.velocity_net, n_steps=merged.get('sde_steps', 20),
        eta=merged.get('eta', 0.5), delta=merged.get('delta', 0.01),
        guidance_scale=merged.get('cfg_guidance_scale', 1.0),
    )

    train_pos = get_user_positive_items(train_data)
    test_pos = get_user_positive_items(test_data)
    valid_pos = get_user_positive_items(valid_data)
    uids = [u for u in train_pos if u < user_emb.shape[0] and train_pos[u]]
    perm = torch.randperm(len(uids))

    # ---------- (1) CEILING: flow-independent ----------
    ceil_uids = [uids[int(i)] for i in perm[:args.ceiling_users]]
    any_hard_safe = 0
    ceil_safe, ceil_all, hard_count, hard_fn_count = [], [], 0, 0
    chunk = 8192
    for uid in ceil_uids:
        u = user_emb[uid]
        item_scores = item_emb @ u  # (n_items,)
        pos_ids = list(train_pos[uid])[:args.pos_cap]
        pos_scores = item_scores[torch.as_tensor(pos_ids, device=device)]
        # W over ALL items
        W_all = torch.empty(n_items, device=device)
        for s in range(0, n_items, chunk):
            e = min(s + chunk, n_items)
            W_all[s:e] = _win_rate(item_scores[s:e], pos_scores)
        forbidden = train_pos.get(uid, set())
        tpos = test_pos.get(uid, set()) | valid_pos.get(uid, set())
        is_train_pos = torch.zeros(n_items, dtype=torch.bool, device=device)
        if forbidden:
            is_train_pos[torch.as_tensor(list(forbidden), device=device)] = True
        is_test_pos = torch.zeros(n_items, dtype=torch.bool, device=device)
        if tpos:
            is_test_pos[torch.as_tensor(list(tpos), device=device)] = True
        is_train_pos[0] = True  # pad item

        safe = ~is_train_pos & ~is_test_pos
        cand = ~is_train_pos  # candidate negatives = anything not a train positive
        # ceiling among SAFE items vs among ALL candidate negatives (incl. FN)
        ceil_safe.append(float(W_all[safe].max()))
        ceil_all.append(float(W_all[cand].max()))
        if float(W_all[safe].max()) > 0.5:
            any_hard_safe += 1
        # among candidate negatives that are HARD (W>0.5): what frac are test-pos (FN)?
        hard = cand & (W_all > 0.5)
        hc = int(hard.sum())
        hard_count += hc
        hard_fn_count += int((hard & is_test_pos).sum())

    cs_mean, cs_med, cs_p90, cs_p99 = _quantiles(torch.tensor(ceil_safe))
    ca_mean, ca_med, ca_p90, ca_p99 = _quantiles(torch.tensor(ceil_all))
    print('\n' + '=' * 72)
    print(f'(1) CEILING over real items  [users={len(ceil_uids)}, pos_cap={args.pos_cap}]')
    print('=' * 72)
    print(f'  P(exists a SAFE item with W>0.5)        = {any_hard_safe/len(ceil_uids):.3f}')
    print(f'  max SAFE W per user   : mean={cs_mean:.3f} med={cs_med:.3f} p90={cs_p90:.3f} p99={cs_p99:.3f}')
    print(f'  max NEG  W per user   : mean={ca_mean:.3f} med={ca_med:.3f} p90={ca_p90:.3f} p99={ca_p99:.3f}')
    print(f'    (NEG includes false negatives; gap vs SAFE = hardness only reachable via FN)')
    frac_fn = hard_fn_count / max(hard_count, 1)
    print(f'  among HARD negatives (W>0.5): {hard_count} items, {frac_fn:.3f} are test/valid positives (FN)')
    print('  READING: low SAFE ceiling AND high FN-frac => hard real negatives are')
    print('           essentially false negatives; the bridge target is empty. ')

    # ---------- (2) BRIDGE EFFICIENCY: flow-dependent ----------
    br_uids = [uids[int(i)] for i in perm[:args.bridge_users]]
    bu = torch.as_tensor(br_uids, device=device)
    u_emb = user_emb[bu]
    with torch.no_grad():
        x_gen, _, _ = sde.sample_trajectories(u_emb, n_trajectories=1)
        x_gen = x_gen.squeeze(1)  # (B, d)

    # W_cont
    gen_scores = (u_emb * x_gen).sum(-1)  # (B,)
    reward_fn = BoundaryAwareReward(a=1.0, gamma=1.0)
    # precompute per-user padded positives + mask
    K = args.pos_cap
    pos_embs = torch.zeros(len(br_uids), K, d, device=device)
    pos_mask = torch.zeros(len(br_uids), K, dtype=torch.bool, device=device)
    for r, uid in enumerate(br_uids):
        pids = list(train_pos[uid])[:K]
        pos_embs[r, :len(pids)] = item_emb[torch.as_tensor(pids, device=device)]
        pos_mask[r, :len(pids)] = True
    W_cont = reward_fn.win_rate(u_emb, x_gen, pos_embs, pos_mask)

    forbidden = [train_pos.get(uid, set()) for uid in br_uids]
    pos_b = pos_embs.unsqueeze(1)  # (B,1,K,d)
    mask_b = pos_mask.unsqueeze(1)
    ue_b = u_emb.unsqueeze(1)

    def map_and_eval(strategy, metric, **kw):
        mapper = EmbeddingToItemMapper(item_emb, metric=metric)
        ids = mapper.map_to_items(
            x_gen, forbidden_item_ids=forbidden, exclude_item_ids=(0,),
            user_emb=ue_b if strategy != 'nearest' else None,
            pos_item_embs=pos_b if strategy in ('reward_topk', 'boundary_topk') else None,
            pos_mask=mask_b if strategy in ('reward_topk', 'boundary_topk') else None,
            reward_fn=reward_fn, strategy=strategy, candidate_topk=args.topk, **kw,
        )
        ids = ids.reshape(-1)
        W = reward_fn.win_rate(u_emb, item_emb[ids], pos_embs, pos_mask)
        fn = sum(
            1 for r, uid in enumerate(br_uids)
            if int(ids[r]) in (test_pos.get(uid, set()) | valid_pos.get(uid, set()))
        ) / len(br_uids)
        return W, fn

    rows = [('W_cont (no mapping)', W_cont, float('nan'))]
    for label, strat, metric, kw in [
        ('cosine-nearest', 'nearest', 'cosine', {}),
        ('dot-nearest', 'nearest', 'dot', {}),
        ('score_topk (dot)', 'score_topk', 'dot', {}),
        ('boundary_topk safe=.5', 'boundary_topk', 'dot', {'boundary_safe_w': 0.5}),
    ]:
        W, fn = map_and_eval(strat, metric, **kw)
        rows.append((label, W, fn))

    print('\n' + '=' * 72)
    print(f'(2) BRIDGE EFFICIENCY  [users={len(br_uids)}, topk={args.topk}, pre-GRPO flow]')
    print('=' * 72)
    print(f'  {"strategy":24s} {"W_mean":>8s} {"W_med":>8s} {"W>0.5":>7s} {"FN":>7s}')
    for label, W, fn in rows:
        med = float(W.median())
        hi = float((W > 0.5).float().mean())
        fn_s = '   -   ' if fn != fn else f'{fn:7.3f}'
        print(f'  {label:24s} {float(W.mean()):8.3f} {med:8.3f} {hi:7.3f} {fn_s}')
    print('  READING: if dot >> cosine, the bridge just used the wrong metric.')
    print('           if every safe strategy stays ~0 while score_topk is high but')
    print('           high-FN, hardness on the real-item manifold requires FN.')


if __name__ == '__main__':
    main()
