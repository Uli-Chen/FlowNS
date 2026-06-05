"""M0 Sanity checks: environment, RecBole, flow model, reward, end-to-end."""
import sys
import os
import traceback
import numpy as np

# Monkey-patch numpy for RecBole compatibility with NumPy 2.0+
for _attr, _replacement in [
    ('bool_', bool), ('int_', int), ('float_', float),
    ('complex_', complex), ('object_', object), ('str_', str),
    ('bool', bool), ('int', int), ('float', float),
    ('complex', complex), ('object', object), ('str', str),
    ('long', int), ('unicode', str), ('unicode_', str),
]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _replacement)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RecBole')))

PASS = 0
FAIL = 0


def check(name, fn):
    global PASS, FAIL
    print(f'\n{"="*60}')
    print(f'  CHECK: {name}')
    print(f'{"="*60}')
    try:
        fn()
        PASS += 1
        print(f'  [PASS] {name}')
    except Exception as e:
        FAIL += 1
        print(f'  [FAIL] {name}: {e}')
        traceback.print_exc()


def test_imports():
    import torch
    import numpy
    import recbole
    from recbole.config import Config
    from recbole.data.utils import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from src.flow_model import ConditionalFlowModel
    from src.sde_sampler import SDESampler
    from src.reward import BoundaryAwareReward
    from src.grpo import GRPOTrainer
    from src.neg_sampling import EmbeddingToItemMapper
    from src.flowns_trainer import FlowNSTrainer
    from src.custom_metrics import compute_fn_rate
    print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')


def test_recbole_ml100k():
    from src.recbole_utils import setup_recbole, train_recbole, evaluate_recbole
    config_dict = {
        'epochs': 5,
        'eval_step': 5,
        'stopping_step': 5,
        'data_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RecBole', 'dataset')),
        'show_progress': False,
    }
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'ml-100k', config_dict=config_dict,
    )
    trainer, best_score, best_result = train_recbole(config, model, train_data, valid_data)
    print(f'  best_valid_score={best_score:.4f}')
    assert best_score > 0, 'Valid score should be positive'

    result = evaluate_recbole(trainer, test_data)
    print(f'  test_result={result}')


def test_embedding_extraction():
    import torch
    from src.recbole_utils import setup_recbole, get_embeddings
    config_dict = {
        'epochs': 1,
        'eval_step': 1,
        'stopping_step': 1,
        'data_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RecBole', 'dataset')),
        'show_progress': False,
    }
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'ml-100k', config_dict=config_dict,
    )
    user_emb, item_emb = get_embeddings(model)
    print(f'  user_emb: {user_emb.shape}, item_emb: {item_emb.shape}')
    assert user_emb.dim() == 2
    assert item_emb.dim() == 2
    assert user_emb.shape[1] == config['embedding_size']


def test_flow_model():
    import torch
    from src.flow_model import ConditionalFlowModel

    d = 64
    B = 32
    flow = ConditionalFlowModel(emb_dim=d, hidden_dim=128, n_layers=3, device='cpu')

    user_emb = torch.randn(B, d)
    neg_emb = torch.randn(B, d)

    losses = []
    optimizer = torch.optim.Adam(flow.velocity_net.parameters(), lr=1e-3)
    for i in range(50):
        optimizer.zero_grad()
        loss = flow.cfm_loss(user_emb, neg_emb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f'  CFM loss: {losses[0]:.4f} -> {losses[-1]:.4f}')
    assert losses[-1] < losses[0], 'CFM loss should decrease'

    flow.save_as_reference()
    ref_net = flow.create_ref_net()
    print(f'  Reference net created successfully')


def test_sde_sampler():
    import torch
    from src.flow_model import ConditionalVelocityNet
    from src.sde_sampler import SDESampler

    d = 64
    B = 4
    G = 8
    net = ConditionalVelocityNet(d, hidden_dim=128)
    sampler = SDESampler(net, n_steps=10, eta=0.5, delta=0.01)

    user_emb = torch.randn(B, d)
    final_emb, trajectories, noises = sampler.sample_trajectories(user_emb, n_trajectories=G)

    print(f'  final_emb: {final_emb.shape}')
    assert final_emb.shape == (B, G, d)
    assert len(trajectories) == 11  # n_steps + 1
    assert len(noises) == 10
    print(f'  trajectories: {len(trajectories)} steps, noises: {len(noises)} steps')


def test_reward():
    import torch
    from src.reward import BoundaryAwareReward

    reward_fn = BoundaryAwareReward(a=1.0, gamma=1.0)
    print(f'  {reward_fn}')

    assert abs(reward_fn.optimal_w - 0.5) < 1e-6, f'W* should be 0.5, got {reward_fn.optimal_w}'
    assert abs(reward_fn.r_max - 0.25) < 1e-6, f'R_max should be 0.25, got {reward_fn.r_max}'

    W = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    R = reward_fn.compute_reward(W)
    print(f'  W={W.tolist()}, R={R.tolist()}')
    # R should peak at W=0.5 for a=1, γ=1
    assert R[2] > R[0] and R[2] > R[4], 'Reward should peak at W=0.5'

    # Asymmetric case: a=1, γ=2 → W*=1/3
    reward_asym = BoundaryAwareReward(a=1.0, gamma=2.0)
    assert abs(reward_asym.optimal_w - 1.0/3.0) < 1e-6
    print(f'  Asymmetric: {reward_asym}')


def test_neg_sampling():
    import torch
    from src.neg_sampling import EmbeddingToItemMapper

    n_items = 100
    d = 64
    item_emb = torch.randn(n_items, d)
    mapper = EmbeddingToItemMapper(item_emb)

    gen_emb = item_emb[42:43]  # should map back to item 42
    item_ids = mapper.map_to_items(gen_emb)
    print(f'  Exact match test: generated from item 42, mapped to {item_ids.item()}')
    assert item_ids.item() == 42

    gen_batch = torch.randn(16, d)
    ids = mapper.map_to_items(gen_batch)
    print(f'  Batch mapping: {gen_batch.shape} -> {ids.shape}')
    assert ids.shape == (16,)
    assert (ids >= 0).all() and (ids < n_items).all()


def test_grpo_one_step():
    import torch
    from src.flow_model import ConditionalFlowModel
    from src.sde_sampler import SDESampler
    from src.reward import BoundaryAwareReward
    from src.grpo import GRPOTrainer
    from src.neg_sampling import EmbeddingToItemMapper

    d = 32
    n_users = 10
    n_items = 50
    G = 4

    flow = ConditionalFlowModel(emb_dim=d, hidden_dim=64, n_layers=2, device='cpu')
    flow.save_as_reference()
    sampler = SDESampler(flow.velocity_net, n_steps=5, eta=0.5, delta=0.01)
    reward_fn = BoundaryAwareReward(a=1.0, gamma=1.0)
    item_emb = torch.randn(n_items, d)
    mapper = EmbeddingToItemMapper(item_emb)

    grpo = GRPOTrainer(
        flow, sampler, reward_fn, mapper,
        group_size=G, clip_eps=0.2, beta=0.1, lr=1e-3, max_pos_samples=5,
    )
    grpo._save_old_policy()

    user_emb = torch.randn(n_users, d)
    user_pos = {i: set(range(i, min(i+5, n_items))) for i in range(n_users)}

    batch_uids = list(range(4))
    u_emb = user_emb[batch_uids]
    loss, stats = grpo.grpo_step(u_emb, batch_uids, item_emb, user_pos)
    print(f'  GRPO step: loss={loss:.4f}, stats={stats}')
    assert not torch.isnan(torch.tensor(loss)), 'Loss should not be NaN'


def test_end_to_end_mini():
    """End-to-end FlowNS on ml-100k with minimal epochs."""
    import torch
    from src.recbole_utils import setup_recbole
    from src.flowns_trainer import FlowNSTrainer

    config_dict = {
        'epochs': 3,
        'eval_step': 3,
        'stopping_step': 3,
        'data_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RecBole', 'dataset')),
        'show_progress': False,
    }
    config, model, dataset, train_data, valid_data, test_data = setup_recbole(
        'LightGCN', 'ml-100k', config_dict=config_dict,
    )

    flow_config = {
        'flow_hidden_dim': 128,
        'flow_n_layers': 2,
        'flow_pretrain_epochs': 3,
        'sde_steps': 5,
        'grpo_epochs': 1,
        'grpo_group_size': 4,
        'grpo_batch_size': 16,
        'grpo_beta': 0.1,
        'joint_rec_epochs': 3,
        'joint_grpo_freq': 3,
        'joint_grpo_steps': 1,
        'eval_step': 3,
        'stopping_step': 3,
        'max_pos_samples': 5,
    }

    trainer = FlowNSTrainer(
        config, model, dataset, train_data, valid_data, test_data,
        flow_config=flow_config,
    )

    trainer.run()
    result = trainer.evaluate()
    print(f'  End-to-end test result: {result}')


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    check('1. Python imports', test_imports)
    check('2. RecBole ml-100k training', test_recbole_ml100k)
    check('3. Embedding extraction', test_embedding_extraction)
    check('4. Flow model CFM loss', test_flow_model)
    check('5. SDE sampler', test_sde_sampler)
    check('6. Reward computation', test_reward)
    check('7. Neg sampling mapper', test_neg_sampling)
    check('8. GRPO one step', test_grpo_one_step)
    check('9. End-to-end mini pipeline', test_end_to_end_mini)

    print(f'\n{"="*60}')
    print(f'  RESULTS: {PASS} passed, {FAIL} failed')
    print(f'{"="*60}')
    sys.exit(1 if FAIL > 0 else 0)
