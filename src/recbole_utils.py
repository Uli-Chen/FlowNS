import sys
import os
import logging
import numpy as np

# Monkey-patch numpy for RecBole compatibility with NumPy 2.0+
# RecBole's compatibility_settings() does np.float = np.float_ etc,
# but NumPy 2.0 removed both np.float and np.float_.
for _attr, _replacement in [
    ('bool_', bool), ('int_', int), ('float_', float),
    ('complex_', complex), ('object_', object), ('str_', str),
    ('bool', bool), ('int', int), ('float', float),
    ('complex', complex), ('object', object), ('str', str),
    ('long', int), ('unicode', str), ('unicode_', str),
]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _replacement)

import torch

# Monkey-patch scipy dok_matrix for RecBole compatibility with scipy 1.14+
import scipy.sparse
if not hasattr(scipy.sparse.dok_matrix, '_update'):
    scipy.sparse.dok_matrix._update = scipy.sparse.dok_matrix.update

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RecBole')))

from recbole.config import Config
from recbole.data.utils import create_dataset, data_preparation
from recbole.utils import get_trainer
from recbole.evaluator import Evaluator, Collector
from recbole.utils import init_seed, init_logger

logger = logging.getLogger(__name__)


def setup_recbole(model_name, dataset_name, config_file_list=None, config_dict=None):
    argv = sys.argv
    try:
        sys.argv = sys.argv[:1]
        config = Config(
            model=model_name,
            dataset=dataset_name,
            config_file_list=config_file_list,
            config_dict=config_dict,
        )
    finally:
        sys.argv = argv
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_class = _get_model_class(model_name)
    model_dataset = _get_dataloader_dataset(train_data, dataset)
    model = model_class(config, model_dataset).to(config['device'])

    return config, model, dataset, train_data, valid_data, test_data


def _get_dataloader_dataset(data, fallback):
    """Return the split-local dataset attached to a RecBole dataloader."""
    dataset = getattr(data, 'dataset', None)
    if dataset is None:
        dataset = getattr(data, '_dataset', None)
    if dataset is None:
        logger.warning('Could not find split dataset on dataloader; using fallback.')
        dataset = fallback
    return dataset


def _get_model_class(model_name):
    name = model_name.upper()
    if name == 'LIGHTGCN':
        from recbole.model.general_recommender.lightgcn import LightGCN
        return LightGCN
    elif name == 'BPR':
        from recbole.model.general_recommender.bpr import BPR
        return BPR
    elif name == 'MULTIVAE':
        from recbole.model.general_recommender.multivae import MultiVAE
        return MultiVAE
    elif name == 'MULTIDAE':
        from recbole.model.general_recommender.multidae import MultiDAE
        return MultiDAE
    elif name == 'RECVAE':
        from recbole.model.general_recommender.recvae import RecVAE
        return RecVAE
    elif name == 'SASREC':
        from recbole.model.sequential_recommender.sasrec import SASRec
        return SASRec
    else:
        raise ValueError(f'Unsupported model: {model_name}')


def build_trainer(config, model):
    trainer_class = get_trainer(model.type, config['model'])
    return trainer_class(config, model)


def _decoder_final_linear(model):
    decoder = getattr(model, 'decoder', None)
    if decoder is None:
        return None
    if isinstance(decoder, torch.nn.Linear):
        return decoder
    modules = list(decoder.children())
    if modules and isinstance(modules[-1], torch.nn.Linear):
        return modules[-1]
    return None


def _run_decoder_prefix(model, latent):
    decoder = getattr(model, 'decoder', None)
    if isinstance(decoder, torch.nn.Linear):
        return latent
    modules = list(decoder.children())
    hidden = latent
    for layer in modules[:-1]:
        hidden = layer(hidden)
    return hidden


@torch.no_grad()
def get_item_embeddings(model):
    """Return item vectors in the model's scoring space."""
    model.eval()
    final_linear = _decoder_final_linear(model)
    if final_linear is not None:
        return final_linear.weight.detach()
    if hasattr(model, 'item_embedding'):
        return model.item_embedding.weight.detach()
    _, item_emb = model.forward()
    return item_emb.detach()


@torch.no_grad()
def get_user_embeddings_for_ids(model, user_ids):
    """Return user vectors aligned with get_item_embeddings()."""
    model.eval()
    if hasattr(model, 'get_rating_matrix') and hasattr(model, 'encoder'):
        rating_matrix = model.get_rating_matrix(user_ids)
        if model.__class__.__name__ == 'MultiVAE':
            h = torch.nn.functional.normalize(rating_matrix)
            h = model.encoder(h)
            latent = h[:, : int(model.lat_dim / 2)]
            return _run_decoder_prefix(model, latent).detach()
        if model.__class__.__name__ == 'MultiDAE':
            h = torch.nn.functional.normalize(rating_matrix)
            latent = model.encoder(h)
            return _run_decoder_prefix(model, latent).detach()
        if model.__class__.__name__ == 'RecVAE':
            mu, _ = model.encoder(rating_matrix, dropout_prob=0)
            return mu.detach()
    user_emb, _ = model.forward()
    return user_emb[user_ids].detach()


@torch.no_grad()
def get_embeddings(model):
    model.eval()
    item_emb = get_item_embeddings(model)
    if hasattr(model, 'get_rating_matrix') and hasattr(model, 'encoder'):
        chunks = []
        for start in range(0, model.n_users, 512):
            end = min(start + 512, model.n_users)
            user_ids = torch.arange(start, end, device=model.device)
            chunks.append(get_user_embeddings_for_ids(model, user_ids).detach())
        return torch.cat(chunks, dim=0), item_emb
    try:
        # Graph models (LightGCN, NGCF) compute all embeddings in a no-arg forward.
        user_emb, item_emb = model.forward()
        return user_emb.detach(), item_emb.detach()
    except TypeError:
        # Pure MF (e.g. BPR): forward() needs (user, item) and the embedding
        # tables themselves are the representations.
        if hasattr(model, 'user_embedding') and hasattr(model, 'item_embedding'):
            return model.user_embedding.weight.detach(), item_emb
        raise


def forward_all_embeddings(model):
    """Return grad-enabled (user_all, item_all) for joint-training losses.

    Graph models (LightGCN) recompute these in forward(); pure MF (BPR) keeps
    them as embedding tables. Raises for backbones with no static user table
    (e.g. sequential SASRec), which FlowNS joint training does not support.
    """
    try:
        return model.forward()
    except TypeError:
        if hasattr(model, 'user_embedding') and hasattr(model, 'item_embedding'):
            return model.user_embedding.weight, model.item_embedding.weight
        raise


def get_user_positive_items(data):
    """Return positive item ids from the provided RecBole split.

    Pass a train/valid/test dataloader when the positives should be split-local.
    Passing the full dataset returns positives from all interactions.
    """
    dataset = getattr(data, '_dataset', data)
    if hasattr(dataset, 'inter_feat'):
        uid_field = dataset.uid_field
        iid_field = dataset.iid_field
        user_pos = {}
        user_ids = dataset.inter_feat[uid_field].numpy()
        item_ids = dataset.inter_feat[iid_field].numpy()
        for uid, iid in zip(user_ids, item_ids):
            user_pos.setdefault(int(uid), set()).add(int(iid))
        return user_pos

    inter_matrix = dataset.inter_matrix(form='csr')
    user_pos = {}
    for uid in range(inter_matrix.shape[0]):
        items = inter_matrix[uid].indices.tolist()
        if items:
            user_pos[uid] = set(items)
    return user_pos


def train_recbole(config, model, train_data, valid_data):
    trainer = build_trainer(config, model)
    show_progress = bool(config['show_progress'])
    best_score, best_result = trainer.fit(
        train_data, valid_data, show_progress=show_progress,
    )
    return trainer, best_score, best_result


def _load_best_checkpoint(trainer):
    """Load best checkpoint into model (compatible with PyTorch 2.6+ weights_only default)."""
    checkpoint = torch.load(
        trainer.saved_model_file, map_location=trainer.device, weights_only=False,
    )
    trainer.model.load_state_dict(checkpoint['state_dict'])
    trainer.model.load_other_parameter(checkpoint.get('other_parameter'))


@torch.no_grad()
def evaluate_recbole(trainer, test_data):
    _load_best_checkpoint(trainer)
    show_progress = bool(trainer.config['show_progress'])
    result = trainer.evaluate(
        test_data, load_best_model=False, show_progress=show_progress,
    )
    return result
