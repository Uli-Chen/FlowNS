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
from recbole.trainer import Trainer
from recbole.evaluator import Evaluator, Collector
from recbole.utils import init_seed, init_logger

logger = logging.getLogger(__name__)


def setup_recbole(model_name, dataset_name, config_file_list=None, config_dict=None):
    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_class = _get_model_class(model_name)
    model = model_class(config, dataset).to(config['device'])

    return config, model, dataset, train_data, valid_data, test_data


def _get_model_class(model_name):
    name = model_name.upper()
    if name == 'LIGHTGCN':
        from recbole.model.general_recommender.lightgcn import LightGCN
        return LightGCN
    elif name == 'BPR':
        from recbole.model.general_recommender.bpr import BPR
        return BPR
    else:
        raise ValueError(f'Unsupported model: {model_name}')


@torch.no_grad()
def get_embeddings(model):
    model.eval()
    user_emb, item_emb = model.forward()
    return user_emb.detach(), item_emb.detach()


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
    trainer = Trainer(config, model)
    best_score, best_result = trainer.fit(train_data, valid_data, show_progress=True)
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
    result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    return result
