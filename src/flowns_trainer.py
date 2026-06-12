import logging
import os
import random
import torch
import torch.nn.functional as F
import numpy as np

from recbole.utils import calculate_valid_score, early_stopping

from .flow_model import ConditionalFlowModel
from .sde_sampler import SDESampler
from .reward import BoundaryAwareReward
from .grpo import GRPOTrainer
from .neg_sampling import EmbeddingToItemMapper
from .recbole_utils import (
    build_trainer,
    forward_all_embeddings,
    get_embeddings,
    get_item_embeddings,
    get_user_embeddings_for_ids,
    get_user_positive_items,
)

logger = logging.getLogger(__name__)


FLOW_CONFIG_DEFAULTS = {
    'enabled_phases': None,
    'flow_hidden_dim': 256,
    'flow_n_layers': 3,
    'flow_pretrain_epochs': 50,
    'flow_pretrain_batch_size': 256,
    'flow_lr': 1e-4,
    'sde_steps': 20,
    'eta': 0.5,
    'delta': 0.01,
    'sde_sigma_min': 0.0,
    'sde_score_clamp': None,
    'grpo_epochs': 10,
    'grpo_group_size': 8,
    'grpo_clip_eps': 0.2,
    'grpo_beta': 0.1,
    'grpo_lr': 1e-4,
    'grpo_batch_size': 64,
    'grpo_old_policy_scope': 'batch',
    'grpo_normalize_log_ratio': False,
    'grpo_reward_mode': 'mapped_item',
    'reward_a': 1.0,
    'reward_gamma': 1.0,
    'joint_rec_epochs': 100,
    'joint_grpo_freq': 5,
    'joint_grpo_steps': 2,
    'joint_num_negatives': 1,
    'joint_lr': None,
    'eval_step': 5,
    'stopping_step': 10,
    'disable_early_stopping': False,
    'max_pos_samples': 10,
    'flow_neg_ratio': 1.0,
    'flow_neg_loss_weight': 0.1,
    'joint_exposed_neg_ratio': 0.0,
    'mapping_strategy': 'nearest',
    'mapping_topk': 50,
    'boundary_safe_w': 0.5,
    'mapper_chunk_size': 16384,
    'negative_source': 'random',
    'use_random_neg': False,
    'rec_checkpoint_path': None,
    'flow_checkpoint_path': None,
    'flow_ref_checkpoint_path': None,
    'flow_log_interval': 10,
    'grpo_log_interval': 1,
    'rerank_num_negatives': 0,
    'rerank_neg_penalty': 0.0,
    'rerank_mapping_strategy': 'nearest',
    'rerank_mapping_topk': 50,
    'rerank_score_mode': 'mapped_item',
    'rerank_candidate_topk': 50,
    'eval_exposed_neg_penalty': 0.0,
}

FLOW_PHASE_ALIASES = {
    'phase1': 'rec_pretrain',
    'rec': 'rec_pretrain',
    'recommender': 'rec_pretrain',
    'recommender_pretrain': 'rec_pretrain',
    'phase2': 'flow_pretrain',
    'flow': 'flow_pretrain',
    'flow_train': 'flow_pretrain',
    'phase3': 'grpo',
    'rl': 'grpo',
    'phase4': 'joint',
    'joint_train': 'joint',
    'finetune': 'joint',
}


def _config_get(config, key, default=None):
    try:
        return config[key]
    except (KeyError, TypeError):
        pass

    for attr in ('final_config_dict', 'parameters'):
        data = getattr(config, attr, None)
        if isinstance(data, dict) and key in data:
            return data[key]
    return default


def _normalize_enabled_phases(phases):
    if phases is None:
        return None
    if isinstance(phases, str):
        phases = [p.strip() for p in phases.split(',') if p.strip()]
    normalized = []
    for phase in phases:
        phase_name = FLOW_PHASE_ALIASES.get(str(phase).strip(), str(phase).strip())
        if phase_name not in {'rec_pretrain', 'flow_pretrain', 'grpo', 'joint'}:
            raise ValueError(
                f'Unknown FlowNS phase: {phase}. '
                'Valid phases: rec_pretrain, flow_pretrain, grpo, joint.'
            )
        normalized.append(phase_name)
    return normalized


class FlowNSTrainer:
    """Configurable FlowNS trainer composing RecBole Trainer with flow + GRPO."""

    def __init__(self, config, rec_model, dataset, train_data, valid_data, test_data,
                 flow_config=None):
        """
        Args:
            config: RecBole Config
            rec_model: LightGCN (or BPR) model
            dataset: RecBole Dataset
            train_data, valid_data, test_data: RecBole DataLoaders
            flow_config: dict with FlowNS-specific hyperparameters
        """
        self.config = config
        self.rec_model = rec_model
        self.dataset = dataset
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.device = config['device']

        fc = dict(FLOW_CONFIG_DEFAULTS)
        for key in FLOW_CONFIG_DEFAULTS:
            value = _config_get(config, key, None)
            if value is not None:
                fc[key] = value
        if flow_config:
            fc.update({k: v for k, v in flow_config.items() if v is not None})

        self.flow_config = fc
        self.is_autoencoder_backbone = (
            hasattr(rec_model, 'get_rating_matrix')
            and hasattr(rec_model, 'encoder')
            and hasattr(rec_model, 'decoder')
        )
        emb_dim = int(get_item_embeddings(rec_model).shape[1])

        self.flow_model = ConditionalFlowModel(
            emb_dim=emb_dim,
            hidden_dim=fc.get('flow_hidden_dim', 256),
            n_layers=fc.get('flow_n_layers', 3),
            device=str(self.device),
        )

        self.sde_sampler = SDESampler(
            velocity_net=self.flow_model.velocity_net,
            n_steps=fc.get('sde_steps', 20),
            eta=fc.get('eta', 0.5),
            delta=fc.get('delta', 0.01),
            sigma_min=fc.get('sde_sigma_min', 0.0),
            score_clamp=fc.get('sde_score_clamp', None),
        )

        self.reward_fn = BoundaryAwareReward(
            a=fc.get('reward_a', 1.0),
            gamma=fc.get('reward_gamma', 1.0),
        )

        self.mapper = None  # initialized after Phase 1

        self.grpo_trainer = None  # initialized after Phase 1

        self.flow_pretrain_epochs = fc.get('flow_pretrain_epochs', 50)
        self.flow_pretrain_batch_size = fc.get('flow_pretrain_batch_size', 256)
        self.flow_lr = fc.get('flow_lr', 1e-4)
        self.grpo_epochs = fc.get('grpo_epochs', 10)
        self.grpo_group_size = fc.get('grpo_group_size', 8)
        self.grpo_clip_eps = fc.get('grpo_clip_eps', 0.2)
        self.grpo_beta = fc.get('grpo_beta', 0.1)
        self.grpo_lr = fc.get('grpo_lr', 1e-4)
        self.grpo_batch_size = fc.get('grpo_batch_size', 64)
        self.joint_rec_epochs = fc.get('joint_rec_epochs', 100)
        self.joint_grpo_freq = fc.get('joint_grpo_freq', 5)
        self.joint_grpo_steps = fc.get('joint_grpo_steps', 2)
        self.joint_num_negatives = max(int(fc.get('joint_num_negatives', 1)), 1)
        self.joint_lr = fc.get('joint_lr', None)
        self.eval_step = fc.get('eval_step', 5)
        self.stopping_step = fc.get('stopping_step', 10)
        self.disable_early_stopping = fc.get('disable_early_stopping', False)
        self.max_pos_samples = fc.get('max_pos_samples', 10)
        self.use_random_neg = fc.get('use_random_neg', False)
        self.flow_neg_ratio = fc.get('flow_neg_ratio', 1.0)
        self.flow_neg_loss_weight = float(fc.get('flow_neg_loss_weight', 0.1))
        self.joint_exposed_neg_ratio = fc.get('joint_exposed_neg_ratio', 0.0)
        self.mapping_strategy = fc.get('mapping_strategy', 'nearest')
        self.mapping_topk = fc.get('mapping_topk', 50)
        self.boundary_safe_w = float(fc.get('boundary_safe_w', 0.5))
        self.mapper_chunk_size = max(int(fc.get('mapper_chunk_size', 16384)), 1)
        self.negative_source = fc.get('negative_source', 'random')
        self.rec_checkpoint_path = fc.get('rec_checkpoint_path')
        self.flow_checkpoint_path = fc.get('flow_checkpoint_path')
        self.flow_ref_checkpoint_path = fc.get('flow_ref_checkpoint_path')
        self.flow_log_interval = fc.get('flow_log_interval', 10)
        self.grpo_log_interval = fc.get('grpo_log_interval', 1)
        self.rerank_num_negatives = max(int(fc.get('rerank_num_negatives', 0)), 0)
        self.rerank_neg_penalty = float(fc.get('rerank_neg_penalty', 0.0))
        self.rerank_mapping_strategy = fc.get('rerank_mapping_strategy', 'nearest')
        self.rerank_mapping_topk = fc.get('rerank_mapping_topk', 50)
        self.rerank_score_mode = fc.get('rerank_score_mode', 'mapped_item')
        self.rerank_candidate_topk = max(int(fc.get('rerank_candidate_topk', 50)), 1)
        self.eval_exposed_neg_penalty = float(
            fc.get('eval_exposed_neg_penalty', 0.0)
        )
        self.enabled_phases = _normalize_enabled_phases(fc.get('enabled_phases'))
        if self.enabled_phases is None:
            if self.use_random_neg:
                self.enabled_phases = ['rec_pretrain', 'joint']
            else:
                self.enabled_phases = [
                    'rec_pretrain', 'flow_pretrain', 'grpo', 'joint',
                ]

        # Match RecBole's train sampler: training negatives only exclude
        # positives in the train split, not future valid/test positives.
        self.user_pos_items = get_user_positive_items(train_data)
        self.user_neg_items = None
        if self.negative_source == 'exposed':
            self.user_neg_items = self._load_exposed_negatives()
        elif self.negative_source != 'random':
            raise ValueError('negative_source must be "random" or "exposed"')
        self._rec_trainer = None
        self._phase1_best_score = None
        self._phase1_best_result = None

        logger.info(
            'FlowNS phases enabled: %s',
            ' -> '.join(self.enabled_phases) if self.enabled_phases else '(none)',
        )
        logger.info(
            'FlowNS config: flow_hidden_dim=%s, flow_layers=%s, '
            'flow_epochs=%s, grpo_epochs=%s, joint_epochs=%s, '
            'joint_negatives=%s, exposed_ratio=%.3f, flow_ratio=%.3f, '
            'flow_neg_loss_weight=%.4f, '
            'mapping=%s@%s, rerank=%s/%d/%.4f, '
            'eval_exposed_penalty=%.4f, random_neg=%s, backbone=%s, '
            'emb_dim=%d, checkpoint=%s',
            fc.get('flow_hidden_dim'), fc.get('flow_n_layers'),
            self.flow_pretrain_epochs, self.grpo_epochs,
            self.joint_rec_epochs, self.joint_num_negatives,
            float(self.joint_exposed_neg_ratio), float(self.flow_neg_ratio),
            self.flow_neg_loss_weight,
            self.mapping_strategy, self.mapping_topk,
            self.rerank_score_mode, self.rerank_num_negatives,
            self.rerank_neg_penalty, self.eval_exposed_neg_penalty,
            self.use_random_neg,
            'autoencoder' if self.is_autoencoder_backbone else 'embedding',
            emb_dim,
            self.rec_checkpoint_path or 'none',
        )

    def _load_exposed_negatives(self):
        dataset_name = str(self.config['dataset'])
        candidates = [
            os.path.join(str(self.config['data_path']), f'{dataset_name}.exposed_neg'),
            os.path.join(
                str(self.config['data_path']),
                dataset_name,
                f'{dataset_name}.exposed_neg',
            ),
        ]
        path = next((item for item in candidates if os.path.exists(item)), None)
        if path is None:
            searched = ', '.join(candidates)
            raise FileNotFoundError(
                f'exposed negative file not found. Searched: {searched}'
            )

        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        user_neg = {}
        total = 0
        kept = 0
        skipped = 0
        with open(path) as f:
            header = next(f, None)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                total += 1
                raw_uid, raw_iid = parts[0], parts[1]
                try:
                    uid = int(self.dataset.token2id(uid_field, raw_uid))
                    iid = int(self.dataset.token2id(iid_field, raw_iid))
                except ValueError:
                    skipped += 1
                    continue
                if iid <= 0 or uid <= 0:
                    skipped += 1
                    continue
                if iid in self.user_pos_items.get(uid, set()):
                    skipped += 1
                    continue
                user_neg.setdefault(uid, []).append(iid)
                kept += 1

        logger.info(
            'Loaded exposed negatives: path=%s, total=%d, kept=%d, '
            'users=%d, skipped=%d',
            path, total, kept, len(user_neg), skipped,
        )
        return user_neg

    def phase1(self):
        """Phase 1: Standard RecBole training or loading a paired checkpoint."""
        if self.rec_checkpoint_path:
            logger.info('=== Phase 1: Load paired RecBole checkpoint ===')
        elif self.use_random_neg:
            logger.info('=== Phase 1: RNS pretrain ===')
        else:
            logger.info('=== Phase 1: RecBole pretrain ===')

        # (a) Train recommender with RecBole
        rec_trainer = build_trainer(self.config, self.rec_model)
        if self.rec_checkpoint_path:
            checkpoint = torch.load(
                self.rec_checkpoint_path, map_location=self.device, weights_only=False,
            )
            self.rec_model.load_state_dict(checkpoint['state_dict'])
            self.rec_model.load_other_parameter(checkpoint.get('other_parameter'))
            rec_trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            rec_trainer.best_valid_score = checkpoint.get(
                'best_valid_score', rec_trainer.best_valid_score,
            )
            rec_trainer.cur_step = checkpoint.get('cur_step', rec_trainer.cur_step)
            best_score = rec_trainer.best_valid_score
            best_result = None

            # RecBole's fit() normally collects train history for full-sort
            # evaluation and writes a best-model checkpoint. Do both explicitly
            # so paired runs start from the same M0 model without overwriting it.
            rec_trainer.eval_collector.data_collect(self.train_data)
            rec_trainer._save_checkpoint(
                checkpoint.get('epoch', -1), verbose=False,
            )
            logger.info(
                f'Phase 1: loaded paired checkpoint from {self.rec_checkpoint_path}'
            )
            logger.info(
                f'Phase 1: copied paired checkpoint to {rec_trainer.saved_model_file}'
            )
        else:
            show_progress = bool(self.config['show_progress'])
            best_score, best_result = rec_trainer.fit(
                self.train_data, self.valid_data, show_progress=show_progress,
            )
            logger.info(f'Phase 1 RecBole best valid score: {best_score}')
        self._phase1_best_score = best_score
        self._phase1_best_result = best_result
        self._rec_trainer = rec_trainer

        if not self.rec_checkpoint_path:
            # Load best checkpoint back into model (fit() leaves last-epoch weights)
            checkpoint = torch.load(
                rec_trainer.saved_model_file, map_location=self.device, weights_only=False,
            )
            self.rec_model.load_state_dict(checkpoint['state_dict'])
            self.rec_model.load_other_parameter(checkpoint.get('other_parameter'))
            rec_trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            rec_trainer.best_valid_score = checkpoint.get('best_valid_score', best_score)
            rec_trainer.cur_step = checkpoint.get('cur_step', rec_trainer.cur_step)
            logger.info(
                f'Phase 1: loaded best checkpoint from {rec_trainer.saved_model_file}'
            )

        return best_score, best_result

    def _refresh_mapper(self, item_emb):
        if self.mapper is None:
            self.mapper = EmbeddingToItemMapper(
                item_emb, chunk_size=self.mapper_chunk_size,
            )
        else:
            self.mapper.update_embeddings(item_emb)

    def _init_flow_runtime(self, item_emb):
        self._refresh_mapper(item_emb)

        if self.grpo_trainer is None:
            self.grpo_trainer = GRPOTrainer(
                flow_model=self.flow_model,
                sde_sampler=self.sde_sampler,
                reward_fn=self.reward_fn,
                mapper=self.mapper,
                group_size=self.grpo_group_size,
                clip_eps=self.grpo_clip_eps,
                beta=self.grpo_beta,
                lr=self.grpo_lr,
                max_pos_samples=self.max_pos_samples,
                log_interval=self.grpo_log_interval,
                old_policy_scope=self.flow_config.get(
                    'grpo_old_policy_scope', 'batch',
                ),
                normalize_log_ratio=self.flow_config.get(
                    'grpo_normalize_log_ratio', False,
                ),
                reward_mode=self.flow_config.get(
                    'grpo_reward_mode', 'mapped_item',
                ),
                mapping_strategy=self.mapping_strategy,
                mapping_topk=self.mapping_topk,
                boundary_safe_w=self.boundary_safe_w,
            )

    def _get_pos_embs_for_users(self, user_ids, item_emb):
        K = self.max_pos_samples
        d = item_emb.shape[1]
        pos_embs = torch.zeros(len(user_ids), K, d, device=item_emb.device)
        for row, uid in enumerate(user_ids):
            pos_ids = list(self.user_pos_items.get(int(uid), []))
            if not pos_ids:
                continue
            if len(pos_ids) > K:
                indices = torch.randperm(len(pos_ids))[:K]
                pos_ids = [pos_ids[int(i)] for i in indices]
            pos_embs[row, :len(pos_ids)] = item_emb[pos_ids]
        return pos_embs

    def _sample_exposed_negatives(self, user_ids, n_negatives, fallback_neg):
        if not self.user_neg_items:
            return fallback_neg

        sampled = torch.empty(
            len(user_ids), n_negatives,
            device=fallback_neg.device,
            dtype=fallback_neg.dtype,
        )
        for row, uid in enumerate(user_ids):
            pool = self.user_neg_items.get(int(uid))
            if pool:
                sampled[row] = torch.as_tensor(
                    random.choices(pool, k=n_negatives),
                    device=fallback_neg.device,
                    dtype=fallback_neg.dtype,
                )
            else:
                sampled[row] = fallback_neg[row]
        return sampled

    def _resolve_local_path(self, path):
        if not path:
            return None
        path = os.path.expanduser(str(path))
        if os.path.isabs(path):
            return path
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.join(project_dir, path)

    def _load_flow_checkpoint(self):
        flow_path = self._resolve_local_path(self.flow_checkpoint_path)
        if not flow_path or not os.path.exists(flow_path):
            return False

        state = torch.load(flow_path, map_location=self.device, weights_only=False)
        self.flow_model.velocity_net.load_state_dict(state)

        ref_path = self._resolve_local_path(self.flow_ref_checkpoint_path)
        if ref_path and os.path.exists(ref_path):
            self.flow_model.ref_state_dict = torch.load(
                ref_path, map_location=self.device, weights_only=False,
            )
        else:
            self.flow_model.save_as_reference()

        logger.info(
            'Phase 2: loaded flow checkpoint from %s (ref=%s)',
            flow_path,
            ref_path if ref_path and os.path.exists(ref_path) else 'current-flow',
        )
        return True

    def _save_flow_checkpoint(self):
        flow_path = self._resolve_local_path(self.flow_checkpoint_path)
        if not flow_path:
            return

        ref_path = self._resolve_local_path(self.flow_ref_checkpoint_path)
        os.makedirs(os.path.dirname(flow_path), exist_ok=True)
        torch.save(self.flow_model.velocity_net.state_dict(), flow_path)
        if ref_path:
            os.makedirs(os.path.dirname(ref_path), exist_ok=True)
            torch.save(self.flow_model.ref_state_dict, ref_path)

        logger.info(
            'Phase 2: saved flow checkpoint to %s (ref=%s)',
            flow_path,
            ref_path or 'not-configured',
        )

    def _ensure_rec_ready(self):
        if self._rec_trainer is None:
            raise RuntimeError(
                'RecBole trainer is not initialized. Enable/run rec_pretrain first.'
            )

    def _ensure_flow_ready(self):
        if self.use_random_neg:
            return
        if self.flow_model.ref_state_dict is None:
            raise RuntimeError(
                'Flow reference policy is missing. Enable/run flow_pretrain before '
                'GRPO or flow-based joint training.'
            )

    def _joint_uses_flow_negatives(self):
        return (
            not self.use_random_neg
            and float(self.flow_neg_ratio) > 0.0
            and float(self.joint_exposed_neg_ratio) < 1.0
        )

    def _joint_uses_grpo(self):
        return (
            not self.use_random_neg
            and self.joint_grpo_steps > 0
            and self.joint_grpo_freq > 0
        )

    def phase2(self):
        """Phase 2: Flow CFM pretraining."""
        self._ensure_rec_ready()

        if self.use_random_neg:
            logger.info('=== Phase 2 skipped: RNS control has no flow pretraining ===')
            return None

        logger.info('=== Phase 2: Flow CFM pretraining ===')
        user_emb, item_emb = get_embeddings(self.rec_model)
        if self._load_flow_checkpoint():
            self._init_flow_runtime(item_emb)
            return None

        self.flow_model.pretrain(
            user_emb, item_emb, self.user_pos_items,
            user_neg_items=self.user_neg_items,
            epochs=self.flow_pretrain_epochs,
            batch_size=self.flow_pretrain_batch_size,
            lr=self.flow_lr,
            log_interval=self.flow_log_interval,
            stopping_step=self.stopping_step,
        )
        self._save_flow_checkpoint()
        self._init_flow_runtime(item_emb)
        return None

    def phase3(self):
        """Phase 3: GRPO fine-tuning of flow model."""
        self._ensure_rec_ready()
        if self.use_random_neg:
            logger.info('=== Phase 3 skipped: RNS control has no flow fine-tuning ===')
            return
        if self.grpo_epochs <= 0:
            logger.info('=== Phase 3 skipped: grpo_epochs <= 0 ===')
            return
        self._ensure_flow_ready()

        logger.info('=== Phase 3: GRPO fine-tuning ===')
        user_emb, item_emb = get_embeddings(self.rec_model)
        self._init_flow_runtime(item_emb)

        self.grpo_trainer.train(
            user_emb, item_emb, self.user_pos_items,
            epochs=self.grpo_epochs,
            batch_size=self.grpo_batch_size,
            stopping_step=self.stopping_step,
        )

    def phase4(self):
        """Phase 4: Joint alternating training or RNS finetuning."""
        self._ensure_rec_ready()
        joint_uses_flow = self._joint_uses_flow_negatives()
        joint_uses_grpo = self._joint_uses_grpo()
        if self.use_random_neg:
            logger.info('=== Phase 4: RNS finetune ===')
        elif joint_uses_flow or joint_uses_grpo:
            self._ensure_flow_ready()
            logger.info('=== Phase 4: Joint training ===')
        else:
            logger.info('=== Phase 4: Exposed/random negative joint training ===')
        rec_trainer = self._rec_trainer

        if self.joint_lr is not None:
            for param_group in rec_trainer.optimizer.param_groups:
                param_group['lr'] = self.joint_lr
            logger.info('Phase 4: learning rate set to %s', self.joint_lr)

        phase1_score = getattr(self, '_phase1_best_score', None)
        best_valid_score = float(phase1_score) if phase1_score is not None else -np.inf
        cur_step = 0

        for epoch in range(self.joint_rec_epochs):
            # (a) Prepare flow state only for flow-generated negatives.
            if self.use_random_neg or not joint_uses_flow:
                user_emb = item_emb = None
            else:
                if self.is_autoencoder_backbone:
                    user_emb = None
                    item_emb = get_item_embeddings(self.rec_model)
                else:
                    user_emb, item_emb = get_embeddings(self.rec_model)
                self._refresh_mapper(item_emb)

            # (b) Train rec model one epoch with per-interaction custom negatives
            custom_loss = self._make_custom_loss(user_emb, item_emb)
            train_loss = rec_trainer._train_epoch(
                self.train_data, epoch, loss_func=custom_loss, show_progress=False
            )

            if (epoch + 1) % self.eval_step == 0:
                valid_result = rec_trainer.evaluate(
                    self.valid_data, load_best_model=False, show_progress=False
                )
                valid_score = calculate_valid_score(
                    valid_result, self.config['valid_metric'].lower()
                )
                logger.info(
                    f'Phase 4 epoch {epoch+1}/{self.joint_rec_epochs}, '
                    f'loss={train_loss}, valid={valid_score:.4f}, '
                    f'best={best_valid_score:.4f}, patience={cur_step}/{self.stopping_step}'
                )

                best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, best_valid_score, cur_step,
                    max_step=self.stopping_step,
                    bigger=self.config['valid_metric_bigger'],
                )
                if update_flag:
                    rec_trainer._save_checkpoint(epoch)
                if stop_flag and not self.disable_early_stopping:
                    logger.info(f'Phase 4 early stopping at epoch {epoch+1}')
                    break

            # (c) GRPO update every joint_grpo_freq epochs
            if (
                joint_uses_grpo
                and (epoch + 1) % self.joint_grpo_freq == 0
            ):
                user_emb, item_emb = get_embeddings(self.rec_model)
                self._refresh_mapper(item_emb)
                self.grpo_trainer.train_steps(
                    user_emb, item_emb, self.user_pos_items,
                    n_steps=self.joint_grpo_steps,
                    batch_size=self.grpo_batch_size,
                )

        return best_valid_score

    def _make_custom_loss(self, user_emb_all, item_emb):
        """Create a loss function that generates per-interaction negatives.

        For flow negatives: runs SDE sampling on the batch's user embeddings.
        For random negatives: reuses RecBole's sampled neg_item_id.
        When flow_neg_ratio < 1.0, mixes both: each sample in the batch
        independently uses flow or random negatives.
        """
        if self.is_autoencoder_backbone:
            return self._make_autoencoder_custom_loss(item_emb)

        model = self.rec_model
        sde_sampler = self.sde_sampler
        mapper = self.mapper
        use_random = self.use_random_neg
        user_pos = self.user_pos_items
        flow_ratio = self.flow_neg_ratio
        exposed_ratio = self.joint_exposed_neg_ratio
        mapping_strategy = self.mapping_strategy
        mapping_topk = self.mapping_topk
        n_negatives = self.joint_num_negatives
        use_flow = self._joint_uses_flow_negatives()

        def custom_loss(interaction):
            user_ids = interaction[model.USER_ID]
            pos_item_ids = interaction[model.ITEM_ID]

            if use_random:
                return model.calculate_loss(interaction)

            B = user_ids.shape[0]
            with torch.no_grad():
                random_neg = interaction[model.NEG_ITEM_ID].unsqueeze(1).expand(
                    B, n_negatives,
                )
                if use_flow:
                    u_emb = user_emb_all[user_ids]
                    final_emb, _, _ = sde_sampler.sample_trajectories(
                        u_emb, n_trajectories=n_negatives,
                    )
                    forbidden = [
                        user_pos.get(int(uid), set())
                        for uid in user_ids.tolist()
                        for _ in range(n_negatives)
                    ]
                    pos_embs = None
                    mapper_user_emb = u_emb.unsqueeze(1).expand(
                        B, n_negatives, u_emb.shape[-1],
                    )
                    if mapping_strategy in ('reward_topk', 'boundary_topk'):
                        pos_base = self._get_pos_embs_for_users(
                            user_ids.tolist(), item_emb,
                        )
                        pos_embs = pos_base.unsqueeze(1).expand(
                            B, n_negatives, pos_base.shape[-2], pos_base.shape[-1],
                        )
                    flow_neg = mapper.map_to_items(
                        final_emb,
                        forbidden_item_ids=forbidden,
                        exclude_item_ids=(0,),
                        user_emb=mapper_user_emb,
                        pos_item_embs=pos_embs,
                        reward_fn=self.reward_fn,
                        strategy=mapping_strategy,
                        candidate_topk=mapping_topk,
                        boundary_safe_w=self.boundary_safe_w,
                    )
                else:
                    flow_neg = random_neg

                if exposed_ratio > 0.0:
                    exposed_neg = self._sample_exposed_negatives(
                        user_ids.tolist(), n_negatives, random_neg,
                    )
                    draw = torch.rand(B, n_negatives, device=flow_neg.device)
                    exposed_mask = draw < exposed_ratio
                    flow_mask = draw < (exposed_ratio + flow_ratio)
                    neg_tensor = torch.where(
                        exposed_mask,
                        exposed_neg,
                        torch.where(flow_mask, flow_neg, random_neg),
                    )
                elif flow_ratio < 1.0:
                    mask = torch.rand(
                        B, n_negatives, device=flow_neg.device,
                    ) < flow_ratio
                    neg_tensor = torch.where(mask, flow_neg, random_neg)
                else:
                    neg_tensor = flow_neg

            if getattr(model, 'restore_user_e', None) is not None:
                model.restore_user_e = None
            if getattr(model, 'restore_item_e', None) is not None:
                model.restore_item_e = None

            user_all_embeddings, item_all_embeddings = forward_all_embeddings(model)
            u_embeddings = user_all_embeddings[user_ids]
            pos_embeddings = item_all_embeddings[pos_item_ids]
            neg_embeddings = item_all_embeddings[neg_tensor]

            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=-1)
            neg_scores = torch.mul(
                u_embeddings.unsqueeze(1), neg_embeddings,
            ).sum(dim=-1)
            score_diff = pos_scores.unsqueeze(1) - neg_scores
            gamma = getattr(getattr(model, 'mf_loss', None), 'gamma', 1e-10)
            mf_loss = -torch.log(gamma + torch.sigmoid(score_diff)).mean()

            # LightGCN-style L2 reg on ego embeddings. Some embedding backbones
            # (e.g. BPR) rely on optimizer weight decay and define neither
            # reg_loss nor reg_weight; for those we return the BPR term alone.
            reg_fn = getattr(model, 'reg_loss', None)
            reg_weight = getattr(model, 'reg_weight', None)
            if reg_fn is None or reg_weight is None:
                return mf_loss

            u_ego = model.user_embedding(user_ids).unsqueeze(1).expand(
                B, n_negatives, -1,
            ).reshape(B * n_negatives, -1)
            pos_ego = model.item_embedding(pos_item_ids).unsqueeze(1).expand(
                B, n_negatives, -1,
            ).reshape(B * n_negatives, -1)
            neg_ego = model.item_embedding(neg_tensor.reshape(-1))
            reg_loss = reg_fn(
                u_ego, pos_ego, neg_ego,
                require_pow=getattr(model, 'require_pow', False),
            )

            return mf_loss + reg_weight * reg_loss

        return custom_loss

    def _sample_random_negatives_for_users(self, user_ids, n_negatives, device):
        n_items = int(getattr(self.rec_model, 'n_items', self.dataset.item_num))
        sampled = torch.empty(
            len(user_ids), n_negatives, device=device, dtype=torch.long,
        )
        for row, uid in enumerate(user_ids):
            positives = self.user_pos_items.get(int(uid), set())
            values = []
            for _ in range(n_negatives):
                for _attempt in range(100):
                    item_id = random.randint(1, n_items - 1)
                    if item_id not in positives:
                        break
                values.append(item_id)
            sampled[row] = torch.as_tensor(values, device=device, dtype=torch.long)
        return sampled

    def _sample_positive_items_for_users(self, user_ids, device):
        sampled = torch.zeros(len(user_ids), device=device, dtype=torch.long)
        valid = torch.zeros(len(user_ids), device=device, dtype=torch.bool)
        for row, uid in enumerate(user_ids):
            positives = list(self.user_pos_items.get(int(uid), []))
            if not positives:
                continue
            sampled[row] = int(random.choice(positives))
            valid[row] = True
        return sampled, valid

    def _autoencoder_scores(self, user_ids):
        model = self.rec_model
        rating_matrix = model.get_rating_matrix(user_ids)
        if model.__class__.__name__ == 'RecVAE':
            scores, _, _, _ = model.forward(
                rating_matrix,
                getattr(model, 'dropout_prob', 0.0) if model.training else 0.0,
            )
            return scores
        out = model.forward(rating_matrix)
        return out[0] if isinstance(out, tuple) else out

    def _make_autoencoder_custom_loss(self, item_emb):
        """MultiVAE/MultiDAE joint objective.

        AE models train from user rating matrices, so there is no per-row
        `item_id`/`neg_item_id` in the batch. Keep the native AE objective and
        add a small sampled-positive BPR penalty on flow/random negatives.
        """
        model = self.rec_model
        sde_sampler = self.sde_sampler
        mapper = self.mapper
        user_pos = self.user_pos_items
        use_flow = self._joint_uses_flow_negatives()
        flow_ratio = float(self.flow_neg_ratio)
        exposed_ratio = float(self.joint_exposed_neg_ratio)
        n_negatives = self.joint_num_negatives
        mapping_strategy = self.mapping_strategy
        mapping_topk = self.mapping_topk
        loss_weight = self.flow_neg_loss_weight

        def custom_loss(interaction):
            user_ids = interaction[model.USER_ID]
            uid_list = [int(uid) for uid in user_ids.detach().cpu().tolist()]
            base_loss = model.calculate_loss(interaction)
            if loss_weight <= 0.0:
                return base_loss

            B = user_ids.shape[0]
            with torch.no_grad():
                random_neg = self._sample_random_negatives_for_users(
                    uid_list, n_negatives, user_ids.device,
                )
                if use_flow:
                    u_emb = get_user_embeddings_for_ids(model, user_ids)
                    final_emb, _, _ = sde_sampler.sample_trajectories(
                        u_emb, n_trajectories=n_negatives,
                    )
                    forbidden = [
                        user_pos.get(uid, set())
                        for uid in uid_list
                        for _ in range(n_negatives)
                    ]
                    mapper_user_emb = u_emb.unsqueeze(1).expand(
                        B, n_negatives, u_emb.shape[-1],
                    )
                    pos_embs = None
                    if mapping_strategy in ('reward_topk', 'boundary_topk'):
                        # For MultiVAE the item embedding is the decoder weight
                        # vector and u_emb is the pre-decoder hidden, so u·item is
                        # the decoder logit. Reward-aware mapping is therefore
                        # consistent with the AE scoring function.
                        pos_base = self._get_pos_embs_for_users(uid_list, item_emb)
                        pos_embs = pos_base.unsqueeze(1).expand(
                            B, n_negatives, pos_base.shape[-2], pos_base.shape[-1],
                        )
                    flow_neg = mapper.map_to_items(
                        final_emb,
                        forbidden_item_ids=forbidden,
                        exclude_item_ids=(0,),
                        user_emb=mapper_user_emb,
                        pos_item_embs=pos_embs,
                        reward_fn=self.reward_fn,
                        strategy=mapping_strategy,
                        candidate_topk=mapping_topk,
                        boundary_safe_w=self.boundary_safe_w,
                    )
                else:
                    flow_neg = random_neg

                if exposed_ratio > 0.0:
                    exposed_neg = self._sample_exposed_negatives(
                        uid_list, n_negatives, random_neg,
                    )
                    draw = torch.rand(B, n_negatives, device=user_ids.device)
                    exposed_mask = draw < exposed_ratio
                    flow_mask = draw < (exposed_ratio + flow_ratio)
                    neg_tensor = torch.where(
                        exposed_mask,
                        exposed_neg,
                        torch.where(flow_mask, flow_neg, random_neg),
                    )
                elif flow_ratio < 1.0:
                    mask = torch.rand(B, n_negatives, device=user_ids.device) < flow_ratio
                    neg_tensor = torch.where(mask, flow_neg, random_neg)
                else:
                    neg_tensor = flow_neg

                pos_item_ids, valid_pos = self._sample_positive_items_for_users(
                    uid_list, user_ids.device,
                )

            if not bool(valid_pos.any()):
                return base_loss

            scores = self._autoencoder_scores(user_ids)
            rows = torch.arange(B, device=user_ids.device)
            pos_scores = scores[rows, pos_item_ids]
            neg_scores = scores[rows.unsqueeze(1), neg_tensor]
            bpr_loss = F.softplus(
                neg_scores[valid_pos] - pos_scores[valid_pos].unsqueeze(1)
            ).mean()
            return base_loss + loss_weight * bpr_loss

        return custom_loss

    def run(self):
        """Execute configured FlowNS phases."""
        best_score = None
        phase_methods = {
            'rec_pretrain': self.phase1,
            'flow_pretrain': self.phase2,
            'grpo': self.phase3,
            'joint': self.phase4,
        }
        for phase in self.enabled_phases:
            result = phase_methods[phase]()
            if phase in {'rec_pretrain', 'joint'}:
                if isinstance(result, tuple):
                    best_score = result[0]
                elif result is not None:
                    best_score = result
        return best_score

    def _flow_rerank_enabled(self):
        return (
            not self.use_random_neg
            and self.rerank_num_negatives > 0
            and self.rerank_neg_penalty > 0.0
        )

    def _eval_exposed_filter_enabled(self):
        return (
            self.user_neg_items is not None
            and self.eval_exposed_neg_penalty > 0.0
        )

    def _install_flow_reranker(self):
        """Temporarily penalize full-sort scores for flow-generated negatives."""
        flow_enabled = self._flow_rerank_enabled()
        exposed_enabled = self._eval_exposed_filter_enabled()
        if not flow_enabled and not exposed_enabled:
            return lambda: None
        if flow_enabled:
            self._ensure_flow_ready()

        model = self.rec_model
        original_full_sort_predict = model.full_sort_predict
        had_instance_method = 'full_sort_predict' in model.__dict__
        previous_instance_method = model.__dict__.get('full_sort_predict')

        n_items = int(getattr(model, 'n_items', self.dataset.item_num))
        user_emb_all = item_emb = item_norm = None
        if flow_enabled:
            user_emb_all, item_emb = get_embeddings(model)
            self._refresh_mapper(item_emb)
            n_items = item_emb.shape[0]
            item_norm = F.normalize(item_emb, dim=-1)

        n_negatives = self.rerank_num_negatives
        penalty = self.rerank_neg_penalty
        exposed_penalty = self.eval_exposed_neg_penalty
        mapping_strategy = self.rerank_mapping_strategy
        mapping_topk = self.rerank_mapping_topk
        score_mode = self.rerank_score_mode
        candidate_topk = self.rerank_candidate_topk
        if score_mode not in {'mapped_item', 'soft_topk', 'score_topk'}:
            raise ValueError(
                'rerank_score_mode must be "mapped_item", "soft_topk", '
                'or "score_topk"'
            )
        stats = {
            'batches': 0,
            'users': 0,
            'flow_penalized': 0,
            'exposed_penalized': 0,
        }

        logger.info(
            'Eval rerank/filter enabled: flow=%s, exposed=%s, mode=%s, '
            'negatives=%d, flow_penalty=%.4f, exposed_penalty=%.4f, '
            'mapping=%s@%s, candidate_topk=%d',
            flow_enabled, exposed_enabled, score_mode, n_negatives, penalty,
            exposed_penalty, mapping_strategy, mapping_topk, candidate_topk,
        )

        def reranked_full_sort_predict(interaction):
            scores = original_full_sort_predict(interaction)
            scores = scores.view(-1, n_items).clone()
            user_ids = interaction[model.USER_ID]
            batch_size = user_ids.shape[0]
            uid_list = [int(uid) for uid in user_ids.detach().cpu().tolist()]

            if exposed_enabled:
                exposed_count = 0
                for row, uid in enumerate(uid_list):
                    ids = self.user_neg_items.get(uid)
                    if not ids:
                        continue
                    ids = torch.as_tensor(
                        ids, device=scores.device, dtype=torch.long,
                    )
                    ids = ids[(ids >= 0) & (ids < scores.shape[1])]
                    if ids.numel() > 0:
                        scores[row, ids] -= exposed_penalty
                        exposed_count += int(ids.numel())
                stats['exposed_penalized'] += exposed_count

            if not flow_enabled:
                stats['batches'] += 1
                stats['users'] += int(batch_size)
                return scores.view(-1)

            with torch.no_grad():
                u_emb = user_emb_all[user_ids]
                final_emb, _, _ = self.sde_sampler.sample_trajectories(
                    u_emb, n_trajectories=n_negatives,
                )
                forbidden = [
                    self.user_pos_items.get(uid, set())
                    for uid in uid_list
                    for _ in range(n_negatives)
                ]

                if score_mode == 'soft_topk':
                    flat = final_emb.reshape(-1, final_emb.shape[-1])
                    sim = F.normalize(flat, dim=-1) @ item_norm.T
                    sim[:, 0] = -torch.inf
                    for row, ids in enumerate(forbidden):
                        if not ids:
                            continue
                        ids = torch.as_tensor(
                            list(ids), device=sim.device, dtype=torch.long,
                        )
                        ids = ids[(ids >= 0) & (ids < sim.shape[1])]
                        if ids.numel() > 0:
                            sim[row, ids] = -torch.inf
                    k = min(candidate_topk, sim.shape[1])
                    cand_vals, cand_ids = sim.topk(k=k, dim=-1)
                    cand_ids = cand_ids.reshape(batch_size, n_negatives, k)
                    cand_vals = cand_vals.reshape(batch_size, n_negatives, k)
                    cand_vals = cand_vals.clamp_min(0.0)
                elif score_mode == 'score_topk':
                    selection_scores = scores.clone()
                    selection_scores[:, 0] = -torch.inf
                    for row, ids in enumerate(
                        self.user_pos_items.get(uid, set()) for uid in uid_list
                    ):
                        if not ids:
                            continue
                        ids = torch.as_tensor(
                            list(ids),
                            device=selection_scores.device,
                            dtype=torch.long,
                        )
                        ids = ids[(ids >= 0) & (ids < selection_scores.shape[1])]
                        if ids.numel() > 0:
                            selection_scores[row, ids] = -torch.inf
                    k = min(candidate_topk, selection_scores.shape[1])
                    cand_ids = selection_scores.topk(k=k, dim=-1).indices
                    gen_norm = F.normalize(final_emb, dim=-1)
                    cand_emb = item_norm[cand_ids]
                    cand_vals = torch.bmm(
                        cand_emb, gen_norm.transpose(1, 2),
                    ).max(dim=-1).values
                    cand_vals = cand_vals.clamp_min(0.0)
                else:
                    mapper_user_emb = u_emb.unsqueeze(1).expand(
                        batch_size, n_negatives, u_emb.shape[-1],
                    )
                    pos_embs = None
                    if mapping_strategy == 'reward_topk':
                        pos_base = self._get_pos_embs_for_users(uid_list, item_emb)
                        pos_embs = pos_base.unsqueeze(1).expand(
                            batch_size, n_negatives,
                            pos_base.shape[-2], pos_base.shape[-1],
                        )

                    neg_ids = self.mapper.map_to_items(
                        final_emb,
                        forbidden_item_ids=forbidden,
                        exclude_item_ids=(0,),
                        user_emb=mapper_user_emb,
                        pos_item_embs=pos_embs,
                        reward_fn=self.reward_fn,
                        strategy=mapping_strategy,
                        candidate_topk=mapping_topk,
                    )

            rows = torch.arange(batch_size, device=scores.device)
            if score_mode == 'soft_topk':
                penalty_matrix = torch.zeros_like(scores)
                for col in range(n_negatives):
                    penalty_values = penalty * cand_vals[:, col, :].to(scores.dtype)
                    penalty_matrix.scatter_add_(
                        1, cand_ids[:, col, :], penalty_values,
                    )
                scores = scores - penalty_matrix
                penalized_count = int(cand_ids.numel())
            elif score_mode == 'score_topk':
                penalty_values = penalty * cand_vals.to(scores.dtype)
                scores[rows.unsqueeze(1), cand_ids] -= penalty_values
                penalized_count = int(cand_ids.numel())
            else:
                neg_ids = neg_ids.reshape(batch_size, n_negatives)
                for col in range(n_negatives):
                    scores[rows, neg_ids[:, col]] -= penalty
                penalized_count = int(neg_ids.numel())

            stats['batches'] += 1
            stats['users'] += int(batch_size)
            stats['flow_penalized'] += penalized_count
            return scores.view(-1)

        model.full_sort_predict = reranked_full_sort_predict

        def restore():
            if had_instance_method:
                model.full_sort_predict = previous_instance_method
            else:
                delattr(model, 'full_sort_predict')
            logger.info(
                'Eval rerank/filter summary: batches=%d, users=%d, '
                'flow_penalties=%d, exposed_penalties=%d',
                stats['batches'], stats['users'], stats['flow_penalized'],
                stats['exposed_penalized'],
            )

        return restore

    def evaluate(self):
        """Evaluate on test set using the best RecBole checkpoint."""
        rec_trainer = self._rec_trainer
        checkpoint = torch.load(
            rec_trainer.saved_model_file, map_location=self.device, weights_only=False,
        )
        self.rec_model.load_state_dict(checkpoint['state_dict'])
        self.rec_model.load_other_parameter(checkpoint.get('other_parameter'))
        restore_reranker = self._install_flow_reranker()
        try:
            show_progress = bool(self.config['show_progress'])
            result = rec_trainer.evaluate(
                self.test_data, load_best_model=False, show_progress=show_progress,
            )
        finally:
            restore_reranker()
        logger.info(f'Test result: {result}')
        return result
