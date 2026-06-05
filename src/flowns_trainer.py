import copy
import logging
import torch
import numpy as np

from recbole.data.interaction import Interaction
from recbole.trainer import Trainer
from recbole.evaluator import Evaluator, Collector
from recbole.utils import calculate_valid_score, early_stopping

from .flow_model import ConditionalFlowModel
from .sde_sampler import SDESampler
from .reward import BoundaryAwareReward
from .grpo import GRPOTrainer
from .neg_sampling import EmbeddingToItemMapper
from .recbole_utils import get_embeddings, get_user_positive_items

logger = logging.getLogger(__name__)


class FlowNSTrainer:
    """Three-phase FlowNS trainer composing RecBole Trainer with flow + GRPO."""

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

        fc = flow_config or {}
        emb_dim = config['embedding_size']

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
        )

        self.reward_fn = BoundaryAwareReward(
            a=fc.get('reward_a', 1.0),
            gamma=fc.get('reward_gamma', 1.0),
        )

        self.mapper = None  # initialized after Phase 1

        self.grpo_trainer = None  # initialized after Phase 1

        self.flow_pretrain_epochs = fc.get('flow_pretrain_epochs', 50)
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
        self.eval_step = fc.get('eval_step', 5)
        self.stopping_step = fc.get('stopping_step', 10)
        self.disable_early_stopping = fc.get('disable_early_stopping', False)
        self.max_pos_samples = fc.get('max_pos_samples', 10)
        self.use_random_neg = fc.get('use_random_neg', False)
        self.rec_checkpoint_path = fc.get('rec_checkpoint_path')

        # Match RecBole's train sampler: training negatives only exclude
        # positives in the train split, not future valid/test positives.
        self.user_pos_items = get_user_positive_items(train_data)

    def phase1(self):
        """Phase 1: Standard RecBole training, optionally followed by flow pretrain."""
        if self.rec_checkpoint_path:
            logger.info('=== Phase 1: Load paired RecBole checkpoint ===')
        elif self.use_random_neg:
            logger.info('=== Phase 1: RNS pretrain ===')
        else:
            logger.info('=== Phase 1: RecBole training + Flow pretrain ===')

        # (a) Train recommender with RecBole
        rec_trainer = Trainer(self.config, self.rec_model)
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
            best_score, best_result = rec_trainer.fit(
                self.train_data, self.valid_data, show_progress=True
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

        # (b) Random-negative control does not use flow, so keep M3b to
        # RNS pretrain + RNS finetune only.
        if self.use_random_neg:
            logger.info('Phase 1: RNS control enabled; skipping flow pretrain.')
            return best_score, best_result

        # (c) Pretrain flow
        user_emb, item_emb = get_embeddings(self.rec_model)
        self.flow_model.pretrain(
            user_emb, item_emb, self.user_pos_items,
            epochs=self.flow_pretrain_epochs,
            lr=self.flow_lr,
        )

        # (d) Initialize mapper and GRPO trainer
        self.mapper = EmbeddingToItemMapper(item_emb)
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
        )

        return best_score, best_result

    def phase2(self):
        """Phase 2: GRPO fine-tuning of flow model."""
        if self.use_random_neg:
            logger.info('=== Phase 2 skipped: RNS control has no flow fine-tuning ===')
            return
        if self.grpo_epochs <= 0:
            logger.info('=== Phase 2 skipped: grpo_epochs <= 0 ===')
            return

        logger.info('=== Phase 2: GRPO fine-tuning ===')
        user_emb, item_emb = get_embeddings(self.rec_model)
        self.mapper.update_embeddings(item_emb)

        self.grpo_trainer.train(
            user_emb, item_emb, self.user_pos_items,
            epochs=self.grpo_epochs,
            batch_size=self.grpo_batch_size,
        )

    def phase3(self):
        """Phase 3: Joint alternating training."""
        if self.use_random_neg:
            logger.info('=== Phase 3: RNS finetune ===')
        else:
            logger.info('=== Phase 3: Joint training ===')
        rec_trainer = self._rec_trainer

        phase1_score = getattr(self, '_phase1_best_score', None)
        best_valid_score = float(phase1_score) if phase1_score is not None else -np.inf
        cur_step = 0

        for epoch in range(self.joint_rec_epochs):
            # (a) Prepare flow state only for flow-generated negatives.
            if self.use_random_neg:
                user_emb = item_emb = None
            else:
                user_emb, item_emb = get_embeddings(self.rec_model)
                self.mapper.update_embeddings(item_emb)

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
                    f'Phase 3 epoch {epoch+1}/{self.joint_rec_epochs}, '
                    f'loss={train_loss}, valid={valid_score:.4f}'
                )

                best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, best_valid_score, cur_step,
                    max_step=self.stopping_step,
                    bigger=self.config['valid_metric_bigger'],
                )
                if update_flag:
                    rec_trainer._save_checkpoint(epoch)
                if stop_flag and not self.disable_early_stopping:
                    logger.info(f'Phase 3 early stopping at epoch {epoch+1}')
                    break

            # (c) GRPO update every joint_grpo_freq epochs
            if (
                not self.use_random_neg
                and self.joint_grpo_steps > 0
                and self.joint_grpo_freq > 0
                and (epoch + 1) % self.joint_grpo_freq == 0
            ):
                user_emb, item_emb = get_embeddings(self.rec_model)
                self.mapper.update_embeddings(item_emb)
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
        """
        model = self.rec_model
        sde_sampler = self.sde_sampler
        mapper = self.mapper
        use_random = self.use_random_neg
        user_pos = self.user_pos_items

        def custom_loss(interaction):
            user_ids = interaction[model.USER_ID]

            if use_random:
                # RecBole has already added neg_item_id in the train dataloader.
                # Reusing it makes the RNS control byte-for-byte the same loss path
                # as standard RecBole training.
                return model.calculate_loss(interaction)
            else:
                with torch.no_grad():
                    u_emb = user_emb_all[user_ids]
                    final_emb, _, _ = sde_sampler.sample_trajectories(
                        u_emb, n_trajectories=1,
                    )
                    final_emb = final_emb.squeeze(1)
                    forbidden = [
                        user_pos.get(int(uid), set()) for uid in user_ids.tolist()
                    ]
                    neg_tensor = mapper.map_to_items(
                        final_emb,
                        forbidden_item_ids=forbidden,
                        exclude_item_ids=(0,),
                    )

            new_inter = copy.deepcopy(interaction)
            new_inter.update(Interaction({model.NEG_ITEM_ID: neg_tensor}))
            return model.calculate_loss(new_inter)

        return custom_loss

    def run(self):
        """Execute FlowNS, or RNS-only control when use_random_neg is enabled."""
        self.phase1()
        if not self.use_random_neg:
            self.phase2()
        best_score = self.phase3()
        return best_score

    def evaluate(self):
        """Evaluate on test set using RecBole (loads best Phase 3 checkpoint)."""
        rec_trainer = self._rec_trainer
        checkpoint = torch.load(
            rec_trainer.saved_model_file, map_location=self.device, weights_only=False,
        )
        self.rec_model.load_state_dict(checkpoint['state_dict'])
        self.rec_model.load_other_parameter(checkpoint.get('other_parameter'))
        result = rec_trainer.evaluate(
            self.test_data, load_best_model=False, show_progress=True
        )
        logger.info(f'Test result: {result}')
        return result
