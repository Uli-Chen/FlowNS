"""
FlowNeg Trainer — extends RecBole's Trainer with CFM training and FAISS rebuild.

Training schedule:
  Phase 1 (epochs 0..W):    Standard warm-up with uniform negatives
  Phase 2 (epochs W..end):  Alternating:
    - Every epoch:  train backbone with FlowNeg-sampled negatives
    - Every K epochs:
      a) Extract embeddings from backbone
      b) Rebuild FAISS index
      c) Fine-tune CFM velocity network
"""

import logging
from time import time

import numpy as np
import torch

from recbole.trainer.trainer import Trainer
from recbole.utils import set_color

from flowneg.cfm_trainer import CFMTrainer
from flowneg.faiss_index import FAISSIndex
from flowneg.utils import tau_schedule

logger = logging.getLogger(__name__)


class FlowNegTrainer(Trainer):
    """Extends RecBole Trainer with FlowNeg's alternating training schedule."""

    def __init__(self, config, model):
        super().__init__(config, model)

        fn_config = config["flowneg"] if "flowneg" in config.final_config_dict else {}

        self.warm_up_epochs = fn_config.get("warm_up_epochs", 5)
        self.cfm_update_every = fn_config.get("cfm_epochs_per_update", 5)
        self.faiss_rebuild_every = fn_config.get("faiss_rebuild_every", 5)
        self.tau_h = fn_config.get("tau_h", 0.5)
        self.tau_h_schedule = fn_config.get("tau_h_schedule", "constant")
        self.tau_h_start = fn_config.get("tau_h_start", 2.0)
        self.tau_h_end = fn_config.get("tau_h_end", 0.1)
        self.d = config["embedding_size"]

        # Initialize CFM trainer
        self.cfm_trainer = CFMTrainer(config, self.device)

        # Initialize FAISS index
        faiss_metric = fn_config.get("faiss_metric", "IP")
        self.faiss_index = FAISSIndex(
            d=self.d, metric=faiss_metric, use_gpu=False
        )

        # Track FlowNeg state
        self._flow_sampler = None
        self._train_dataloader = None

    def _get_embeddings(self):
        """Extract current embeddings from the backbone model."""
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "forward") and hasattr(self.model, "restore_user_e"):
                # LightGCN-style: has graph propagation
                try:
                    user_embs, item_embs = self.model.forward()
                    return user_embs.detach(), item_embs.detach()
                except Exception:
                    pass
            # BPR-style: direct embeddings
            user_embs = self.model.user_embedding.weight.detach()
            item_embs = self.model.item_embedding.weight.detach()
        return user_embs, item_embs

    def _rebuild_faiss(self):
        """Rebuild FAISS index from current item embeddings."""
        user_embs, item_embs = self._get_embeddings()
        item_np = item_embs.cpu().numpy().astype(np.float32)
        self.faiss_index.build(item_np)
        logger.info(
            f"FAISS index rebuilt with {self.faiss_index.ntotal} items"
        )
        return user_embs, item_embs

    def _update_cfm(self):
        """Fine-tune CFM velocity network on current embeddings."""
        user_embs, item_embs = self._get_embeddings()
        avg_loss = self.cfm_trainer.train_step(
            user_embs.cpu(), item_embs.cpu()
        )
        return avg_loss

    def _setup_flow_sampler(self, train_data):
        """Connect flow components to the sampler and load exposure data."""
        # Build ID mappings from the dataset for remapping exposure cache
        dataset = train_data._dataset
        uid_map = None
        iid_map = None
        if hasattr(dataset, "field2token_id"):
            uid_field = dataset.uid_field
            iid_field = dataset.iid_field
            uid_token2id = dataset.field2token_id.get(uid_field, {})
            iid_token2id = dataset.field2token_id.get(iid_field, {})
            if uid_token2id and iid_token2id:
                uid_map = uid_token2id
                iid_map = iid_token2id
                logger.info(
                    f"ID mappings: {len(uid_map)} users, {len(iid_map)} items"
                )

        # Load exposure data with remapping
        self.cfm_trainer.load_exposure_data(uid_map=uid_map, iid_map=iid_map)

        sampler = train_data._sampler
        if hasattr(sampler, "set_flow_components"):
            sampler.set_flow_components(self.cfm_trainer, self.faiss_index)
            sampler.set_model(self.model)
            self._flow_sampler = sampler
            self._train_dataloader = train_data
            logger.info("FlowNeg sampler connected to flow components")
        else:
            logger.warning(
                "Train sampler does not support FlowNeg. "
                "Using standard training."
            )

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        """Override fit to add FlowNeg schedule."""
        # Setup flow sampler connection
        self._setup_flow_sampler(train_data)

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # --- FlowNeg schedule ---
            is_post_warmup = epoch_idx >= self.warm_up_epochs

            # Notify sampler of current epoch
            if hasattr(train_data, "set_epoch"):
                train_data.set_epoch(epoch_idx)

            # Update tau_h if using schedule
            if is_post_warmup and self._flow_sampler is not None:
                current_tau = tau_schedule(
                    epoch_idx - self.warm_up_epochs,
                    self.epochs - self.warm_up_epochs,
                    schedule=self.tau_h_schedule,
                    tau_start=self.tau_h_start if self.tau_h_schedule != "constant" else self.tau_h,
                    tau_end=self.tau_h_end if self.tau_h_schedule != "constant" else self.tau_h,
                )
                self._flow_sampler.tau_h = current_tau

            # Rebuild FAISS and update CFM periodically after warm-up
            if is_post_warmup and self._flow_sampler is not None:
                epochs_since_warmup = epoch_idx - self.warm_up_epochs
                if epochs_since_warmup % self.faiss_rebuild_every == 0:
                    t0 = time()
                    user_embs, item_embs = self._rebuild_faiss()
                    self._flow_sampler.update_embeddings(user_embs, item_embs)
                    rebuild_time = time() - t0
                    if verbose:
                        self.logger.info(
                            set_color(f"FAISS rebuild", "cyan")
                            + f" at epoch {epoch_idx} [{rebuild_time:.1f}s]"
                        )

                if epochs_since_warmup % self.cfm_update_every == 0:
                    t0 = time()
                    cfm_loss = self._update_cfm()
                    cfm_time = time() - t0
                    if verbose:
                        self.logger.info(
                            set_color(f"CFM update", "cyan")
                            + f" at epoch {epoch_idx}: loss={cfm_loss:.4f} [{cfm_time:.1f}s]"
                        )

            if is_post_warmup and epoch_idx == self.warm_up_epochs:
                if verbose:
                    self.logger.info(
                        set_color("FlowNeg activated", "yellow")
                        + f" — switching from uniform to flow-based sampling"
                    )

            # --- Standard training epoch ---
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # --- Validation ---
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                from recbole.utils import early_stopping, calculate_valid_score, dict2str

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result
