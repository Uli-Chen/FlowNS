from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from reinforcens.checkpoint import load_checkpoint, save_checkpoint
from reinforcens.config import TrainConfig
from reinforcens.data import InteractionData, SPLIT_FILENAMES
from reinforcens.models import BarycentricGenerator
from reinforcens.trainer import Trainer
from test_data import write_tiny_dataset


def test_ehrns_rejects_unsupported_full_catalog_sampling() -> None:
    with np.testing.assert_raises_regex(
        ValueError, "requires reduced candidate sampling"
    ):
        TrainConfig(model="ehrns", reduced=False)


def test_all_trainers_smoke(tmp_path: Path) -> None:
    data = InteractionData.load(write_tiny_dataset(tmp_path), num_users=3, num_items=10)
    for model in (
        "bpr",
        "dns",
        "kbgan",
        "rns",
        "eprns",
        "berns",
        "cberns",
        "ehrns",
    ):
        config = TrainConfig(
            model=model,
            num_users=3,
            num_items=10,
            embedding_dim=4,
            batch_size=2,
            epochs=1,
            candidates=4,
            exposure_candidates=1,
            dns_candidates=4,
            sigma_range=(1.0, 2.0),
            recommendation_list_length=5,
            eval_batch_size=2,
            device="cpu",
            drop_last=False,
        )
        trainer = Trainer(data, config)
        before = {
            name: value.detach().clone()
            for name, value in trainer.discriminator.state_dict().items()
        }
        generator_before = (
            None
            if trainer.generator is None
            else {
                name: value.detach().clone()
                for name, value in trainer.generator.state_dict().items()
            }
        )
        stats = trainer.train_epoch(1)
        assert stats.examples == 4
        assert np.isfinite(stats.discriminator_loss)
        assert any(
            not torch.equal(before[name], value)
            for name, value in trainer.discriminator.state_dict().items()
        )
        if generator_before is not None:
            assert trainer.generator is not None
            assert any(
                not torch.equal(generator_before[name], value)
                for name, value in trainer.generator.state_dict().items()
            )
            if isinstance(trainer.generator, BarycentricGenerator):
                for name, value in trainer.generator.exposure_prior.state_dict().items():
                    assert torch.equal(
                        generator_before[f"exposure_prior.{name}"], value
                    )
        result = trainer.evaluate("validation")
        assert 0.0 <= result.primary <= 1.0
        assert 0.0 <= result.ndcg <= 1.0


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    data = InteractionData.load(
        write_tiny_dataset(data_path), num_users=3, num_items=10
    )
    for model in ("rns", "berns", "cberns", "ehrns"):
        config = TrainConfig(
            model=model,
            num_users=3,
            num_items=10,
            embedding_dim=4,
            batch_size=2,
            epochs=1,
            candidates=4,
            exposure_candidates=1,
            recommendation_list_length=5,
            device="cpu",
        )
        trainer = Trainer(data, config)
        target = tmp_path / f"{model}.pt"
        save_checkpoint(
            target,
            config=config,
            discriminator=trainer.discriminator,
            generator=trainer.generator,
            epoch=3,
            metrics={"auc": 0.5},
        )
        loaded_config, discriminator, generator, metadata = load_checkpoint(target)
        assert loaded_config.to_dict() == config.to_dict()
        assert generator is not None and metadata["epoch"] == 3
        for key, value in trainer.discriminator.state_dict().items():
            assert torch.equal(value, discriminator.state_dict()[key])
        assert trainer.generator is not None
        for key, value in trainer.generator.state_dict().items():
            assert torch.equal(value, generator.state_dict()[key])


def test_density_pretraining_does_not_read_evaluation_splits(tmp_path: Path) -> None:
    roots = [write_tiny_dataset(tmp_path / name) for name in ("left", "right")]
    (roots[1] / SPLIT_FILENAMES["validation"]).write_text(
        "2,0,1,event_click,0\n2,1|2|3,2,list_show,0\n", encoding="utf-8"
    )
    (roots[1] / SPLIT_FILENAMES["test"]).write_text(
        "2,4,1,event_click,0\n2,5|6|7,2,list_show,0\n", encoding="utf-8"
    )
    for model in ("eprns", "ehrns"):
        states: list[dict[str, torch.Tensor]] = []
        for root in roots:
            data = InteractionData.load(root, num_users=3, num_items=10)
            config = TrainConfig(
                model=model,
                num_users=3,
                num_items=10,
                embedding_dim=4,
                batch_size=2,
                epochs=1,
                candidates=4,
                exposure_candidates=1,
                sigma_range=(1.0, 2.0),
                recommendation_list_length=5,
                eval_batch_size=2,
                device="cpu",
                drop_last=False,
                generator_pretrain_epochs=2,
                generator_pretrain_batch_size=2,
                generator_pretrain_negatives=2,
                generator_validation_fraction=0.25,
                generator_validation_examples=2,
            )
            trainer = Trainer(data, config)
            records = trainer.pretrain_generator_density()
            assert len(records) == 2 and trainer.generator is not None
            if isinstance(trainer.generator, BarycentricGenerator):
                assert not any(
                    parameter.requires_grad
                    for parameter in trainer.generator.exposure_prior.parameters()
                )
            states.append(
                {
                    name: value.detach().clone()
                    for name, value in trainer.generator.state_dict().items()
                }
            )
        for key in states[0]:
            assert torch.equal(states[0][key], states[1][key])


def test_itempop_and_topk_evaluation(tmp_path: Path) -> None:
    data = InteractionData.load(write_tiny_dataset(tmp_path), num_users=3, num_items=10)
    config = TrainConfig(
        model="itempop",
        num_users=3,
        num_items=10,
        epochs=0,
        eval_mode="topk",
        top_k=3,
        eval_batch_size=2,
        device="cpu",
    )
    result = Trainer(data, config).evaluate("test")
    assert result.primary_name == "HR"
    assert 0.0 <= result.primary <= 1.0
    assert 0.0 <= result.ndcg <= 1.0
