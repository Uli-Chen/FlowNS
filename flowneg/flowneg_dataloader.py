"""
FlowNeg DataLoader — minimal extension of RecBole's TrainDataLoader.

The only difference is that it notifies the sampler of the current epoch
so the sampler can switch between warm-up (uniform) and flow-based sampling.
"""

import numpy as np
from recbole.data.dataloader.general_dataloader import TrainDataLoader


class FlowNegTrainDataLoader(TrainDataLoader):
    """Training dataloader that supports FlowNeg's epoch-aware sampling."""

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self._current_epoch = 0

    def set_epoch(self, epoch):
        """Notify the sampler of current epoch for warm-up logic."""
        self._current_epoch = epoch
        if hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(epoch)

    def collate_fn(self, index):
        """Same as parent — the sampling logic is in the sampler."""
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return self._neg_sampling(transformed_data)
