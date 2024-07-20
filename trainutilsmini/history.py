import numpy as np
from typing import Any

class ModeDescriptor:
    def __init__(self, mode: str):
        assert mode in ('train', 'val')
        self.train = mode == 'train'

    def __get__(self, obj, objtype=None):
        obj._train = self.train
        return obj

class History:
    train = ModeDescriptor('train')
    val = ModeDescriptor('val')

    def __init__(self, nepoch: int, losses_names: list[str], metrics_names: list[str]):
        assert 'losses' not in losses_names, "'losses' is not a valid loss name."
        self.nepoch = nepoch
        self._train = True

        self.losses_names = losses_names
        self.metrics_names = metrics_names
        loss_size = len(losses_names)

        self._train_losses: np.ndarray = np.full((nepoch, loss_size), np.nan)
        self._train_metrics: dict[str, list[Any]] = {key: [] for key in metrics_names}
        
        self._val_losses: np.ndarray = np.full((nepoch, loss_size), np.nan)
        self._val_metrics: dict[str, list[Any]] = {key: [] for key in metrics_names}
        
        self.state: np.ndarray = np.full(nepoch, np.nan)

    def __getitem__(self, key: str) -> Any:
        if self._train:
            if key in self.losses_names:
                return self._train_losses[:, self.losses_names.index(key)]
            elif key == 'losses':
                return self._train_losses
            elif key in self.metrics_names:
                return self._train_metrics[key]
        else:
            if key in self.losses_names:
                return self._val_losses[:, self.losses_names.index(key)]
            elif key == 'losses':
                return self._val_losses
            elif key in self.metrics_names:
                return self._val_metrics[key]
        raise KeyError(f'{key} is an unknown loss or metric.')
