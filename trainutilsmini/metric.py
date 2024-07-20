import torch
import numpy as np

from .data import DataTensor

class Metric:
    def __init__(self):
        self.result = []

    def store(self, y_hat: DataTensor, y: DataTensor) -> None:
        self.result.append(self.compute_bash(y_hat, y))

    def compute(self) -> torch.Tensor| np.ndarray | float:
        t = torch.concat(self.result)
        t = self.reduce(t)
        self.result = []
        return t

    def compute_bash(self, y_hat: DataTensor, y: DataTensor) -> torch.Tensor:
        raise NotImplementedError()

    def reduce(self, t: torch.Tensor) -> torch.Tensor | np.ndarray | float:
        raise NotImplementedError()
