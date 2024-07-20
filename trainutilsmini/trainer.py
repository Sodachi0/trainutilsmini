import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
from typing import Any, Callable
from pathlib import Path
from rich.progress import Progress, TimeElapsedColumn

from .data import DataTensor
from .history import History
from .metric import Metric

class Trainer:
    def __init__(self,
                 path: str,
                 models: dict[str, nn.Module] | nn.Module,
                 losses: dict[str, nn.Module] | nn.Module,
                 computed_losses_names: list[str],
                 optimizers: dict[str, Optimizer] | Optimizer,
                 scheduler: dict[str, Any] | Any | None,
                 metrics: dict[str, Metric] | Metric | None,
                 config: dict[str, Any]):
        
        assert isinstance(path, str), "Invalid path parameter."
        assert all([n not in metrics for n in computed_losses_names]), 'Computed losses names and metrics names should be distinct.'
        
        Path(path).mkdir(parents=True, exist_ok=True)
        self.path = path

        self.models = None
        if isinstance(models, dict):
            self.models = {key: models[key](**config["models"][key]) for key in models}
        else:
            self.models = {"model": models(**config["models"])}

        self.losses = None
        if isinstance(losses, dict):
            self.losses = {key: losses[key](**config["losses"][key]) for key in losses}
        else:
            self.losses = {"loss": losses(**config["losses"])}

        self.computed_losses_names = computed_losses_names

        self.optimizers = None
        get_net_params = lambda name: (self.models[name]\
                                       if name in self.models
                                       else self.losses[name]).parameters()
        if isinstance(optimizers, dict):
            self.optimizers = {}
            for key in optimizers:
                opt_conf = config["optimizers"][key]
                if isinstance(opt_conf['params'], list):
                    for idx, params in enumerate(opt_conf['params']):
                        if isinstance(params, dict):
                            params['params'] = get_net_params(params['params'])
                        else:
                            opt_conf['params'][idx] = {'params': get_net_params(params)}
                else:
                    opt_conf['params'] = get_net_params(opt_conf['params'])

                self.optimizers[key] = optimizers[key](**opt_conf)
        else:
            opt_conf = config["optimizers"]
            if isinstance(opt_conf['params'], list):
                for idx, params in enumerate(opt_conf['params']):
                    if isinstance(params, dict):
                        params['params'] = get_net_params(params['params'])
                    else:
                        opt_conf['params'][idx] = {'params': get_net_params(params)}
            else:
                opt_conf['params'] = get_net_params(opt_conf['params'])
            self.optimizers = {"optimizer": optimizers(**opt_conf)}

        self.scheduler = {}
        if scheduler is not None:
            if isinstance(scheduler, dict):
                self.scheduler =  {}
                for key in scheduler:
                    sch_conf = config["scheduler"][key]
                    sch_conf['optimizer'] = self.optimizers[sch_conf['optimizer']]
                    self.scheduler[key] = scheduler[key](**sch_conf)
            else:
                sch_conf = config["scheduler"]
                sch_conf['optimizer'] = self.optimizers[sch_conf['optimizer']]
                self.scheduler = {"scheduler": scheduler(**sch_conf)}

        self.metrics = {}
        if metrics is not None:
            if isinstance(metrics, dict):
                self.metrics = {key: metrics[key](**config["metrics"][key]) for key in metrics}
            else:
                self.metrics = {"metric": metrics(**config["metrics"])}

        self.history = None
        self.state_dic = {0: "", 1: "(Overfit)", 2: "(Best)"}

        self.eval_mode()
        self.is_training = False
        self.train_state = 0
        self.overfit_count = 0
        self.max_overfit = 3
        self.computed_losses = None
        self.best_val_losses = torch.tensor(torch.inf)
        self.old_train_losses = torch.tensor(torch.inf)
        self.old_val_losses = torch.tensor(torch.inf)
        self.device = "cpu"

    def set_device(self, device: str | None = None) -> None:
        assert device in [None, "default", "cpu", "cuda"]

        if device is None or device == "default":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.load_on_device(self.device)

    def load_on_device(self, device: str) -> None:
        if device == "cpu":
            return
        raise NotImplementedError()

    def forward(self, x: DataTensor) -> DataTensor:
        raise NotImplementedError()

    def train_mode(self) -> None:
        for m in self.models.values():
            m.train()
        self.is_training = True

    def eval_mode(self) -> None:
        for m in self.models.values():
            m.eval()
        self.is_training = False

    def step(self, x: DataTensor, y: DataTensor) -> torch.Tensor:
        y_hat = self.forward(x)
        self.computed_losses = self.compute_losses(y_hat, y)
        self.store_metrics(y_hat, y)

        return self.reduce_losses()

    def compute_losses(self, y_hat: DataTensor, y: DataTensor) -> DataTensor:
        raise NotImplementedError()
    
    def reduce_losses(self) -> torch.Tensor:
        if isinstance(self.computed_losses, torch.Tensor)\
            and len(self.computed_losses.shape) <= 1:
            return self.computed_losses
        raise NotImplementedError()
    
    def backward(self, computed_losses: DataTensor) -> None:
        raise NotImplementedError()

    def optimize(self) -> None:
        for o in self.optimizers.values():
            o.zero_grad()
        self.backward(self.computed_losses)
        for o in self.optimizers.values():
            o.step()

    def schedule(self) -> None:
        for s in self.scheduler.values():
            s.step()

    def store_metrics(self, y_hat: DataTensor, y: DataTensor) -> None:
        for m in self.metrics.values():
            m.store(y_hat, y)

    def compute_metrics(self) -> torch.Tensor:
        computed_metrics = []
        for m in self.metrics.values():
            computed_metrics.append(m.compute())
        return computed_metrics
    
    def check_overfit(self, train_losses: torch.Tensor, val_losses: torch.Tensor) -> bool:
        
        self.train_state = 0
        if (val_losses > self.old_val_losses).all() and\
            (train_losses < self.old_train_losses).all():
            self.overfit_count += 1
            self.train_state = 1
        else:
            self.overfit_count = 0

        if (val_losses < self.best_val_losses).all():
            self.best_val_losses = val_losses
            self.train_state = 2

        self.old_train_losses = train_losses.detach()
        self.old_val_losses = val_losses.detach()

        return self.overfit_count >= self.max_overfit
    
    def to_history(self, epoch: int, train_metrics: torch.Tensor, val_metrics: torch.Tensor) -> None:
        self.history.train['losses'][epoch] = self.old_train_losses.cpu()
        self.history.val['losses'][epoch] = self.old_val_losses.cpu()
        self.history.state[epoch] = self.train_state
        for idx, m in enumerate(self.metrics):
            self.history.train[m].append(train_metrics[idx])
            self.history.val[m].append(val_metrics[idx])

    def train(self, nepoch: int, train_loader: Dataset, val_loader: Dataset, monitor: Callable[[int, History], Any] | None = None) -> History:
        print("Device:", self.device)

        progress_bar = lambda : Progress(
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        )

        mean: Callable[[torch.Tensor], torch.Tensor] = lambda t: sum(t) / len(t)

        self.history = History(nepoch, self.computed_losses_names, list(self.metrics.keys()))

        with progress_bar() as p:
            t1 = p.add_task("[bold]Epochs", total=nepoch - 1)

            for epoch in range(nepoch):
                t2 = p.add_task("[green]Train", total=len(train_loader))
                t3 = p.add_task("[red]Valid", total=len(val_loader))

                self.train_mode()
                train_losses = []
                for x, y in train_loader:
                    losses = self.step(x, y)
                    train_losses.append(losses)
                    self.optimize()
                    p.advance(t2)
                train_losses = mean(train_losses)
                train_metrics = self.compute_metrics()

                self.eval_mode()
                val_losses = []
                for x, y in val_loader:
                    losses = self.step(x, y)
                    val_losses.append(losses)
                    p.advance(t3)
                val_losses = mean(val_losses)
                val_metrics = self.compute_metrics()

                exit = self.check_overfit(train_losses, val_losses)

                self.to_history(epoch, train_metrics, val_metrics)
                if monitor is not None:
                    monitor(epoch, self.history)

                if exit:
                    break

                if self.train_state == 2:
                    self.save()

                self.schedule()

                p.remove_task(t2)
                p.remove_task(t3)
                p.advance(t1)
        
        return self.history

    def test(self, test_loader: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        print("Device:", self.device)

        progress_bar = lambda : Progress(
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        )

        mean: Callable[[torch.Tensor], torch.Tensor] = lambda t: sum(t) / len(t)

        with progress_bar() as p:
            t = p.add_task("[bold]Test", total=len(test_loader))

            self.eval_mode()
            test_losses = []
            for data in test_loader:
                losses = self.step(data)
                test_losses.append(losses)
                p.advance(t)
            test_losses = mean(test_losses)
            test_metrics = self.compute_metrics()

        return test_losses.cpu(), test_metrics
    
    def simple_monitor(epoch: int, history: History) -> None:
        state_dic = {0: "", 1: "(Overfit)", 2: "(Best)"}
        print(f"Epoch {epoch}: {state_dic[history.state[epoch]]}\n" +\
              f"Train loss: {history.train['losses'][epoch]}\n" +\
              "\n".join([f"{key}: {history.train[key][epoch]}" for key in history.metrics_names]) +\
              "\n----------\n" +\
              f"Val loss: {history.val['losses'][epoch]}\n" +\
              "\n".join([f"{key}: {history.val[key][epoch]}" for key in history.metrics_names]) +\
              "\n")
             
    def save(self) -> None:
        for k, m in self.models.items():
            torch.save(m.state_dict(), self.path + k + '.pth')

    def load(self) -> None:
        for k, m in self.models.items():
            m.load_state_dict(torch.load(self.path + k + '.pth'))
            m.eval()

    # def count_parameters(self):
    #     params = sum(p.numel() for p in self.parameters())
    #     train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     return params, train_params
