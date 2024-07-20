import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import random_split

from trainutilsmini import *

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class NetTrainer(Trainer):
    def load_on_device(self, device: str) -> None:
        if device == "cpu":
            return
        elif device == "cuda":
            for m in self.models:
                self.models[m] = self.models[m].to(self.device)
        else:
            raise NotImplementedError()

    def forward(self, x: DataTensor) -> DataTensor:
        return self.models['model'](x.to(self.device))
    
    def compute_losses(self, y_hat: DataTensor, y: DataTensor) -> DataTensor:
        return self.losses['loss'](y_hat, y.to(self.device))
    
    def backward(self, computed_losses: DataTensor) -> None:
        computed_losses.backward()

class Macc(Metric):
    def compute_bash(self, y_hat: DataTensor, y: DataTensor) -> torch.Tensor:
        return (y_hat.argmax(dim=1).cpu() == y)

    def reduce(self, t: torch.Tensor) -> float:
        return t.float().mean().item()

class Mcm(Metric):
    def __init__(self, normalize: str = 'all'):
        super().__init__()
        self.normalize = normalize
        self.classes = None

    def compute_bash(self, y_hat: DataTensor, y: DataTensor) -> torch.Tensor:
        self.classes = np.arange(y_hat.shape[1])
        self.compute_bash = self.__compute_bash
        return torch.stack((y, y_hat.argmax(dim=1).cpu()))
    
    def __compute_bash(self, y_hat: DataTensor, y: DataTensor) -> torch.Tensor:
        return torch.stack((y, y_hat.argmax(dim=1).cpu()))

    def reduce(self, t: torch.Tensor) -> np.ndarray:
        return confusion_matrix(t[0], t[1], normalize=self.normalize, labels=self.classes)
    
class ExampleMonitor(Monitor):
    @metric_init("Accuracy")
    def init_acc(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[], y=[], name='Train',
            line={'color': 'gray'}
        ))
        fig.add_trace(go.Scatter(
            x=[], y=[], name='Validation',
            line={'color': 'orange'}
        ))
        fig.add_vline(x=0,
                    line_width=2,
                    line_color="green",
                    annotation_text="",
                    annotation_bgcolor="#000000",
                    name='best',
                    showlegend=True)
        fig.add_vrect(x0=0, 
                    x1=0, 
                    fillcolor="red",
                    opacity=0.25, 
                    line_width=0,
                    name='overfit',
                    showlegend=True)
        fig.update_layout(title="Accuracy", xaxis={'title': 'Epoch'}, template='plotly_dark')
        return fig
    
    @metric_updater("Accuracy")
    def update_acc(self, fig: go.Figure) -> go.Figure:
        fig.data[0].x = self.x_axis
        fig.data[0].y = self.history.train['Accuracy']
        fig.data[1].x = self.x_axis
        fig.data[1].y = self.history.val['Accuracy']
        if self.history.state[self.epoch] == 2:
            fig.layout.shapes[0].x0 = self.epoch
            fig.layout.shapes[0].x1 = self.epoch
            fig.layout.annotations[0].text=f"<b>Train: {self.history.train['Accuracy'][self.epoch]:.4f}<br>Val: {self.history.val['Accuracy'][self.epoch]:.4f}</b>"
        elif self.history.state[self.epoch] == 1:
            fig.add_vrect(
                x0=self.epoch-0.5,
                x1=self.epoch+0.5,
                fillcolor="red",
                opacity=0.25,
                line_width=0,
            )
        return fig
    
    @metric_slider("Confusion Matrix")
    def update_mcm(self, idx: int) -> matplotlib.figure.Figure:
        f, axes = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
        disp0 = ConfusionMatrixDisplay(self.history.train['Confusion matrix'][idx]).plot(ax=axes[0], values_format=".1%", text_kw={'size': 5})
        disp1 = ConfusionMatrixDisplay(self.history.val['Confusion matrix'][idx]).plot(ax=axes[1], values_format=".1%", text_kw={'size': 5})
        disp0.ax_.set_title('Train')
        disp1.ax_.set_title('Validation')
        disp0.im_.colorbar.remove()
        disp1.im_.colorbar.remove()
        disp1.ax_.set_ylabel('')
        f.colorbar(disp1.im_, ax=axes)
        return f

if __name__ == "__main__":
    fix_seed(2024)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 16

    trainset = torchvision.datasets.CIFAR10(root='./example/data/', train=True, download=True, transform=transform)
    trainset, valset = random_split(trainset, [0.8, 0.2])
    testset = torchvision.datasets.CIFAR10(root='./example/data/', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    config = {
        "models": {},
        "losses": {},
        "optimizers": {'params': 'model', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.001},
        "scheduler": {'optimizer': 'optimizer', 'step_size': 1, 'gamma': 0.9},
        "metrics": {'Accuracy': {}, 'Confusion matrix': {}}
    }

    computed_losses_names = ['Cross Entropy']

    simple_trainer = NetTrainer('./example/mdoel/',
                         Net,
                         nn.CrossEntropyLoss,
                         computed_losses_names,
                         optim.SGD,
                         optim.lr_scheduler.StepLR,
                         {'Accuracy': Macc, 'Confusion matrix': Mcm},
                         config)

    # Use Trainer.simple_monitor for a simplier alternative
    simple_monitor = ExampleMonitor(computed_losses_names)
    simple_monitor.start()

    if torch.cuda.is_available():
        simple_trainer.set_device('cuda')
    simple_trainer.train(50, trainloader, valloader, simple_monitor)
    
    input()
    simple_monitor.stop()
