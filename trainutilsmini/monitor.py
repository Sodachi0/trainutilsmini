import numpy as np
import panel as pn
import plotly.graph_objs as go
from collections import defaultdict

from .history import History

def metric_init(name: str):
    def inner(func):
        func._is_init = True
        func._metric_name = name
        return func
    return inner

def metric_updater(name: str):
    def inner(func):
        func._is_updater = True
        func._metric_name = name
        return func
    return inner

def metric_slider(name: str):
    def inner(func):
        func._is_slider = True
        func._metric_name = name
        return func
    return inner

class Monitor:
    to_init = {}
    to_updater = {}
    to_slider = {}

    def __init_subclass__(cls, **kwargs):
        cls.to_init = {}
        cls.to_updater = {}
        cls.to_slider = {}
        for method in cls.__dict__.values():
            if callable(method):
                if getattr(method, '_is_init', False):
                    cls.to_init[method._metric_name] = method
                if getattr(method, '_is_updater', False):
                    cls.to_updater[method._metric_name] = method
                if getattr(method, '_is_slider', False):
                    cls.to_slider[method._metric_name] = method

    def __init__(self, losses_names: list[str]):
        self.lossses_name = losses_names
        self.server = None
        self.epoch = -1
        self.history = None
        
        self.losses_fig = {}
        self.metrics_up_fig = {}
        self.metrics_sl_fig = {}
        self.slider_memory = {}

        for loss in self.lossses_name:
            self.losses_fig[loss] = pn.panel(self.init_loss_fig(loss), sizing_mode='stretch_width')

        for metric, metric_init in self.to_init.items():
            self.metrics_up_fig[metric] = pn.panel(metric_init(self), sizing_mode='stretch_width')

        for metric, metric_init in self.to_slider.items():
            self.slider_memory[metric] = defaultdict(lambda: None)

            def slider_callback(idx: int) -> pn.viewable:
                if idx < 0:
                    return f"## Epoch {idx} is invalid"
                elif idx > self.epoch:
                    return f"## Epoch {idx} has not been processed yet..."
                
                if self.slider_memory[metric][idx] is None:
                    self.slider_memory[metric][idx] = self.to_slider[metric](self, idx)
                return pn.panel(self.slider_memory[metric][idx], sizing_mode='stretch_width')

            slider = pn.widgets.IntSlider(name='Epoch', start=0, end=0, step=1, sizing_mode='stretch_width')
            bind = pn.bind(slider_callback, idx=slider)
            self.metrics_sl_fig[metric] = pn.Column(bind, slider)

    def __call__(self, epoch: int, history: History) -> None:
        self.update_data(epoch, history)

    def update_data(self, epoch: int, history: History) -> None:
        self.epoch = epoch
        self.history = history
        self.update()

    def update(self) -> None:
        self.x_axis = np.arange(self.epoch + 1)
        
        for loss, loss_fig in self.losses_fig.items():
            self.update_loss_fig(loss, loss_fig.object)

        for metric, metric_fig in self.metrics_up_fig.items():
            self.metrics_up_fig[metric].object = self.to_updater[metric](self, metric_fig.object)

        for metric_fig in self.metrics_sl_fig.values():
            metric_fig[1].param.trigger('value')
            metric_fig[1].end = self.epoch

    def init_loss_fig(self, loss: str) -> go.Figure:
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
        fig.update_layout(title=loss, xaxis={'title': 'Epoch'}, template='plotly_dark')
        return fig
    
    def update_loss_fig(self, loss: str, fig: go.Figure) -> None:
        fig.data[0].x = self.x_axis
        fig.data[0].y = self.history.train[loss]
        fig.data[1].x = self.x_axis
        fig.data[1].y = self.history.val[loss]
        if self.history.state[self.epoch] == 2:
            fig.layout.shapes[0].x0 = self.epoch
            fig.layout.shapes[0].x1 = self.epoch
            fig.layout.annotations[0].text=f"<b>Train: {self.history.train[loss][self.epoch]:.4f}<br>Val: {self.history.val[loss][self.epoch]:.4f}</b>"
        elif self.history.state[self.epoch] == 1:
            fig.add_vrect(
                x0=self.epoch-0.5,
                x1=self.epoch+0.5,
                fillcolor="red",
                opacity=0.25,
                line_width=0,
            )

    def get_page(self) -> pn.template.FastListTemplate:
        stylesheet=(
            ":host {--accent-fill-hover: transparent;}"
            ".bk-input {border: none;}"
            ".bk-input:focus {border: none; box-shadow: none;}"
            ".bk-input option {"
                "background-color: transparent;"
                "color: var(--neutral-foreground-rest);"
                "font-size: 20px;"
                "font-weight: bold;"
                "padding: 4px;"
                "padding-left: 6px;"
            "}"
            ".bk-input option:checked {"
                "background: linear-gradient(var(--neutral-fill-rest), var(--neutral-fill-rest));"
            "}"
        )

        options = ['Losses', ''] + list(self.metrics_up_fig) + list(self.metrics_sl_fig)
        select_area = pn.widgets.Select(options=options, disabled_options=[''], size=len(options), stylesheets=[stylesheet], sizing_mode='stretch_width')

        def navigate(selected: str):
            if selected == 'Losses':
                return pn.Column("# Losses", *list(self.losses_fig.values()))
            elif selected in self.metrics_up_fig:
                return pn.Column(f"# {selected}", self.metrics_up_fig[selected])
            elif selected in self.metrics_sl_fig:
                return pn.Column(f"# {selected}", self.metrics_sl_fig[selected])
            else:
                return "# Metric Not Found"
        
        return pn.template.FastListTemplate(
            title="Model Monitoring Dashboard",
            sidebar=select_area,
            sidebar_width=250,
            main=pn.bind(navigate, selected=select_area),
            theme="dark",
            theme_toggle=False,
            header_background="transparent",
            raw_css=["fast-tooltip {display: none};"]
        )

    def start(self, port: int = 8050) -> None:
        self.server = pn.serve(self.get_page, port=port, show=True, threaded=True)

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()
