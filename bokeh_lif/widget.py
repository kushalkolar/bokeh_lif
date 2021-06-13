import numpy as np
from typing import *
from pathlib import Path
import explore_lif
from explore_lif.explore_lif import Serie


from bokeh.plotting import figure, Figure
from bokeh.models.glyphs import Image, MultiLine
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool, TapTool, BoxAnnotation, Patches
from bokeh.models import Slider, MultiSelect, TextInput, Select, RadioButtonGroup  # UI widgets for data selction
from bokeh.layouts import gridplot, column, row
from bokeh.io import show, output_notebook

from .core import WebPlot, BokehCallbackSignal
from .utils import get_channel_mapping


_default_image_figure_params = dict(
    plot_height=500,
    plot_width=500,
    tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
    output_backend='webgl',
    match_aspect=True,
)

_default_curve_figure_params = dict(
    plot_height=250,
    plot_width=1000,
    tools='tap,hover,pan,wheel_zoom,box_zoom,reset',
)


class LifWidget(WebPlot):
    def __init__(
            self,
            lif_file_path: str,
            tooltip_columns: List[str] = None,
            image_figure_params: dict = None,
            curve_figure_params: dict = None
    ):
        WebPlot.__init__(self)

        self.sig_series_changed = BokehCallbackSignal()
        self.sig_channel_left_changed = BokehCallbackSignal()
        self.sig_channel_right_changed = BokehCallbackSignal()

        self.lif_file_path = lif_file_path
        self.lif_reader = explore_lif.Reader(self.lif_file_path)

        self.image_figure_params = image_figure_params
        self.tooltip_columns = tooltip_columns

        if self.image_figure_params is None:
            self.image_figure_params = dict()

        self.image_figures: List[Figure] = [
            figure(
                **{
                    **_default_image_figure_params,
                    **self.image_figure_params
                }
            ) for i in range(2)
        ]

        # must initialize with some array else it won't work
        empty_img = np.zeros(shape=(100, 100), dtype=np.uint8)

        self.image_glyphs: List[Image] = \
            [
                fig.image(
                    image=[empty_img],
                    x=0, y=0,
                    dw=10, dh=10,
                    level="image",
                ) for fig in self.image_figures
            ]

        for fig in self.image_figures:
            fig.grid.grid_line_width = 0

        self.tooltips = None

        if self.tooltip_columns is not None:
            self.tooltips = [(col, f'@{col}') for col in self.tooltip_columns]

        #############################################################################
        # UI Selection stuff

        # series ui
        self.series_list = [s.getName() for s in self.lif_reader.getSeries()]

        self.selector_series = MultiSelect(
            title="Series",
            value=[self.series_list[0]],
            options=self.series_list
        )
        self.selector_series.on_change("value", self.sig_series_changed.trigger)
        self.sig_series_changed.connect(self.set_series)

        self.series_current: Serie = None
        self.channel_left_ix: int = None
        self.channel_right_ix: int = None

        # channel ui
        # assume laser options are the same for the entire lif file
        self.channel_mapping = get_channel_mapping(
            self.lif_file_path,
            series=self.series_list[0]
        )

        self.radiobutton_channel_left = RadioButtonGroup(
            labels=list(map(str, self.channel_mapping.values()))
        )
        self.radiobutton_channel_left.on_change("active", self.sig_channel_left_changed.trigger)
        self.sig_channel_left_changed.connect(lambda x: self.set_channel_left(int(x)))

        self.radiobutton_channel_right = RadioButtonGroup(
            labels=list(map(str, self.channel_mapping.values()))
        )
        self.radiobutton_channel_right.on_change("active", self.sig_channel_right_changed.trigger)
        self.sig_channel_right_changed.connect(lambda x: self.set_channel_right(int(x)))

    @WebPlot.signal_blocker
    def set_series(self, s: str):
        if len(s) != 1:
            return
        s = s[0]
        if self.series_current == s:
            return

        if self.channel_mapping != get_channel_mapping(self.lif_file_path, s):
            raise ValueError(f"Channel mapping is different for Series: {s}")

        series_ix = self.series_list.index(s)

        self.series_current = self.lif_reader.getSeries()[series_ix]

        self.set_channel_left(self.channel_left_ix)
        self.set_channel_right(self.channel_right_ix)

    @WebPlot.signal_blocker
    def set_channel_left(self, channel_ix):
        if self.series_current is None:
            return

        if self.channel_left_ix == channel_ix or channel_ix is None:
            return

        self.channel_left_ix = channel_ix

        self.stack_left = self.series_current.getFrame(channel=channel_ix)
        self.image_glyphs[0].data_source.data['image'] = [self.stack_left.max(axis=0)]

    @WebPlot.signal_blocker
    def set_channel_right(self, channel_ix):
        if self.series_current is None:
            return

        if self.channel_right_ix == channel_ix or channel_ix is None:
            return

        self.channel_right_ix = channel_ix

        self.stack_right = self.series_current.getFrame(channel=channel_ix)
        self.image_glyphs[1].data_source.data['image'] = [self.stack_right.max(axis=0)]

    def set_dashboard(self, doc):
        doc.add_root(
            column(
                self.selector_series,
                row(self.radiobutton_channel_left, self.radiobutton_channel_right),
                row(*self.image_figures),
            )
        )
