import numpy as np
from typing import *
from pathlib import Path
import explore_lif
import pandas as pd
from explore_lif.explore_lif import Serie


from bokeh.plotting import figure, Figure
from bokeh.models.glyphs import Image, MultiLine
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool, TapTool, BoxAnnotation, Patches, PolyDrawTool, PolyEditTool
from bokeh.models import Slider, MultiSelect, TextInput, Select, RadioButtonGroup, DataTable, TableColumn  # UI widgets for data selction
from bokeh.events import DoubleTap, Tap
from bokeh.layouts import gridplot, column, row
from bokeh.io import show, output_notebook

from .core import WebPlot, BokehCallbackSignal
from .utils import get_channel_mapping


_default_image_figure_params = dict(
    plot_height=1000,
    plot_width=1000,
    tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
    output_backend='webgl',
    tools='tap,hover,pan,wheel_zoom,box_zoom,reset',
    match_aspect=True,
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
        self.sig_channel_changed = BokehCallbackSignal()
        self.sig_roi_tapped = BokehCallbackSignal()
        self.sig_rois_changed = BokehCallbackSignal()

        self.lif_file_path = lif_file_path
        self.lif_reader = explore_lif.Reader(self.lif_file_path)

        self.image_figure_params = image_figure_params
        self.tooltip_columns = tooltip_columns

        dataframe_columns =\
        [
            'filename',
            'series_name',
            'xs',
            'ys',
            'cell_name'
        ]
        self.dataframe = pd.DataFrame(columns=dataframe_columns)

        if self.image_figure_params is None:
            self.image_figure_params = dict()

        self.image_figure: Figure = figure(
            **{
                **_default_image_figure_params,
                **self.image_figure_params
            }
        )

        self.roi_data_source = ColumnDataSource({'xs': [], 'ys': []})

        self.roi_renderer = self.image_figure.patches('xs', 'ys', source=self.roi_data_source, line_width=3, alpha=0.4)
        self.vertex_renderer = self.image_figure.circle(
            [], [], size=5, color='white'
        )

        self.roi_draw_tool = PolyDrawTool(renderers=[self.roi_renderer])
        self.roi_edit_tool = PolyEditTool(renderers=[self.roi_renderer], vertex_renderer=self.vertex_renderer)

        self.image_figure.add_tools(self.roi_draw_tool, self.roi_edit_tool)
        self.image_figure.toolbar.active_drag = self.roi_edit_tool

        self.roi_renderer.data_source.selected.on_change('indices', self.sig_roi_tapped.trigger)
        self.sig_roi_tapped.connect(self.roi_tapped)

        self.roi_renderer.data_source.on_change('data', self.sig_rois_changed.trigger)
        self.sig_rois_changed.connect(self.set_roi_table)


        # must initialize with some array else it won't work
        empty_img = np.zeros(shape=(100, 100), dtype=np.uint8)

        self.image_glyph: Image = self.image_figure.image(
            image=[empty_img],
            x=0, y=0,
            dw=10, dh=10,
            level="image",
        )

        self.image_figure.grid.grid_line_width = 0

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
        self.channel_ix: int = None

        # channel ui
        # assume laser options are the same for the entire lif file
        self.channel_mapping = get_channel_mapping(
            self.lif_file_path,
            series=self.series_list[0]
        )

        self.radiobutton_channel = RadioButtonGroup(
            labels=list(map(str, self.channel_mapping.values()))
        )
        self.radiobutton_channel.on_change("active", self.sig_channel_changed.trigger)
        self.sig_channel_changed.connect(lambda x: self.set_channel(int(x)))

        self.roi_table_columns = ColumnDataSource({'ix': [], 'tags': []})
        self.roi_tags_table = DataTable(source=self.roi_table_columns)

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

        self.set_channel(self.channel_ix)

    @WebPlot.signal_blocker
    def set_channel(self, channel_ix):
        if self.series_current is None or channel_ix is None:
            return

        self.channel_ix = channel_ix

        self.stack = self.series_current.getFrame(channel=channel_ix)
        max_proj = self.stack.max(axis=0)

        self.image_glyph.glyph.dh = max_proj.shape[0]
        self.image_glyph.glyph.dw = max_proj.shape[1]
        self.image_glyph.data_source.data['image'] = [max_proj]

    def roi_tapped(self, ix):
        print(ix)
        ix = ix[0]

        ds = self.roi_renderer.data_source.data
        coors = [np.column_stack([ds['xs'][i], ds['ys'][i]]) for i in range(len(ds['xs']))]
        print(coors[ix])

    def set_roi_table(self, *args):
        print(args)
        self.roi_tags_table.source.data['ix'] = list(range(10))
        self.roi_tags_table.source.data['tags'] = list(range(10))

    def set_dashboard(self, doc):
        doc.add_root(
            column(
                self.selector_series,
                self.radiobutton_channel,
                self.image_figure,
                self.roi_tags_table
            )
        )
