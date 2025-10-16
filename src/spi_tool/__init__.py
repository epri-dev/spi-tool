import panel as pn
import holoviews as hv
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.no_silent_downcasting = True

mpl.use("agg")
plt.rcParams["figure.constrained_layout.use"] = True

pn.extension(
    "tabulator",
    "perspective",
    "mathjax",
    "modal",
    sizing_mode="stretch_width",
    notifications=True,
    throttled=True,
)
pn.config.layout_compatibility = "error"
hv.extension("bokeh")

pn.pane.Markdown.styles = {"font-size": "16px", "line-height": "1.6"}
pn.widgets.Button.styles = {"font-size": "16px"}

from . import models  # noqa: E402, F401
from . import ui  # noqa: E402, F401
from . import _utils  # noqa: E402, F401
from . import _helper  # noqa: E402, F401
from . import version  # noqa: E402, F401
from . import cli  # noqa: E402, F401
