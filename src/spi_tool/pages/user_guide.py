import panel as pn
import os
import subprocess
import shlex
import shutil

from .. import _utils


manual_path = os.path.join(_utils.get_resources_path(), "SPI-Tool-Software-Manual.pdf")


class UserGuide(pn.viewable.Viewer):
    def __panel__(self):
        if os.path.exists(manual_path):
            return pn.pane.PDF(manual_path, width=700, height=1000)
        else:
            return pn.pane.Markdown(
                "⚠️ **User Guide not available.**\n\n"
                "The file `SPI-Tool-Software-Manual.pdf` could not be found"
            )
