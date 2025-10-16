import panel as pn
import os

try:
    with open(os.path.join(os.path.dirname(__file__), "../../../CHANGELOG.md")) as f:
        CHANGELOG = f.read()
except Exception as _:
    CHANGELOG = "Unable to load changelog"


class ChangeLog(pn.viewable.Viewer):
    def __panel__(self):
        return pn.pane.Markdown(CHANGELOG)
