import param
import panel as pn
import os

from .. import models
from .. import _utils

from . import user_guide
from . import faq
from . import changelog
from . import about


class HomePage(pn.viewable.Viewer):
    template = param.ClassSelector(class_=(pn.template.BaseTemplate))

    def __init__(self, **params):
        super().__init__(**params)
        regression_manual = models.regression.RegressionManual()
        beta_manual = models.beta.BetaManual()
        self.pipeline_dict = {
            "Load": {
                "pipeline": models.regression.RegressionPipeline(),
                "user_guide": regression_manual,
                "icon_svg": os.path.join(_utils.IMAGE_FOLDER_PATH, "load-data.svg"),
                "icon_name": "home-bolt",  # https://tabler.io/icons
                "page": None,
            },
            "Gas Price": {
                "pipeline": models.regression.RegressionPipeline(
                    label="Gas Price",
                    unit="$/MMBtu",
                    regression_kind="lognormal",
                    use_day_type=False,
                    default_filename=os.path.join(
                        _utils.DATA_FOLDER_PATH,
                        "Henry-Hub-Natural-Gas-Spot-Price.csv",
                    ),
                ),
                "user_guide": regression_manual,
                "icon_svg": os.path.join(_utils.IMAGE_FOLDER_PATH, "gas-station.svg"),
                "icon_name": "gas-station",  # https://tabler.io/icons
                "page": None,
            },
            "Carbon Price": {
                "pipeline": models.beta.BetaPipeline(),
                "user_guide": beta_manual,
                "icon_svg": os.path.join(_utils.IMAGE_FOLDER_PATH, "flame.svg"),
                "icon_name": "flame",  # https://tabler.io/icons
                "page": None,
            },
        }

        self.footer = pn.pane.Markdown(
            """
### Questions

Please contact [Dheepak Krishnamurthy](mailto:dkrishnamurthy@epri.com), [Rachel Moglen](mailto:rmoglen@epri.com), or [Nidhi Santen](mailto:nsanten@epri.com) for further questions.


""",
            max_width=800,
        )

        self.main_page = self._build_main_menu()
        self.about_page = self._build_about()
        self.user_guide_page = self._build_user_guide()
        self.changelog_page = self._build_changelog()
        self.faq_page = self._build_faq()

    def _build_about(self):
        return pn.Column(
            pn.pane.Markdown(
                """
            # About
            """
            ),
            about.About(),
        )

    def _build_user_guide(self):
        return pn.Column(
            pn.pane.Markdown(
                """
            # User Guide

            This is the manual page.
            """
            ),
            user_guide.UserGuide(),
        )

    def _build_changelog(self):
        return pn.Column(
            changelog.ChangeLog(),
        )

    def _build_faq(self):
        return faq.FAQ()

    def _build_main_menu(self):
        card_list = []
        for name, pipeline_schema in self.pipeline_dict.items():
            btn = pn.widgets.Button(
                name=name,
                sizing_mode="stretch_width",
                align="end",
                button_type="primary",
                on_click=self.start_pipeline,
            )
            card = pn.Card(
                pn.pane.SVG(
                    pipeline_schema["icon_svg"],
                    align=("center"),
                    sizing_mode="stretch_width",
                    height=100,
                ),
                btn,
                title=name,
                collapsible=False,
                width=200,
                height=200,
            )
            card_list.append(card)
        model_selection = pn.GridBox(
            *card_list,
            ncols=3,
            nrows=2,
            sizing_mode=None,
        )
        pane = pn.Column(
            pn.Row(
                pn.pane.SVG(
                    os.path.join(_utils.IMAGE_FOLDER_PATH, "SPI-Tool-no-text.svg"),
                    height=100,
                    width=100,
                ),
                pn.pane.Markdown(
                    r"""
# Welcome to the SPI-Tool Dashboard

"""
                ),
                align=("center", "center"),
                sizing_mode="stretch_width",
            ),
            pn.pane.Markdown(
                r"""

SPI-Tool stands for **Stochastic Planning Inputs - Tool**.

SPI-Tool helps electric system resource planners analyze and characterize uncertainty in their resource planning inputs
by streamlining the initial steps of risk assessment and mitigation through stochastic planning.

This version of SPI-Tool helps users with one of the following for each of the inputs:

- Fit stochastic process parameters
- Generate future samples of these uncertain inputs

SPI-Tool currently considers two types of uncertainty:

- **Intra**-annual uncertainty: uses natural variation (e.g., day-to-day volatility) in historical data to generate futures that exhibit these behaviors.
- **Inter**-annual uncertainty: uses information provided by the resource planner about long-term uncertainty in the system under study.

SPI-Tool can generate sample distribution parameters and probabilistic samples for any of the following types of uncertainties.

Select one of the options below to begin:
            """,
                max_width=800,
            ),
            model_selection,
            self.footer,
        )
        return pane

    def _update_sidebar_button_state(self, event=None):
        if event is None or event.obj.name == "Home":
            self.home_button.button_type = "primary"
        else:
            self.home_button.button_type = "default"
        if event is not None and event.obj.name == "Changelog":
            self.changelog_button.button_type = "primary"
        else:
            self.changelog_button.button_type = "default"
        if event is not None and event.obj.name == "User Guide":
            self.user_guide_button.button_type = "primary"
        else:
            self.user_guide_button.button_type = "default"
        if event is not None and event.obj.name == "About":
            self.about_button.button_type = "primary"
        else:
            self.about_button.button_type = "default"
        if event is not None and event.obj.name == "FAQ":
            self.faq_button.button_type = "primary"
        else:
            self.faq_button.button_type = "default"
        for name, pipeline_schema in self.pipeline_dict.items():
            if event is not None and event.obj.name == name:
                pipeline_schema["button"].button_type = "primary"
            else:
                pipeline_schema["button"].button_type = "default"

    def get_sidebar(self):
        self.home_button = pn.widgets.Button(
            name="Home",
            button_type="primary",
            icon="home",
            sizing_mode="stretch_width",
            on_click=self.get_main_menu,
        )
        self.user_guide_button = pn.widgets.Button(
            name="User Guide",
            button_type="default",
            icon="help",
            sizing_mode="stretch_width",
            on_click=self.start_user_guide,
        )
        self.changelog_button = pn.widgets.Button(
            name="Changelog",
            button_type="default",
            icon="notes",
            sizing_mode="stretch_width",
            on_click=self.start_change_log,
        )
        self.faq_button = pn.widgets.Button(
            name="FAQ",
            button_type="default",
            icon="info-square-rounded",
            sizing_mode="stretch_width",
            on_click=self.start_faq,
        )
        self.about_button = pn.widgets.Button(
            name="About",
            button_type="default",
            icon="info-circle",
            sizing_mode="stretch_width",
            on_click=self.start_about,
        )
        feed = pn.Column(sizing_mode="stretch_height")
        column = pn.Column(
            pn.Row(self.home_button),
            pn.layout.Divider(
                stylesheets=[
                    _utils.divider_stylesheet,
                ]
            ),
            pn.pane.Markdown("**Stochastic Inputs**"),
            feed,
            pn.layout.HSpacer(height=20),
            pn.layout.Divider(
                stylesheets=[
                    _utils.divider_stylesheet,
                ]
            ),
            pn.layout.VSpacer(),
            pn.Column(
                pn.Row(self.user_guide_button),
                pn.Row(self.faq_button),
                pn.Row(self.about_button),
                # pn.Row(self.changelog_button),
                pn.widgets.StaticText(
                    name="version",
                    value=_utils.read_version_file(),
                    align=("center", "center"),
                ),
            ),
            sizing_mode="stretch_height",
        )
        for name, pipeline_schema in self.pipeline_dict.items():
            sidebar_btn = pn.widgets.Button(
                name=name,
                button_type="default",
                icon=pipeline_schema["icon_name"],
                sizing_mode="stretch_width",
                on_click=self.start_pipeline,
            )
            self.pipeline_dict[name]["button"] = sidebar_btn
            feed.append(sidebar_btn)

        return column

    def get_main_menu(self, event=None):
        self._update_sidebar_button_state(event)
        self.main_page = self._build_main_menu()
        self._update_template(self.main_page)

    def start_pipeline(self, event):
        self._update_sidebar_button_state(event)
        pipeline_name = event.obj.name
        if pipeline_name is None or pipeline_name not in self.pipeline_dict.keys():
            pn.state.notifications.error("No Pipeline found for this Button")
            return
        if self.pipeline_dict[pipeline_name]["page"] is not None:
            pane = self.pipeline_dict[pipeline_name]["page"]
            self._update_template(pane)
        else:
            p = self.pipeline_dict[pipeline_name]["pipeline"]
            pane = pn.Tabs(
                (
                    "Model",
                    pn.Column(
                        pn.Row(
                            p.pipeline.title,
                            pn.layout.HSpacer(),
                            p.pipeline.prev_button,
                            p.pipeline.next_button,
                        ),
                        p.pipeline.stage,
                        max_width=1200,
                    ),
                ),
                (
                    "User Guide",
                    self.pipeline_dict[pipeline_name]["user_guide"],
                ),
                dynamic=True,
            )
            self.pipeline_dict[pipeline_name]["page"] = pane
            self._update_template(pane)

    def start_user_guide(self, event=None):
        self._update_sidebar_button_state(event)
        self._update_template(self.user_guide_page)

    def start_about(self, event=None):
        self._update_sidebar_button_state(event)
        self._update_template(self.about_page)

    def start_change_log(self, event=None):
        self._update_sidebar_button_state(event)
        self._update_template(self.changelog_page)

    def start_faq(self, event=None):
        self._update_sidebar_button_state(event)
        self._update_template(self.faq_page)

    def _update_template(self, page):
        self.template.main[0].objects = [page]
