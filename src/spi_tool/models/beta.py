import numpy as np
import pandas as pd
import param as pm
import panel as pn
import os
import io
import datetime
import scipy
import matplotlib as mpl

from .. import _utils
from .. import _helper


class TimeseriesScenarioInput(pm.Parameterized):
    filename = pm.Bytes(label="Select a CSV file")
    input_df = pm.DataFrame(precedence=-1)
    default_filename = pm.Path(
        default=os.path.join(_utils.DATA_FOLDER_PATH, "carbon-prices.csv"),
        allow_refs=True,
    )

    label = pm.String(default="Carbon Price", allow_refs=True, precedence=-1)
    unit = pm.String(default="2022 $/MTCO2e", allow_refs=True, precedence=-1)
    error = pm.Boolean(default=False)
    error_message = pm.String(default="")
    has_warning = pm.Boolean(default=False)
    warning_message = pm.String(default="")
    ready = pm.Boolean(False, precedence=-1)

    def get_data(self, filename):
        df = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
        self.unit = list(set(col[1].strip() for col in df.columns))[0]
        df.index.name = "date"
        if df.index.duplicated().any():
            self.has_warning = True
            duplicated_dates = df.index[df.index.duplicated()].unique()
            self.warning_message = f"Duplicated dates found.\nAveraging values for dates: {list(duplicated_dates)}"
        df.columns = [col[0].strip() for col in df.columns]
        df = df.ffill().bfill().resample("1YS").mean().interpolate()
        return df

    def generate_sample_data_csv(self):
        df = pd.read_csv(self.default_filename)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return csv_buffer

    def load_sample_data(self, event=None):
        self.error = False
        self.ready = False
        self.has_warning = False
        filename = self.default_filename
        try:
            self.input_df = self.get_data(filename=filename)
            self.input_df = self.input_df[
                [
                    col
                    for col in self.input_df.columns
                    if "2022 IEPR Preliminary GHG Allowance Price Projections - "
                    not in col
                ]
                + [
                    col
                    for col in self.input_df.columns
                    if "2022 IEPR Preliminary GHG Allowance Price Projections - " in col
                ]
            ]
            self.ready = True
        except Exception as e:
            self.error_message = f"**Error:** {str(e)}"
            self.error = True

    @pm.depends("filename", watch=True)
    def load_data(self, event=None):
        self.error = False
        self.has_warning = False
        self.ready = False
        if self.filename is None:
            filename = self.default_filename
        else:
            filename = io.BytesIO(self.filename)
        try:
            self.input_df = self.get_data(filename=filename)
            self.ready = True
        except Exception as e:
            self.error_message = f"**Error:** {str(e)}"
            self.error = True

    @pm.depends("input_df")
    def plot(self):
        fig = mpl.figure.Figure(figsize=(16, 10), layout="constrained")
        if self.input_df is None:
            return fig
        ax = fig.add_subplot(211)

        self.input_df.plot.line(
            ax=ax,
            label=f"{self.label} Scenarios",
            legend=False,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.label} ({self.unit})")
        ax.set_title(f"{self.label} Scenarios")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True
        )

        return fig

    @pm.output(input_df=pm.DataFrame, label=pm.String, unit=pm.String)
    def output(self):
        if self.input_df is None:
            return pd.DataFrame(
                data={"scenario1": [0.0, 0.0], "scenario2": [0.0, 0.0]},
                index=pd.DatetimeIndex(["2024", "2025"]),
            )
        return self.input_df, self.label, self.unit

    def panel(self):
        def _update_notifications(evt=None):
            if evt.new:
                if pn.state.notifications is not None:
                    pn.state.notifications.position = "top-right"
                    pn.state.notifications.success(
                        'Click the "Next" Button to proceed.'
                    )

        self.param.watch(_update_notifications, "ready")

        file_input = pn.Row(
            pn.widgets.FileInput.from_param(self.param.filename, accept=".csv"),
            pn.widgets.TooltipIcon(
                value=f"Upload a CSV file with only the first column as 'date' and the remaining columns with '{self.unit}' data.",
                sizing_mode=None,
            ),
        )

        button = pn.widgets.Button(
            name="Load Default CSV",
            button_type="primary",
            on_click=self.load_sample_data,
            align=("center", "center"),
            sizing_mode=None,
        )
        sample_button = pn.widgets.FileDownload(
            filename=f"{self.label.lower()}-data.csv",
            callback=self.generate_sample_data_csv,
            icon="download",
            label="Download Default CSV",
            align=("center", "center"),
            sizing_mode=None,
        )

        file_input_row = pn.Row(file_input, button, sample_button)

        def _update_input_df(df):
            if df is None:
                return pn.indicators.LoadingSpinner(
                    value=True, width=50, height=50, margin=10
                )
            df = df.copy()
            df.index = df.index.strftime("%Y-%m-%d")
            # Make header a multi index header
            df.columns = pd.MultiIndex.from_tuples(
                [(col, self.unit) for col in df.columns]
            )
            return pn.widgets.Tabulator(
                df,
                pagination="remote",
                page_size=10,
                frozen_columns=["date"],
                header_tooltips={self.label: self.unit},
                selectable=False,
                disabled=True,
            )

        preview = pn.Column(
            pn.bind(_update_input_df, self.param.input_df),
            pn.Column(
                pn.pane.Matplotlib(
                    self.plot, max_height=800, align=("center", "center")
                ),
            ),
            visible=self.param.ready,
        )

        intro = pn.pane.Markdown(
            rf"""
This module allows resource planners to generate Monte Carlo samples for uncertain future {self.label.lower()}. The samples are based on {self.label.lower()} data provided by the user. Default {self.label.lower()} forecasts are also provided.

### User Requirements:

Two (or more) {self.label.lower()} forecasts. In this version of SPI-Tool, only two {self.label.lower()} forecasts will be used to generate samples. The user may also use default {self.label.lower()} forecasts available in SPI-Tool.

### SPI-Tool Methodology:

The Monte-Carlo {self.label.lower()} samples generated by SPI-Tool will be bounded by user-supplied "lower bound" and "upper bound" inputs. These bounds typically correspond to low and high {self.label.lower()} forecasts, respectively. A user-defined probability distribution determines how samples are distributed within that range.

### Handling Uncertainty:

This module allows the user to generate samples of future {self.label.lower()} that consider inter-annual uncertainty (e.g. long-term policy trends), leveraging the uncertainty information across the {self.label.lower()} forecasts identified by the user.

For more information on how SPI-Tool generates stochastic samples, please refer to the User Guide tab at the top of the screen.

### Instructions:

To get started with generating probabilistic {self.label.lower()} samples, upload {self.label.lower()} forecasts that serve as bounds for the samples generated by SPI-Tool.

You can upload {self.label.lower()} forecasts using the "Browse..." button. Alternatively, click the "Load Default CSV" button to load the sample data or "Download Default CSV" to download the sample data to examine it, edit it and upload it using "Browse..." button.

Once you have loaded the data, click "Next" to proceed.

                                 """,
            max_width=800,
        )

        error = pn.Column(
            pn.pane.Alert(
                object=self.param.error_message,
                alert_type="danger",
                margin=(5, 15, 0, 15),
                visible=self.param.error,
            ),
        )

        warning = pn.Column(
            pn.pane.Alert(
                object=self.param.warning_message,
                alert_type="warning",
                margin=(5, 15, 0, 15),
            ),
            visible=self.param.has_warning,
        )

        info = pn.Card(
            pn.pane.Alert(
                object=f"""
**Note:** The CSV file should have 2 header rows.

The first row should have the following columns:

- `date`: The date column contains the dates for the scenarios
- `scenario_1` ({self.unit}): The values of the first scenario
- `scenario_2` ({self.unit}): The values of the second scenario
- `scenario_3` ({self.unit}): The values of the third scenario
- ...
- `scenario_N` ({self.unit}): The values of the Nth scenario

The second row should contain the units for the carbon price bounds.
All units should be the same and use real (i.e., base year) dollars. For example, the units may be "2022 $/MTCO2e".

The third row onwards should contain the data for the bounding carbon price forecasts.

The data should be sorted by date in ascending order.
The date format should be `YYYY` or `YYYY-MM-DD`.

The values of the scenarios should be numbers.

The second row should contain the units without any units for the date.

Here's an example of a valid CSV file:

```plaintext
date,scenario_1,scenario_2,scenario_3,scenario_4
,2022 $/MTCO2e,2022 $/MTCO2e,2022 $/MTCO2e,2022 $/MTCO2e
2024-01-01,24.39,37.97,30.12,28.45
2025-01-01,40.23,22.18,34.06,38.00
2026-01-01,34.06,38,40.23,22.18
```

The column names for the bounding carbon price forecasts can be any string, but the column names should not contain any special characters. Column names will be used as labels in the next section.

If any dates are missing, the values will be interpolated.
If any hourly or monthly dates are provided, or if the years are duplicated, those values will be averaged.

""",
                alert_type="info",
                margin=(5, 15, 0, 15),
            ),
            title="Info",
            collapsed=pn.bind(lambda x: x, self.param.ready),
        )

        return pn.Card(
            pn.Column(
                intro,
                file_input_row,
                warning,
                error,
                info,
                preview,
                max_width=1200,
            ),
            title="Input Data Preview",
            margin=(5, 8, 10, 8),
            collapsible=False,
            align="center",
            max_width=1200,
        )


class BetaPredictionModel(pm.Parameterized):
    input_df = pm.DataFrame(precedence=-1, allow_refs=True)
    prediction_df = pm.DataFrame(precedence=-1)
    output_df = pm.DataFrame(precedence=-1)
    scaling_factor_df = pm.DataFrame(precedence=-1)

    alpha = pm.Number(default=2)
    beta = pm.Number(default=5)

    scenario_bound_1 = pm.Selector()
    scenario_bound_2 = pm.Selector()

    random_seed = pm.Integer(default=42)
    n_samples = pm.Integer(default=10)
    samples = pm.List([f"sample_{i + 1}" for i in range(0, 10)], precedence=-1)

    label = pm.String(default="Carbon Price", allow_refs=True, precedence=-1)
    unit = pm.String(default="2022 $/MTCO2e", allow_refs=True, precedence=-1)

    ready = pm.Boolean(False, precedence=-1)

    @pm.depends("input_df", watch=True, on_init=True)
    def _update_low_high(self):
        if self.input_df is None:
            return
        self.param.scenario_bound_1.objects = self.input_df.columns
        self.param.scenario_bound_2.objects = self.input_df.columns

        self.scenario_bound_1 = self.input_df.columns[0]
        self.scenario_bound_2 = self.input_df.columns[-1]

    def _update_random_seed(self, evt=None):
        self.random_seed = np.random.randint(0, 2**16)

    @pm.depends("random_seed", "n_samples", "alpha", "beta", watch=True, on_init=True)
    def _update_prediction_df(self):
        self.samples = [f"sample_{i + 1}" for i in range(0, self.n_samples)]

        np.random.seed(self.random_seed)

        random_values = np.random.rand(self.n_samples)

        self.prediction_df = pd.DataFrame(
            data=scipy.stats.beta.ppf(random_values, self.alpha, self.beta),
            index=self.samples,
            columns=["scaling_factor"],
        )
        self.prediction_df.index.name = "sample"

    @pm.depends(
        "input_df",
        "scenario_bound_1",
        "scenario_bound_2",
        "prediction_df",
        watch=True,
        on_init=True,
    )
    def _update_output_df(self):
        if (
            self.input_df is None
            or self.prediction_df is None
            or self.scenario_bound_1 is None
            or self.scenario_bound_2 is None
        ):
            return
        low_df = self.input_df[self.scenario_bound_1]
        high_df = self.input_df[self.scenario_bound_2]

        output_samples = {}
        for sample in self.samples:
            output_samples[sample] = (high_df - low_df) * self.prediction_df.loc[
                sample, "scaling_factor"
            ] + low_df

        self.output_df = pd.DataFrame(output_samples)

    @pm.depends("input_df", "scenario_bound_1", "scenario_bound_2")
    def plot_inputs(self):
        fig = mpl.figure.Figure(figsize=(16, 10), layout="constrained")
        if self.input_df is None:
            return fig

        ax = fig.add_subplot(211)

        ax.fill_between(
            self.input_df.index,
            self.input_df[self.scenario_bound_1],
            self.input_df[self.scenario_bound_2],
            color="gray",
            alpha=0.1,
            label="Sample Set Bounds",
        )

        for col in self.input_df.columns:
            if col == self.scenario_bound_1 or col == self.scenario_bound_2:
                label = col
                alpha = 1
            else:
                label = col
                alpha = 0.1
            ax.plot(self.input_df.index, self.input_df[col], label=label, alpha=alpha)

        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.label} ({self.unit})")
        ax.set_title(f"{self.label} Scenarios")

        ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.2))

        return fig

    @pm.depends("output_df")
    def plot_outputs(self):
        fig = mpl.figure.Figure(figsize=(16, 6), layout="constrained")
        if self.output_df is None:
            return fig

        ax = fig.add_subplot(111)

        for i, col in enumerate(self.output_df.columns):
            if i == 0:
                label = col.replace("_1", "").replace("scenario", "sample")
            else:
                label = None

            ax.plot(
                self.output_df.index,
                self.output_df[col],
                alpha=1 / self.n_samples,
                color="gray",
                label=label,
            )

        for col in self.input_df.columns:
            if col == self.scenario_bound_1 or col == self.scenario_bound_2:
                label = col
                alpha = 1
            else:
                continue
            ax.plot(
                self.input_df.index,
                self.input_df[col],
                linewidth=2,
                label=label,
                alpha=alpha,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.label} ({self.unit})")
        ax.set_title(f"{self.label} Samples")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc="best")

        return fig

    @pm.depends("alpha", "beta", "prediction_df")
    def plot_pdf_cdf(self):
        x = np.linspace(0, 1, 100)
        pdf_values = scipy.stats.beta.pdf(x, self.alpha, self.beta)

        fig = mpl.figure.Figure(figsize=(10, 5), layout="constrained")

        axs = fig.subplot_mosaic([["pdf", "cdf"]])

        ax = axs["pdf"]
        ax.plot(x, pdf_values, label="PDF", color="blue")
        ax.set_xlabel("Scaling Factor")
        ax.set_ylabel("Probability Density Function")
        ax.set_title("Beta Distribution PDF")
        ax.legend(loc="best")

        for k, v in self.prediction_df.to_dict()["scaling_factor"].items():
            ax.axvline(v, color="gray", alpha=0.1, linestyle="--", label=k)

        x = np.linspace(0, 1, 100)
        cdf_values = scipy.stats.beta.cdf(x, self.alpha, self.beta)

        ax = axs["cdf"]
        ax.plot(x, cdf_values, label="CDF", color="green")
        ax.set_xlabel("Scaling Factor")
        ax.set_ylabel("Cumulative Density Function")
        ax.set_title("Beta Distribution CDF")
        ax.legend(loc="best")

        for k, v in self.prediction_df.to_dict()["scaling_factor"].items():
            ax.axvline(
                v,
                color="gray",
                alpha=0.1,
                linestyle="--",
                label=k,
            )

        return fig

    def generate_csv(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        num_samples = len(self.output_df.columns)
        training_info = f"alpha-{self.alpha}_beta-{self.beta}"
        label = self.label.replace(" ", "-")

        # Set filename dynamically
        self.download_button.filename = (
            f"output_{label}_{num_samples}-samples_{training_info}_{timestamp}.csv"
        )

        csv_buffer = io.StringIO()
        self.output_df.to_csv(csv_buffer)
        csv_buffer.seek(0)
        return csv_buffer

    def _update_df(self, df, units=False, parse_dates=True):
        if df is None:
            return pn.indicators.LoadingSpinner(
                value=True, width=50, height=50, margin=10
            )
        df = df.copy()
        if parse_dates:
            df.index = df.index.strftime("%Y-%m-%d")
        if units:
            # Make header a multi index header
            df.columns = pd.MultiIndex.from_tuples(
                [(col, self.unit) for col in df.columns]
            )
        return pn.widgets.Tabulator(
            df,
            pagination="remote",
            page_size=10,
            sizing_mode="stretch_width",
            frozen_columns=["date"],
            header_tooltips={self.label: self.unit},
            selectable=False,
            disabled=True,
        )

    def panel(self):
        alpha_slider = pn.Row(
            pn.widgets.TooltipIcon(
                value="Select alpha value.",
                sizing_mode=None,
            ),
            pn.widgets.FloatInput.from_param(
                self.param.alpha, start=1, step=0.5, sizing_mode=None
            ),
        )
        beta_slider = pn.Row(
            pn.widgets.TooltipIcon(
                value="Select beta value.",
                sizing_mode=None,
            ),
            pn.widgets.FloatInput.from_param(
                self.param.beta, start=1, step=0.5, sizing_mode=None
            ),
        )
        scenario_bound_1_select = pn.Row(
            pn.widgets.TooltipIcon(
                value="Select first scenario that will serve as the lower bound.",
                sizing_mode=None,
            ),
            pn.widgets.Select.from_param(
                self.param.scenario_bound_1,
                name="Lower Bound Scenario",
                sizing_mode=None,
            ),
        )
        scenario_bound_2_select = pn.Row(
            pn.widgets.TooltipIcon(
                value="Select second scenario that will serve as the upper bound.",
                sizing_mode=None,
            ),
            pn.widgets.Select.from_param(
                self.param.scenario_bound_2,
                name="Upper Bound Scenario",
                sizing_mode=None,
            ),
        )
        random_seed_input = pn.Row(
            pn.widgets.TooltipIcon(
                value="Random seed for reproducibility.",
                sizing_mode=None,
            ),
            pn.widgets.IntInput.from_param(
                self.param.random_seed,
                sizing_mode=None,
            ),
            pn.Column(
                pn.widgets.Button(
                    icon="arrows-shuffle",
                    description="New random seed",
                    on_click=self._update_random_seed,
                    sizing_mode=None,
                ),
                align="end",
            ),
            sizing_mode=None,
            align=("center", "center"),
        )

        n_samples_input = pn.Row(
            pn.widgets.TooltipIcon(
                value="Number of samples to generate.",
                sizing_mode=None,
            ),
            pn.widgets.IntInput.from_param(
                self.param.n_samples,
                name="Number of Samples",
                start=1,
                sizing_mode=None,
            ),
        )

        # Markdown pane for Beta distribution equation
        beta_equation = pn.pane.Markdown(
            r"""

The Beta distribution is defined by the probability density function (PDF). The shape of the distribution is controlled by the parameters $$\alpha$$ and $$\beta$$. The PDF of the Beta distribution is given by:

$$f(x; \alpha, \beta) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)} \quad \text{for } 0 \leq x \leq 1$$

Carbon price samples are constrained by the user-defined lower and upper bounds. A Beta distribution (as shown above) determines how values are distributed within that range. A scaling factor ( $$f(x; \alpha, \beta) \in [0,1]$$ ) is drawn from the Beta distribution and used to interpolate between the user-provided lower and upper bounds, generating the carbon price sample.

The equation for generating carbon price samples is given by:

$$\text{Sample}_x = \text{scenario1} + f(x; \alpha, \beta) \times (\text{scenario2} - \text{scenario1})$$

For example, a scaling factor of 0.25 places the carbon price at 25% of the distance from the lower bound to the upper bound for every forecast year.
            """,
            max_width=800,
        )

        beta_model = pn.Card(
            pn.Column(
                beta_equation,
                pn.Row(
                    pn.Column(
                        alpha_slider,
                        beta_slider,
                        scenario_bound_1_select,
                        scenario_bound_2_select,
                        n_samples_input,
                        random_seed_input,
                    ),
                    pn.Column(
                        pn.Row(
                            pn.pane.Matplotlib(
                                self.plot_pdf_cdf,
                                max_height=400,
                            ),
                        ),
                    ),
                ),
                pn.pane.Markdown(
                    "The following is a visualization of the bounds provided by the user; followed by the samples generated by SPI-Tool within those bounds."
                ),
                pn.Column(
                    pn.pane.Matplotlib(
                        self.plot_inputs,
                        max_height=600,
                    ),
                    pn.pane.Matplotlib(
                        self.plot_outputs,
                        max_height=400,
                    ),
                ),
            ),
            title="Sample Generation Settings",
            collapsed=False,
            loading=pn.state.param.busy,
        )

        input_df_view = pn.Card(
            pn.pane.Markdown(
                "The following table contains the input data provided by the user which can be used as bounds for the Beta distribution."
            ),
            pn.bind(self._update_df, self.param.input_df),
            title=f"Input {self.label} Data",
            collapsed=True,
            collapsible=True,
        )

        self.download_button = pn.widgets.FileDownload(
            filename="data.csv",
            callback=self.generate_csv,
            button_type="primary",
            icon="download",
            label=f"Download {self.label} Samples",
            align=("center", "center"),
            sizing_mode=None,
            width=400,
        )

        output_df_view = pn.Card(
            pn.pane.Markdown(
                "The following table contains the output samples data generated SPI-Tool using the following scaling factor values."
            ),
            pn.bind(
                lambda df: pn.widgets.Tabulator(
                    df.T.rename_axis("samples"),
                    pagination="remote",
                    page_size=10,
                    sizing_mode="stretch_width",
                    selectable=False,
                    disabled=True,
                ),
                self.param.prediction_df,
            ),
            pn.bind(self._update_df, self.param.output_df, units=True),
            pn.Row(
                self.download_button,
                align="center",
                width=250,
                margin=(5, 8, 10, 8),
            ),
            title="Output Data",
            collapsed=False,
            collapsible=True,
        )

        layout = pn.Column(beta_model, input_df_view, output_df_view)
        return layout


class BetaPipeline(pn.viewable.Viewer):
    label = pm.String(default="Carbon Price", allow_refs=True, precedence=-1)
    unit = pm.String(default="2022 $/MTCO2e", allow_refs=True, precedence=-1)
    default_filename = pm.Path(
        default=os.path.join(_utils.DATA_FOLDER_PATH, "carbon-prices.csv")
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipeline = _helper.CustomPipeline(debug=True)

        self.input_data = TimeseriesScenarioInput(
            label=self.param.label,
            unit=self.param.unit,
            default_filename=self.param.default_filename,
        )
        self.prediction_model = BetaPredictionModel(
            input_df=self.input_data.param.input_df,
            label=self.param.label,
            unit=self.param.unit,
        )

        self.pipeline.add_stage(
            f"Step 1: Input {self.label}",
            self.input_data,
            ready_parameter="ready",
            auto_advance=False,
        )

        self.pipeline.add_stage(
            f"Step 2: Generate {self.label} Samples",
            self.prediction_model,
            ready_parameter="ready",
            auto_advance=False,
        )

        self.layout = pn.Column(
            self.pipeline,
        )

    def __panel__(self):
        return self.layout


class BetaManual(pn.viewable.Viewer):
    scenario = pm.Selector(
        default="sample_1", objects=[f"sample_{i}" for i in range(1, 6)]
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.input_data = TimeseriesScenarioInput(
            default_filename=os.path.join(
                _utils.DATA_FOLDER_PATH, "simple-carbon-prices.csv"
            )
        )
        self.input_data.load_sample_data()
        self.output_data = BetaPredictionModel(
            input_df=self.input_data.input_df,
            n_samples=5,
        )

    def __panel__(self):
        alpha_slider = pn.Row(
            pn.widgets.TooltipIcon(
                value="Select alpha value.",
                sizing_mode=None,
            ),
            pn.widgets.FloatInput.from_param(
                self.output_data.param.alpha, start=1, step=0.5, sizing_mode=None
            ),
        )
        beta_slider = pn.Row(
            pn.widgets.TooltipIcon(
                value="Select beta value.",
                sizing_mode=None,
            ),
            pn.widgets.FloatInput.from_param(
                self.output_data.param.beta, start=1, step=0.5, sizing_mode=None
            ),
        )
        scenario_bound_1_select = pn.Row(
            pn.widgets.TooltipIcon(
                value="Select first scenario.",
                sizing_mode=None,
            ),
            pn.widgets.Select.from_param(
                self.output_data.param.scenario_bound_1, sizing_mode=None
            ),
        )
        scenario_bound_2_select = pn.Row(
            pn.widgets.TooltipIcon(
                value="Select second scenario.",
                sizing_mode=None,
            ),
            pn.widgets.Select.from_param(
                self.output_data.param.scenario_bound_2, sizing_mode=None
            ),
        )

        n_samples_input = pn.Row(
            pn.widgets.TooltipIcon(
                value="Number of samples to generate.",
                sizing_mode=None,
            ),
            pn.widgets.IntInput.from_param(
                self.output_data.param.n_samples,
                name="Number of Samples",
                start=1,
                sizing_mode=None,
            ),
        )

        return pn.Column(
            pn.pane.Markdown(
                r"""
# Carbon Price Forecast

The Monte-Carlo carbon price samples generated by SPI-Tool will be bounded by user-supplied "lower bound" and "upper bound" inputs. These bounds typically correspond to low and high carbon price forecasts, respectively.

Let us assume you have uploaded the following CSV file with carbon price forecasts:
""",
                max_width=800,
            ),
            pn.Card(
                self.input_data.input_df,
                pn.pane.Matplotlib(
                    self.input_data.plot,
                    height=600,
                ),
                title="Example Inputs",
            ),
            pn.pane.Markdown(
                r"""
The user can now choose which scenarios to use as the lower and upper bounds for the Beta distribution.
""",
                max_width=800,
            ),
            pn.Card(
                "Change the following to choose a different lower or upper bound:",
                pn.Row(
                    self.output_data.param.scenario_bound_1,
                    self.output_data.param.scenario_bound_2,
                ),
                pn.pane.Matplotlib(self.output_data.plot_inputs, height=600),
                title="Example Bounds",
            ),
            pn.pane.Markdown(
                r"""
            These bounds will be used to generate carbon price samples using the Beta distribution. In the section below, you'll learn more about how the lower and upper bounds are used in a Beta distribution.

            ## Beta Distribution

            The Beta distribution is defined by the probability density function (PDF).
            The shape of the distribution is controlled by the parameters $$\alpha$$ and $$\beta$$.
            The PDF of the Beta distribution is given by:

            $$f(x; \alpha, \beta) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)} \quad \text{for } 0 \leq x \leq 1$$

            where:
            - $$\alpha > 0$$ and $$\beta > 0$$ are shape parameters.
            - $$B(\alpha, \beta)$$ is the Beta function

            The parameters $$\alpha$$ and $$\beta$$ control the shape of the distribution.
            """,
                max_width=800,
            ),
            pn.Card(
                "Change alpha, beta or number of samples to experiment:",
                pn.Row(
                    pn.Column(
                        alpha_slider,
                        beta_slider,
                        n_samples_input,
                    ),
                    pn.Column(
                        pn.Row(
                            pn.pane.Matplotlib(
                                self.output_data.plot_pdf_cdf,
                                max_height=400,
                            ),
                        ),
                    ),
                ),
                title="Alpha and Beta Sample Parameters",
            ),
            pn.pane.Markdown(
                r"""
            The Beta distribution can be used to generate scenarios for a carbon price forecast by scaling the distribution to a range defined by two input timeseries scenarios.

            The equation for generating carbon price samples is given by:

            $$\text{Sample}_x = \text{scenario1} + f(x; \alpha, \beta) \times (\text{scenario2} - \text{scenario1})$$

            This approach ensures that the scenarios are constrained to the desired range.
                            """,
                max_width=800,
            ),
            pn.Card(
                pn.bind(self.output_data._update_df, self.output_data.param.output_df),
                pn.pane.Matplotlib(
                    self.output_data.plot_outputs,
                    max_height=400,
                ),
                title="Output Samples",
            ),
            pn.pane.Markdown(
                r"""
            This approach ensures that the scenarios are constrained to the desired range, while the shape of the distribution reflects the uncertainty in the scenarios.
            ## Conclusion
            In this manual, you learned how to generate carbon price samples using the Beta distribution. You can experiment with different values of $$\alpha$$, $$\beta$$, and the number of samples to generate different scenarios.
                        """,
                max_width=800,
            ),
        )
