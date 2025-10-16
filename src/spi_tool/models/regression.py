import numpy as np
import pandas as pd
import param
import scipy
import panel as pn
import calendar
import os
import io
import datetime
import matplotlib as mpl
import math
import warnings
import statsmodels.api as sm

from .. import _utils
from .. import _helper


def linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    y_predictions = intercept + slope * x

    steyx = np.sqrt(np.sum((y - y_predictions) ** 2) / (len(y) - 2))
    mean_reversion = -float(slope)
    long_run_mean = float(intercept / -slope)
    volatility = abs(float((steyx / long_run_mean) * 100))

    return dict(
        slope=slope,
        intercept=intercept,
        p_value=p_value,
        r_value=r_value,
        steyx=steyx,
        mean_reversion=mean_reversion,
        long_run_mean=long_run_mean,
        volatility=volatility,
    )


class TimeseriesInput(param.Parameterized):
    default_filename = param.Path(
        default=os.path.join(_utils.DATA_FOLDER_PATH, "miso-daily-demand.csv"),
        allow_refs=True,
    )
    input_df = param.DataFrame(precedence=-1)
    filename = param.Bytes(label="Select a CSV file")
    label = param.String(default="Load", allow_refs=True)
    unit = param.String(default="MW", allow_refs=True)
    ready = param.Boolean(default=False)
    error = param.Boolean(default=False)
    error_message = param.String(default="")
    has_warning = param.Boolean(default=False)
    warning_message = param.String(default="")
    regression_kind = param.Selector(
        default="normal", objects=["normal", "lognormal"], allow_refs=True
    )
    use_day_type = param.Boolean(default=False, allow_refs=True)
    use_month = param.Boolean(default=False, allow_refs=True)

    def get_data(self, filename):
        try:
            input_df = pd.read_csv(filename, header=[0, 1], parse_dates=True)
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
        unit = list(set(col[1].strip() for col in input_df.columns[1:]))[-1]
        input_df.columns = [col[0].strip() for col in input_df.columns]
        input_df = self._validate_data(input_df)
        return input_df, unit

    def _validate_data(self, input_df):
        required_columns = ["date", self.label.lower()]
        if not all(col in input_df.columns for col in required_columns):
            raise ValueError(
                f"Data is missing required columns:\n\n\n{','.join(required_columns)}\n\n\nInstead found the following:\n\n\n{','.join(input_df.columns)}"
            )

        if input_df.index.duplicated().any():
            self.has_warning = True
            duplicated_dates = input_df.index[input_df.index.duplicated()].unique()
            self.warning_message = f"Duplicated dates found.\nAveraging values for dates: {list(duplicated_dates)}"

        with warnings.catch_warnings(record=True) as w:
            pd.to_datetime(input_df["date"])
            for warning in w:
                self.has_warning = True
                self.warning_message = (
                    self.warning_message
                    + f"Warning when parsing date column: {str(warning.message)}"
                )

        try:
            input_df = (
                input_df.assign(date=lambda df: pd.to_datetime(df["date"]))
                .sort_values("date")
                .set_index("date")
                .dropna()
                .resample("1D")
                .mean()
                .interpolate()
            )
        except Exception as e:
            raise ValueError(f"Error parsing date column: {str(e)}")
        return input_df

    def load_data(self, filename):
        self.error = False
        self.has_warning = False
        self.ready = False
        try:
            df, unit = self.get_data(filename=filename)
            self.unit = unit
            self.input_df = df
            self.ready = True
        except Exception as e:
            self.error_message = f"**Error:** {str(e)}"
            self.error = True

    def load_example_data(self, event=None):
        self.load_data(filename=self.default_filename)

    @param.depends("filename", watch=True)
    def _update_input_df_from_filename(self, event=None):
        filename = io.BytesIO(self.filename)
        self.load_data(filename)

    @param.depends("input_df", "label", "unit")
    def plot(self):
        # Create the plot
        fig = mpl.figure.Figure(figsize=(16, 6))

        if self.input_df is None:
            return fig

        axs = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]}, sharey=True)

        ax = axs[0]

        self.input_df.plot.line(
            y=self.label.lower(), ax=ax, color="blue", linewidth=2, label=self.label
        )

        # Set the title and labels
        ax.set_title("Input Historical Data")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.label} ({self.unit})")

        # Format the y-axis
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.f"))

        # Show legend
        ax.legend(loc="upper right")

        ax = axs[1]

        self.input_df[self.label.lower()].plot.hist(
            ax=ax,
            bins=30,
            color="blue",
            alpha=0.25,
            label="Input Historical Data",
            orientation="horizontal",
        )
        ax.set_title("Histogram")
        ax.set_ylabel(f"{self.label} ({self.unit})")

        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.f"))

        ax.legend(loc="upper right")

        return fig

    @param.depends("input_df", "label", "unit")
    def plot_seasonal_decomposed_components(self):
        # Create the plot
        fig = mpl.figure.Figure(figsize=(16, 6 * 2))

        if self.input_df is None:
            return fig

        axs = fig.subplots(3, 1, sharex=True)

        decompose = sm.tsa.seasonal_decompose(
            self.input_df[self.label.lower()], model="additive"
        )

        axs[0].set_title("Trend Component")
        decompose.trend.plot(ax=axs[0])
        axs[0].set_ylabel(f"{self.label} ({self.unit})")

        axs[1].set_title("Seasonal Component")
        decompose.seasonal.plot(ax=axs[1])
        axs[1].set_ylabel(f"{self.label} ({self.unit})")

        axs[2].set_title("Residual Component")
        decompose.resid.plot(ax=axs[2])
        axs[2].set_ylabel(f"{self.label} ({self.unit})")

        return fig

    @param.depends("input_df", "label", "unit")
    def plot_acf_pacf(self):
        # Create the plot
        fig = mpl.figure.Figure(figsize=(16, 6))

        if self.input_df is None:
            return fig

        data = self.input_df[self.label.lower()]

        axs = fig.subplots(1, 2, sharey=True, sharex=True)

        ax = axs[0]

        sm.graphics.tsa.plot_acf(data, ax=ax, lags=50)
        ax.set_title("Autocorrelation")

        ax = axs[1]

        sm.graphics.tsa.plot_pacf(data, ax=ax, lags=50)
        ax.set_title("Partial Autocorrelation")

        return fig

    @param.depends("input_df", "label", "unit")
    def plot_day_of_week_and_month_averages(self):
        # Create the plot
        fig = mpl.figure.Figure(figsize=(16, 6))

        if self.input_df is None:
            return fig

        df = self.input_df.copy()

        df["month"] = df.index.month
        avg_per_month = df.groupby("month").mean()

        df["day_of_week"] = df.index.dayofweek
        avg_per_day = df.groupby("day_of_week").mean()

        axs = fig.subplots(1, 2, sharey=True)

        ax = axs[0]

        avg_per_day.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        avg_per_day.plot(
            y=self.label.lower(),
            kind="bar",
            color="skyblue",
            edgecolor="black",
            ax=ax,
            legend=False,
        )

        ax.set_title(f"Average {self.label} Per Day of the Week")
        ax.set_ylabel(f"Average {self.label} ({self.unit})")
        ax.set_xlabel("Day of the Week")
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0)

        ax = axs[1]

        avg_per_month.index = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        avg_per_month.plot(
            y=self.label.lower(),
            kind="bar",
            color="salmon",
            edgecolor="black",
            ax=ax,
            legend=False,
        )
        ax.set_title(f"Average {self.label} Per Month of the Year")
        ax.set_xlabel("Month")
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0)

        return fig

    @param.depends("input_df", "label", "unit")
    def _update_adfuller(self):
        if self.input_df is None:
            return
        result = sm.tsa.adfuller(self.input_df[self.label.lower()], autolag="AIC")
        output_df = pd.DataFrame(
            {
                "Values": [
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                    result[4]["1%"],
                    result[4]["5%"],
                    result[4]["10%"],
                ],
                "Metric": [
                    "Test Statistics",
                    "p-value",
                    "No. of lags used",
                    "Number of observations used",
                    "critical value (1%)",
                    "critical value (5%)",
                    "critical value (10%)",
                ],
            }
        ).set_index("Metric")
        return output_df

    @param.output(input_df=param.DataFrame)
    def output(self):
        if self.input_df is None:
            return None
        return self.input_df

    def generate_sample_data_csv(self):
        df = pd.read_csv(self.default_filename)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return csv_buffer

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
                value=f"Upload a CSV file with only two columns: 'date' and '{self.label.lower()}'.",
                sizing_mode=None,
            ),
        )

        def _update_input_df(df):
            if df is None:
                return pn.indicators.LoadingSpinner(
                    value=True, width=50, height=50, margin=10
                )
            df = df.copy()
            df.index = df.index.strftime("%Y-%m-%d")
            # Make header a multi index header
            df.columns = pd.MultiIndex.from_tuples(
                [(f"{self.label.lower()}", self.unit) for col in df.columns]
            )
            return pn.widgets.Tabulator(
                df,
                pagination="remote",
                page_size=10,
                max_width=400,
                sizing_mode=None,
                frozen_columns=["date"],
                header_tooltips={self.label: self.unit},
                selectable=False,
                disabled=True,
            )

        preview = pn.Column(
            pn.Card(
                pn.Row(
                    pn.bind(_update_input_df, self.param.input_df),
                    pn.pane.Matplotlib(self.plot, height=400),
                ),
                title="Input Data",
            ),
            pn.Card(
                pn.Row(
                    pn.bind(
                        lambda df: pn.widgets.Tabulator(
                            self._update_adfuller(),
                            max_width=400,
                            sizing_mode=None,
                            selectable=False,
                            disabled=True,
                        ),
                        self.param.input_df,
                    ),
                    pn.pane.Matplotlib(self.plot_acf_pacf, height=400),
                ),
                collapsed=True,
                title="Augmented Dickey-Fuller Tests",
            ),
            pn.Card(
                pn.pane.Matplotlib(
                    self.plot_day_of_week_and_month_averages, height=400
                ),
                pn.pane.Matplotlib(
                    self.plot_seasonal_decomposed_components, height=400 * 2
                ),
                collapsed=True,
                title="Seasonality Plots",
            ),
            visible=self.param.ready,
        )

        button = pn.widgets.Button(
            name="Load Example CSV",
            button_type="primary",
            on_click=self.load_example_data,
            align=("center", "center"),
            sizing_mode=None,
        )
        sample_button = pn.widgets.FileDownload(
            filename=f"{self.label.lower().replace(' ', '-')}-data.csv",
            callback=self.generate_sample_data_csv,
            icon="download",
            label="Download Example CSV",
            align=("center", "center"),
            sizing_mode=None,
        )

        intro = pn.pane.Markdown(
            f"""
      This module allows resource planners to model patterns observed in historical {self.label.lower()} data by generating Monte-Carlo samples.

      The process parameters and samples are derived from the historical {self.label.lower()} data you provide.
      SPI-Tool generates samples that reflect a range of plausible {self.label.lower()} futures based on the natural variability detected in the input data. Note, SPI-Tool is not a {self.label.lower()} forecasting tool.

      ### User Requirements:

      - Historical {self.label.lower()} data without any underlying long-term trends

      ### SPI-Tool Methodology:

      Auto-regressive models are a common approach for generating samples of future time series in IRP. SPI-Tool fits one of the most-used forms of auto-regressive models --- the AR(1) model --- to historical {self.label.lower()} data. For more information on AR(1) models and auto-regressive models in general, please refer to the FAQ section.

      For examples of IRPs that use auto-regressive models for stochastic analyses, refer to:

      - PacifiCorp. “2023 Integrated Resource Plan,” March 31, 2023.
      - AES Indiana. “2022 Integrated Resource Plan,” December 1, 2022.
      - CenterPoint Energy. “2022/2023 Integrated Resource Plan,” 2023.
      - Idaho Power. “2023 Integrated Resource Plan,” September, 2023.
      - Tennessee Valley Authority. “2019 Integrated Resource Plan,” 2019.

      ### Handling Uncertainty:

      This module allows the user to generate samples of future {self.label.lower()} that consider uncertainty inherent in day-to-day fluctuations in load levels. SPI-Tool exploits this natural variation in historical data to generate futures that exhibit these behaviors. {"Future load values can be adjusted by a user-provided annual growth factor." if self.label.lower() == "load" else ""}

      ### Instructions

      To begin generating probabilistic samples, upload your historical {self.label.lower()} data using the "Browse..." button. Alternatively, you can load example data with the "Load Example CSV" button or download the example data with the "Download Example CSV" button to examine it, edit it and upload it using the "Browse..." button.

      After loading the data, click "Next" at the top right to proceed.

      For more information on how SPI-Tool generates stochastic samples, please refer to the User Guide tab at the top of the screen. Also see the FAQ section for more information.
        """,
            max_width=800,
        )

        file_input_row = pn.Row(file_input, button, sample_button)

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
                visible=self.param.has_warning,
            ),
        )

        info = pn.Card(
            pn.pane.Alert(
                f"""
**Note:** The CSV file should have 2 header rows.

The first row should have the following columns:

- `date`: The date of the data point
- `{self.label.lower()}`: The value of the data point

The data should be sorted by date in ascending order.
The date format should be `YYYY-MM-DD`.

The value of the data point should be a number.

The second row should contain the units without any units for the date.

Here's an example of a valid CSV file:

````plaintext
```
date,{self.label.lower()}
,{self.unit}
2015-07-01,10
2015-07-02,10.5
```
````

If any dates are missing, the values will be interpolated.
If any hourly dates are present or dates are duplicated, those values will be averaged.
""",
                alert_type="info",
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
            ),
            title="Input Historical Data Preview",
            margin=(5, 8, 10, 8),
            collapsible=False,
            align="center",
        )


class TimeseriesPredictionModel(param.Parameterized):
    input_filename = param.Path()
    input_df = param.DataFrame(allow_refs=True)
    processed_df = param.Parameter()
    output_df = param.DataFrame()
    prediction_df = param.DataFrame()
    _prediction_plot_height = param.Integer()

    selected_index = param.Selector(default="all", objects=["all"])
    indices = param.List()

    precomputed_predictions = param.Dict(default={}, precedence=-1)

    ready = param.Boolean(default=False)
    label = param.String(default="Load", allow_refs=True)
    unit = param.String(default="MW", allow_refs=True)
    regression_kind = param.Selector(
        default="normal", objects=["normal", "lognormal"], allow_refs=True
    )
    regression_y_term = param.Selector(
        default="lag_1", objects=["lag_1"], allow_refs=True
    )
    use_day_type = param.Boolean(default=False, allow_refs=True)
    use_month = param.Boolean(default=False, allow_refs=True)

    end_date = param.Date(default=pd.Timestamp("2035-12-31"))
    random_seed = param.Integer(default=42)
    scenario = param.Selector(
        default="sample_1", objects=[f"sample_{i + 1}" for i in range(0, 10)]
    )
    n_samples = param.Integer(default=10)
    annual_growth_rate = param.Number(default=0, bounds=(0, None))

    subset_start_date = param.Date(default=pd.Timestamp("2028-01-01"))
    subset_end_date = param.Date(default=pd.Timestamp("2028-12-31"))

    shift = param.Boolean(default=True)

    def __init__(self, **params):
        super().__init__(**params)

    @param.depends("input_df", watch=True, on_init=True)
    def _update_end_date_bounds(self):
        if self.input_df is None:
            return
        self.param.end_date.bounds = (
            self.input_df.index[-1].date() + pd.Timedelta(days=1),
            None,
        )
        with param.parameterized.discard_events(self):
            self.end_date = (
                self.input_df.index[-1] + pd.Timedelta(days=len(self.input_df.index))
            ).date()

    @param.depends("n_samples", watch=True)
    def _update_scenario_options(self):
        self.scenario = "sample_1"
        self.param.scenario.objects = [
            f"sample_{i + 1}" for i in range(0, self.n_samples)
        ]

    def _update_random_seed(self, evt=None):
        self.random_seed = np.random.randint(0, 2**16)

    def compute_processed_df(self):
        df = (
            self.input_df.resample("1D")
            .mean()
            .interpolate()
            .reset_index()
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .set_index("date")
        )

        df = (
            df.assign(
                month=lambda df: df.index.month_name().str[0:3],
                day_type=lambda df: [
                    "Weekday" if x.dayofweek < 5 else "Weekend" for x in df.index
                ],
                normal_values=lambda df: df[self.label.lower()],
                lognormal_values=lambda df: np.log(df["normal_values"].values),
                normal_lag_1_values=lambda df: df["normal_values"].shift(1),
                lognormal_lag_1_values=lambda df: df["lognormal_values"].shift(1),
            )
            .assign(
                month=lambda df: df["month"].shift(1 if self.shift else 0),
                day_type=lambda df: df["day_type"].shift(1 if self.shift else 0),
            )
            .dropna()
        )

        self.ylim_lower = 0.5 * df["normal_values"].min()
        self.ylim_upper = 1.5 * df["normal_values"].max()

        self.xlim_lower = df.index[0]
        self.xlim_upper = self.end_date

        if self.use_day_type and self.use_month:
            df = df.assign(parameter_index=lambda df: df.day_type + "-" + df.month)
        elif self.use_day_type:
            df = df.assign(parameter_index=lambda df: df.day_type)
        elif self.use_month:
            df = df.assign(parameter_index=lambda df: df.month)
        else:
            df = df.assign(parameter_index="all")

        return df

    def _calculate_dates(self) -> list[pd.Timestamp]:
        return pd.date_range(
            start=self.processed_df.index[-1], end=self.end_date, freq="D"
        )

    def _calculate_indices_x_y(self):
        processed_df, use_day_type, use_month, regression_kind, regression_y_term = (
            self.processed_df,
            self.use_day_type,
            self.use_month,
            self.regression_kind,
            self.regression_y_term,
        )

        indices = []
        xs = []
        ys = []
        if use_day_type and use_month:
            for month in range(1, 13):
                month_str = calendar.month_abbr[month]
                for day_type in ["Weekday", "Weekend"]:
                    df = processed_df.query(
                        "day_type == @day_type and month == @month_str"
                    )
                    i = f"{day_type}-{month_str}"
                    x = df[f"{regression_kind}_values"].values
                    y = df[f"{regression_kind}_{regression_y_term}_values"].values
                    xs.append(x)
                    ys.append(y)
                    indices.append(i)
        elif use_day_type:
            for day_type in ["Weekday", "Weekend"]:
                df = processed_df.query("day_type == @day_type")
                i = day_type
                x = df[f"{regression_kind}_values"].values
                y = df[f"{regression_kind}_{regression_y_term}_values"].values
                xs.append(x)
                ys.append(y)
                indices.append(i)
        elif self.use_month:
            for month in range(1, 13):
                month_str = calendar.month_abbr[month]
                df = processed_df.query("month == @month_str")
                i = month_str
                x = df[f"{regression_kind}_values"].values
                y = df[f"{regression_kind}_{regression_y_term}_values"].values
                xs.append(x)
                ys.append(y)
                indices.append(i)
        else:
            x = processed_df[f"{regression_kind}_values"].values
            y = processed_df[f"{regression_kind}_{regression_y_term}_values"].values
            xs.append(x)
            ys.append(y)
            indices.append("all")

        return indices, xs, ys

    @param.depends("processed_df", "regression_kind", watch=True)
    def _update_prediction_df(self):
        indices, xs, ys = self._calculate_indices_x_y()

        self.indices = indices
        self.xs = xs
        self.ys = ys

        results = []
        for x, y in zip(xs, ys):
            r = linear_regression(x, y)
            results.append(r)

        df = pd.DataFrame(
            index=indices,
            data=results,
        )
        self.precomputed_predictions = {
            index: dict(df.query("index == @index").iloc[0]) for index in indices
        }

        self.prediction_df = df

    def _get_predictions_from_table(self, dt) -> dict:
        if self.use_day_type and self.use_month:
            day_type = "Weekday" if dt.day_of_week < 5 else "Weekend"
            month = calendar.month_abbr[dt.month]
            index = f"{day_type}-{month}"
        elif self.use_day_type:
            day_type = "Weekday" if dt.day_of_week < 5 else "Weekend"
            index = day_type
        elif self.use_month:
            month = calendar.month_abbr[dt.month]
            index = month
        else:
            index = "all"

        return self.precomputed_predictions[index]

    def generate_sample(self, predictions) -> np.array:
        output_values = np.full(
            len(predictions), self.processed_df["normal_values"].values[-1]
        ).astype("float64")

        randomness = scipy.stats.norm.ppf(np.random.rand(len(output_values)))

        slopes = np.array([p["slope"] for p in predictions])
        steyxes = np.array([p["steyx"] for p in predictions])
        intercepts = np.array([p["intercept"] for p in predictions])

        # this has to be a for loop because previous value is used to calculate next value
        for i in range(1, len(output_values)):
            if self.regression_kind == "normal":
                output_values[i] = (
                    intercepts[i]
                    + slopes[i] * output_values[i - 1]
                    + steyxes[i] * randomness[i]
                )
            elif self.regression_kind == "lognormal":
                output_values[i] = np.exp(
                    intercepts[i]
                    + slopes[i] * np.log(output_values[i - 1])
                    + steyxes[i] * randomness[i]
                )

        return output_values

    @param.depends(
        "prediction_df",
        "n_samples",
        "random_seed",
        "end_date",
        "annual_growth_rate",
        watch=True,
    )
    def output_dataframe(self) -> pd.DataFrame:
        if self.input_df is None or self.prediction_df is None:
            return pd.DataFrame()

        np.random.seed(self.random_seed)
        output_dates = self._calculate_dates()

        output_df = pd.DataFrame(data=dict(date=output_dates)).set_index("date")

        predictions = [self._get_predictions_from_table(date) for date in output_dates]

        long_run_mean = self.input_df[self.label.lower()].mean()

        growth_multiplier = (long_run_mean * output_df.reset_index().index) * (
            self.annual_growth_rate / 365 / 100
        )

        for s in range(self.n_samples):
            output_values = self.generate_sample(predictions)
            output_df[f"sample_{s + 1}"] = output_values + growth_multiplier

        self.output_df = output_df
        return self.output_df

    def generate_predictions_csv(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        has_day = self.use_day_type
        has_month = self.use_month
        training_info = f"{'using-default-' if not has_day and not has_month else ''}{'using-day-' if has_day else ''}{'using-month-' if has_month else ''}".strip(
            "-"
        )
        regression_kind = self.regression_kind
        label = self.label

        # Set filename dynamically
        self.download_predictions_button.filename = f"prediction_{label}_{regression_kind}_{training_info}_downloaded-{timestamp}.csv"

        csv_buffer = io.StringIO()
        self.prediction_df.to_csv(csv_buffer)
        csv_buffer.seek(0)
        return csv_buffer

    def generate_scenarios_csv(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        num_samples = len(self.output_df.columns)
        has_day = self.use_day_type
        has_month = self.use_month
        training_info = f"{'using-default-' if not has_day and not has_month else ''}{'using-day-' if has_day else ''}{'using-month-' if has_month else ''}".strip(
            "-"
        )
        end_date = self.end_date.strftime("%Y-%m-%d")
        regression_kind = self.regression_kind
        label = self.label

        # Set filename dynamically
        self.download_scenarios_button.filename = f"output_{label}_{regression_kind}_{num_samples}-samples_{training_info}_end-date-{end_date}_downloaded-{timestamp}.csv"

        csv_buffer = io.StringIO()
        self.output_df.pipe(
            lambda df: df.set_index(df.index.strftime("%Y-%m-%d"))
        ).to_csv(csv_buffer)
        csv_buffer.seek(0)
        return csv_buffer

    def compute_sample_set(self):
        self._update_processed_df()

    @param.depends("shift", "label", "use_day_type", "use_month", watch=True)
    def _update_processed_df(self):
        self.processed_df = (
            self.compute_processed_df()
        )  # triggers updates of dependencies of processed_df

    def _update_histogram_input_plot(self, fig=None, ax=None):
        if fig is None:
            fig = mpl.figure.Figure(figsize=(10, 6))
        if ax is None:
            ax = fig.add_subplot(111)

        if self.input_df is None or self.processed_df is None:
            ax.axhline(0, color="black")
            ax.set_title("Histogram")
            return fig

        # Create histogram plots
        self.processed_df.query(f"parameter_index == '{self.selected_index}'")[
            self.label.lower()
        ].plot.hist(
            ax=ax,
            bins=30,
            color="blue",
            alpha=0.25,
            label="Input Historical Data",
            orientation="horizontal",
            density=True,
        )

        ax.set_ylabel(f"{self.label} ({self.unit})")

        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.f"))

        ax.set_ylim(self.ylim_lower, self.ylim_upper)

        ax.legend(loc="upper right")
        ax.set_xlabel("")

        return fig

    def _update_histogram_output_plot(self, fig=None, ax=None):
        if fig is None:
            fig = mpl.figure.Figure(figsize=(10, 6))
        if ax is None:
            ax = fig.add_subplot(111)

        if self.input_df is None or self.processed_df is None or self.output_df is None:
            ax.axhline(0, color="black")
            ax.set_title("Histogram")
            return fig

        df = self.output_df.assign(
            month=lambda df: df.index.month_name().str[0:3],
            day_type=lambda df: [
                "Weekday" if x.dayofweek < 5 else "Weekend" for x in df.index
            ],
        )

        if self.use_day_type and self.use_month:
            df = df.assign(parameter_index=lambda df: df.day_type + "-" + df.month)
        elif self.use_day_type:
            df = df.assign(parameter_index=lambda df: df.day_type)
        elif self.use_month:
            df = df.assign(parameter_index=lambda df: df.month)
        else:
            df = df.assign(parameter_index="all")

        # Create histogram plots
        df.query(f"parameter_index == '{self.selected_index}'")[
            self.scenario
        ].plot.hist(
            ax=ax,
            bins=30,
            density=True,
            color="red",
            alpha=0.25,
            label=f"{self.scenario}",
            orientation="horizontal",
        )

        # ax.set_ylabel(f"{self.label} ({self.unit})")

        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.f"))

        ax.set_ylim(self.ylim_lower, self.ylim_upper)

        ax.legend(loc="upper right")
        ax.set_xlabel("\nNormalized Frequency")

        return fig

    def _update_histogram_mean_plot(self, fig=None, ax=None):
        if fig is None:
            fig = mpl.figure.Figure(figsize=(10, 6))
        if ax is None:
            ax = fig.add_subplot(111)

        if self.input_df is None or self.processed_df is None or self.output_df is None:
            ax.axhline(0, color="black")
            ax.set_title("Histogram")
            return fig

        df = pd.DataFrame(
            self.output_df.mean(axis=1),
            index=self.output_df.index,
            columns=["Mean"],
        ).assign(
            month=lambda df: df.index.month_name().str[0:3],
            day_type=lambda df: [
                "Weekday" if x.dayofweek < 5 else "Weekend" for x in df.index
            ],
        )

        if self.use_day_type and self.use_month:
            df = df.assign(parameter_index=lambda df: df.day_type + "-" + df.month)
        elif self.use_day_type:
            df = df.assign(parameter_index=lambda df: df.day_type)
        elif self.use_month:
            df = df.assign(parameter_index=lambda df: df.month)
        else:
            df = df.assign(parameter_index="all")

        # Create histogram plots
        df.query(f"parameter_index == '{self.selected_index}'").plot.hist(
            ax=ax,
            bins=30,
            density=True,
            color="black",
            alpha=0.25,
            label=f"{self.scenario} (average)",
            orientation="horizontal",
        )

        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.f"))

        ax.set_ylim(self.ylim_lower, self.ylim_upper)

        ax.legend(loc="upper right")
        ax.set_xlabel("")

        return fig

    @param.depends("output_df", "scenario")
    def _update_line_plot(self, fig=None, ax=None):
        if fig is None:
            fig = mpl.figure.Figure(figsize=(10, 6))
        if ax is None:
            ax = fig.add_subplot(111)

        if self.output_df is None:
            return fig

        if self.processed_df is None:
            ax.axhline(0, color="black")
            ax.set_title("Historical (blue) vs Generated Samples (red)")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{self.label} ({self.unit})")
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.f"))
            return fig

        ax.plot(
            self.processed_df.index,
            self.processed_df[self.label.lower()],
            color="blue",
            label="Input",
            linewidth=2,
        )

        for i, col in enumerate(
            [col for col in self.output_df.columns if col != self.scenario]
        ):
            if i == 0:
                label = "All samples"
            else:
                label = None
            ax.plot(
                self.output_df.index,
                self.output_df[col],
                color="red",
                alpha=1 / self.n_samples,
                label=label,
            )

        ax.plot(
            self.output_df.index,
            self.output_df[self.scenario],
            color="red",
            linewidth=2,
            label=f"{self.scenario}",
        )

        ax.plot(
            self.output_df.index,
            self.output_df.mean(axis=1),
            color="black",
            linewidth=2,
            label=f"Average {self.label}",
        )

        ax.set_title("Historical (blue) vs Generated Samples (red)")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.label} ({self.unit})")

        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.f"))

        ax.legend(loc="upper right")

        ax.set_ylim(self.ylim_lower, self.ylim_upper)

        return fig

    @param.depends("output_df", "scenario", "selected_index")
    def _update_plot(self, fig=None):
        if fig is None:
            fig = mpl.figure.Figure(figsize=(16, 5))

        axs = fig.subplots(
            1,
            5,
            sharey=True,
            gridspec_kw={"wspace": 0.0},
            width_ratios=[4, 1, 1, 1, 1],
        )

        ax = axs[0]
        self._update_line_plot(fig=fig, ax=ax)

        self._update_histogram_input_plot(fig=fig, ax=axs[1])
        self._update_histogram_output_plot(fig=fig, ax=axs[2])
        self._update_histogram_mean_plot(fig=fig, ax=axs[3])

        # Collect handles and labels from the first three axes
        handles, labels = [], []
        for ax in axs[1:4]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
            l = ax.get_legend()
            if l is not None:
                l.remove()

        # Place the legend in the fourth axis
        axs[4].legend(handles, labels, loc="center")
        axs[4].axis("off")  # Hide the ticks and frame for the fourth axis

        axs[1].get_yaxis().set_visible(False)

        # axs[1].set_title("Input")
        axs[2].set_title(f"{self.selected_index}")
        # axs[3].set_title("Average")

        min = 0.0
        max = np.max([ax.get_xlim()[1] for ax in axs[1:4]])
        for ax in axs[1:4]:
            ax.set_xlim(min, max)

        return fig

    @param.depends("prediction_df", watch=True)
    def _update_prediction_plot_height(self):
        if self.prediction_df is None:
            self._prediction_plot_height = 240
        self._prediction_plot_height = 240 * (
            math.ceil(len(self.prediction_df.index) / 4)
        )

    @param.depends("output_df", "scenario")
    def _update_input_output_mean_histogram_plots(self):
        if self.processed_df is None or self.output_df is None:
            return mpl.figure.Figure()

        indices = self.indices
        n_plots = len(indices)

        n_cols = 4
        n_rows = math.ceil(n_plots / n_cols)

        fig = mpl.figure.Figure(figsize=(n_cols * 5, n_rows * 5))
        axs = fig.subplots(n_rows, n_cols, sharey=True, sharex=True)

        axs = axs.flatten()

        output_df = pd.concat(
            [self.output_df.mean(axis=1), self.output_df[self.scenario]], axis=1
        )
        output_df.index = self.output_df.index
        output_df.columns = ["Mean", self.scenario]
        output_df = output_df.assign(
            month=lambda df: df.index.month_name().str[0:3],
            day_type=lambda df: [
                "Weekday" if x.dayofweek < 5 else "Weekend" for x in df.index
            ],
        )

        if self.use_day_type and self.use_month:
            output_df = output_df.assign(
                parameter_index=lambda df: df.day_type + "-" + df.month
            )
        elif self.use_day_type:
            output_df = output_df.assign(parameter_index=lambda df: df.day_type)
        elif self.use_month:
            output_df = output_df.assign(parameter_index=lambda df: df.month)
        else:
            output_df = output_df.assign(parameter_index="all")

        for i in range(n_plots):
            idx = indices[i]
            data = self.processed_df.query(f"parameter_index == '{idx}'")[
                self.label.lower()
            ]
            axs[i].hist(data, color="blue", alpha=0.25, label="Input", density=True)

            # data = output_df.query(f"parameter_index == '{idx}'")["Mean"]
            # axs[i].hist(
            #     data, color="red", alpha=0.25, label="Average Output", density=True
            # )

            data = output_df.query(f"parameter_index == '{idx}'")[self.scenario]
            axs[i].hist(
                data,
                color="green",
                alpha=0.25,
                label=f"{self.scenario}",
                density=True,
            )

            axs[i].set_title(idx)

            if i // n_cols == n_rows - 1:
                axs[i].set_xlabel(f"{self.label} ({self.unit})")
            if i % n_cols == 0:
                axs[i].set_ylabel("Normalized Frequency")

            axs[i].legend(loc="best")

        for i in range(n_plots, len(axs)):
            axs[i].axis("off")

        return fig

    @param.depends("prediction_df")
    def _update_prediction_timeseries(self):
        if self.prediction_df is None:
            return mpl.figure.Figure()

        fig = mpl.figure.Figure(figsize=(16, 6))
        axs = fig.subplots(1, 1, sharey=True, sharex=True)

        ax = axs

        # make a bar plot of the prediction_df where each row is a bar
        self.prediction_df.assign(
            long_run_mean=lambda df: -1 * df["long_run_mean"]
        ).plot.bar(
            y="long_run_mean",
            ax=ax,
            color="blue",
            alpha=0.5,
            yerr="steyx",
            capsize=5,
            legend=False,
        )
        ax.set_title(
            "Long Run Mean and Standard Error of the Residuals of the Regression"
        )

        return fig

    @param.depends("prediction_df")
    def _update_prediction_plots(self):
        if self.prediction_df is None:
            return mpl.figure.Figure()

        indices, xs, ys = self.indices, self.xs, self.ys
        n_plots = len(indices)

        n_cols = 4
        n_rows = math.ceil(n_plots / n_cols)

        fig = mpl.figure.Figure(figsize=(n_cols * 5, n_rows * 5))
        axs = fig.subplots(n_rows, n_cols, sharey=True, sharex=True)

        axs = axs.flatten()

        for i in range(n_plots):
            idx = indices[i]
            x = xs[i]
            y = ys[i]
            slope = self.prediction_df.loc[idx, "slope"]
            intercept = self.prediction_df.loc[idx, "intercept"]
            axs[i].plot(x, y, "o", label="historical data")
            axs[i].plot(x, intercept + slope * x, "r", label="fitted")
            axs[i].set_title(f"{idx}")

            if i // n_cols == n_rows - 1:
                axs[i].set_xlabel(f"{self.regression_kind} values")
            if i % n_cols == 0:
                axs[i].set_ylabel(f"{self.regression_kind} {self.regression_y_term}")

        for i in range(n_plots, len(axs)):
            axs[i].axis("off")

        return fig

    @param.depends("output_df", "scenario", "subset_start_date", "subset_end_date")
    def _update_subset_plot(self, fig=None):
        if fig is None:
            fig = mpl.figure.Figure(figsize=(8, 5))

        if self.output_df is None:
            return fig

        axs = fig.subplots(1, 1, gridspec_kw={"wspace": 0.0})

        ax = axs

        if pd.Timestamp(self.subset_end_date) > pd.Timestamp(self.end_date):
            self.subset_end_date = self.end_date

        if pd.Timestamp(self.subset_start_date) > pd.Timestamp(self.subset_end_date):
            self.subset_start_date = self.subset_end_date - pd.Timedelta(days=7)

        for i, col in enumerate(
            [col for col in self.output_df.columns if col != self.scenario]
        ):
            if i == 0:
                label = "All samples"
            else:
                label = None
            ax.plot(
                self.output_df.loc[self.subset_start_date : self.subset_end_date].index,
                self.output_df.loc[self.subset_start_date : self.subset_end_date][col],
                color="red",
                alpha=1 / self.n_samples,
                label=label,
            )

        ax.plot(
            self.output_df.loc[self.subset_start_date : self.subset_end_date].index,
            self.output_df.loc[self.subset_start_date : self.subset_end_date][
                self.scenario
            ],
            color="red",
            linewidth=2,
            label=f"{self.scenario}",
        )

        ax.plot(
            self.output_df.loc[self.subset_start_date : self.subset_end_date].index,
            self.output_df.loc[self.subset_start_date : self.subset_end_date].mean(
                axis=1
            ),
            color="black",
            linewidth=2,
            label=f"Average {self.label}",
        )

        ax.set_title("Time slice of Probabilistic Samples")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.label} ({self.unit})")

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_ylim(self.ylim_lower, self.ylim_upper)

        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.f"))

        ax.legend(loc="upper right")

        return fig

    def _update_limits(self, x_range, y_range):
        if y_range is not None:
            self.ylim_lower, self.ylim_upper = y_range
        if x_range is not None:
            self.xlim_lower, self.xlim_upper = x_range

    def _update_df(self, df, units=False):
        if df is None:
            return pn.indicators.LoadingSpinner(
                value=True, width=50, height=50, margin=10
            )
        df = df.copy()
        df.index = df.index.strftime("%Y-%m-%d")
        if units:
            # Make header a multi index header
            df.columns = pd.MultiIndex.from_tuples(
                [(col, self.unit) for col in df.columns]
            )
        return pn.widgets.Tabulator(
            df.round(2),
            pagination="remote",
            page_size=10,
            sizing_mode="stretch_width",
            frozen_columns=["date"],
            selectable=False,
            disabled=True,
        )

    def _init_view(self):
        self.download_predictions_button = pn.widgets.FileDownload(
            filename="predictions.csv",
            callback=self.generate_predictions_csv,
            button_type="primary",
            icon="download",
            label="Download Sample Generation Parameters",
            width=400,
        )

        self.download_scenarios_button = pn.widgets.FileDownload(
            filename="data.csv",
            callback=self.generate_scenarios_csv,
            button_type="primary",
            icon="download",
            label="Download Probabilistic Samples",
            width=400,
        )

        self.processed_df_view = pn.Card(
            pn.pane.Markdown("""
                This table shows the input data after it is processed. The table includes columns for the values and the lag values used for the fitting in the regression, as well as other internal intermediate state. This table is useful for debugging and understanding what the SPI-Tool is doing under the hood.
                        """),
            pn.bind(self._update_df, self.param.processed_df),
            title=f"Processed {self.label} Data",
            collapsed=True,
            collapsible=True,
        )

        normal_distribution_explanation = pn.pane.Markdown(r"""

            The following table shows the results of the linear regression analysis for each group of historical data. The table includes the following columns:

            $$\text{steyx} = S = \sqrt{\frac{\sum{(y - \hat{y})^2}}{n - 2}}$$

            $$\text{mean reversion} = \theta = \( 1 - \varphi \) = -\text{slope}$$

            $$\text{long run mean} = \mu = \frac{\text{intercept}}{-\text{slope}}$$

            $$\text{volatility} = \left|\frac{S}{ \mu } \times 100\right|$$

            The AR(1) process used is:

            $$y_t = \mu \cdot \( 1 - \varphi \) + \varphi \cdot y_{t - 1} + S \cdot Z_t$$

            and can be rewritten as:

            $$y_t = \mu \cdot \theta + \( 1 - \theta \) \cdot y_{t - 1} + S \cdot Z_t$$

            where:

            - $$ Z_t \sim N(0,1) $$: A standard normal random variable (mean 0, variance 1), introducing stochasticity into the process.
            - $$ S $$: The standard deviation of the noise, scaling the randomness.

      """)

        lognormal_distribution_explanation = pn.pane.Markdown(r"""

            The following table shows the results of the linear regression analysis for each group of historical data. The table includes the following columns:

            $$\text{steyx} = S = \sqrt{\frac{\sum{(y - \hat{y})^2}}{n - 2}}$$

            $$\text{mean reversion} = \theta = \( 1 - \varphi \) = -\text{slope}$$

            $$\text{long run mean} = \mu = \frac{\text{intercept}}{-\text{slope}}$$

            $$\text{volatility} = \left|\frac{S}{ \mu } \times 100\right|$$

            The lognormal AR(1) process used is:

            $$\ln(y_t) = \mu \cdot \( 1 - \varphi \) + \varphi \cdot \ln(y_{t - 1}) + S \cdot Z_t$$

            and can be rewritten as:

            $$\ln(y_t) = \mu \cdot \theta + \( 1 - \theta \) \cdot \ln(y_{t - 1}) + S \cdot Z_t$$

            where:

            - $$ Z_t \sim N(0,1) $$: A standard normal random variable (mean 0, variance 1), introducing stochasticity into the process.
            - $$ S $$: The standard deviation of the noise, scaling the randomness.

            """)

        self.prediction_df_view = pn.Card(
            pn.bind(
                lambda distribution: normal_distribution_explanation
                if distribution == "normal"
                else lognormal_distribution_explanation,
                self.param.regression_kind,
            ),
            pn.widgets.Tabulator.from_param(
                self.param.prediction_df,
                pagination="remote",
                page_size=5,
                sizing_mode="stretch_width",
                selectable=False,
                disabled=True,
            ),
            pn.Row(
                self.download_predictions_button,
                align="center",
                width=250,
                margin=(5, 8, 10, 8),
            ),
            pn.pane.Markdown("""
            The following is a visualization of the y(t) and the y(t-1) for each group of historical data. The red line is the fitted line for the regression.
            This plot is useful for debugging and for understand what SPI-Tool is doing under the hood.
                             """),
            # pn.pane.Matplotlib(
            #     self._update_prediction_timeseries,
            #     height=400,
            # ),
            pn.pane.Matplotlib(
                self._update_prediction_plots,
                height=self.param._prediction_plot_height,
            ),
            title="Sample Generation Parameters",
            collapsed=True,
            collapsible=True,
        )

        self.output_df_view = pn.Card(
            pn.pane.Markdown(r"""

            Output values can be generated using the AR(1) process described above.

            The following table shows the generated output values for each sample.
                 """),
            pn.bind(self._update_df, self.param.output_df, units=True),
            pn.Row(
                self.download_scenarios_button,
                align="center",
                width=250,
                margin=(5, 8, 10, 8),
            ),
            pn.pane.Markdown("""
            The following is a visualization of the histogram of the sample chosen below and the input data
            This plot is useful for validation purposes.
                             """),
            pn.widgets.Select.from_param(
                self.param.scenario,
                name="Sample",
                options=self.param.scenario.objects,
            ),
            pn.pane.Matplotlib(
                self._update_input_output_mean_histogram_plots,
                height=self.param._prediction_plot_height,
            ),
            title="Output Data",
            collapsed=False,
            collapsible=True,
        )

        drp = pn.widgets.DateRangePicker(
            name="Date Range",
            value=(
                self.subset_start_date,
                self.subset_end_date,
            ),
            start=self.input_df.index[-1].date(),
            end=None,
        )

        def _watch_date_range_picker(event):
            self.subset_start_date = drp.value[0]
            self.subset_end_date = drp.value[1]

        drp.param.watch(_watch_date_range_picker, "value")

        def _update_date_range_picker_bounds(df):
            try:
                drp.start = df.index[0].date()
                drp.end = df.index[-1].date()
            except Exception as e:
                drp.value = (
                    df.index[0].date() + pd.Timedelta(days=1),
                    df.index[-1].date(),
                )
                drp.start = df.index[0].date()
                drp.end = df.index[-1].date()

        pn.bind(_update_date_range_picker_bounds, self.param.output_df, watch=True)

        selected_index_selector = pn.widgets.Select.from_param(
            self.param.selected_index
        )

        @param.depends(self.param.indices, watch=True)
        def _update_selected_index(indices):
            with param.parameterized.discard_events(self):
                self.param.selected_index.objects = indices
                self.param.selected_index.default = indices[0]
            selected_index_selector.options = indices
            selected_index_selector.value = indices[0]

        self.scenario_model_data_view = pn.Card(
            pn.pane.Markdown(r"""
            This section allows configuring the settings involved when generating the probabilistic samples. Changing any setting will update the output data at the bottom of this page accordingly.
                             """),
            pn.FlexBox(
                pn.Row(
                    pn.widgets.TooltipIcon(
                        value="Uses whether day is 'Weekend' or 'Weekday' to group the input historical data for prediction.",
                        sizing_mode=None,
                    ),
                    pn.widgets.Checkbox.from_param(
                        self.param.use_day_type,
                        name="Consider day-of-week effects",
                        sizing_mode=None,
                    ),
                    align=("center", "center"),
                ),
                pn.Row(
                    pn.widgets.TooltipIcon(
                        value="Uses the month of the year to group the input historical data for prediction.",
                        sizing_mode=None,
                    ),
                    pn.widgets.Checkbox.from_param(
                        self.param.use_month,
                        name="Consider seasonal effects",
                        sizing_mode=None,
                    ),
                    align=("center", "center"),
                ),
                pn.Row(
                    pn.widgets.TooltipIcon(
                        value="The assumed probability distribution of historical observations.",
                        sizing_mode=None,
                    ),
                    pn.widgets.Select.from_param(
                        self.param.regression_kind,
                        name="Probability Distribution",
                        sizing_mode=None,
                    ),
                    align=("center", "center"),
                ),
                pn.Row(
                    pn.widgets.TooltipIcon(
                        value="Changes the start date of the sample set.",
                        sizing_mode=None,
                    ),
                    pn.bind(
                        lambda df: pn.widgets.DatePicker(
                            name="Start date",
                            value=df.index[-1],
                            sizing_mode=None,
                            disabled=True,
                        ),
                        self.param.input_df,
                    ),
                ),
                pn.Row(
                    pn.widgets.TooltipIcon(
                        value="Changes the end date of the sample set.",
                        sizing_mode=None,
                    ),
                    pn.widgets.DatePicker.from_param(
                        self.param.end_date,
                        sizing_mode=None,
                    ),
                ),
                pn.Row(
                    pn.widgets.TooltipIcon(
                        value="Number of probabilistic samples to generate.",
                        sizing_mode=None,
                    ),
                    pn.widgets.IntInput.from_param(
                        self.param.n_samples,
                        name="Number of Samples",
                        start=1,
                        sizing_mode=None,
                    ),
                ),
                pn.Row(
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
                ),
                pn.Row(
                    pn.widgets.TooltipIcon(
                        value=f"Annual {self.label} growth % that is applied after the forecast is generated.",
                        sizing_mode=None,
                    ),
                    pn.widgets.FloatInput.from_param(
                        self.param.annual_growth_rate,
                        name="Annual Growth %",
                        start=0,
                        step=0.1,
                        sizing_mode=None,
                    ),
                )
                if self.label.lower() == "load"
                else None,
            ),
            pn.Column(
                pn.pane.Matplotlib(
                    self._update_plot, sizing_mode="stretch_width", height=400
                ),
            ),
            pn.Column(
                pn.Row(
                    pn.pane.Matplotlib(
                        self._update_subset_plot,
                        sizing_mode="stretch_width",
                        height=400,
                    ),
                    pn.Column(
                        pn.Row(
                            pn.widgets.TooltipIcon(
                                value="Select a sample to highlight in the plots.",
                                sizing_mode=None,
                                align=("center", "center"),
                            ),
                            pn.widgets.Select.from_param(
                                self.param.scenario,
                                name="Sample",
                                options=self.param.scenario.objects,
                            ),
                        ),
                        pn.Row(
                            pn.widgets.TooltipIcon(
                                value="Select a generation parameter category to view a different facet of the histogram.",
                                sizing_mode=None,
                                align=("center", "center"),
                            ),
                            selected_index_selector,
                        ),
                        pn.Row(
                            pn.widgets.TooltipIcon(
                                value="Select start and end dates to view a subset of the sample set.",
                                sizing_mode=None,
                                align=("center", "center"),
                            ),
                            drp,
                        ),
                    ),
                )
            ),
            title="Sample Generation Settings",
            collapsed=False,
            loading=pn.state.param.busy,
        )

        self.layout = pn.Column(
            self.scenario_model_data_view,
            self.processed_df_view,
            self.prediction_df_view,
            self.output_df_view,
        )

        return self.layout

    def panel(self):
        self._init_view()

        def _onload():
            self.compute_sample_set()

        pn.state.onload(_onload)

        return self.layout


class RegressionPipeline(pn.viewable.Viewer):
    label = param.String(default="Load")
    unit = param.String(default="MW")
    regression_kind = param.Selector(default="normal", objects=["normal", "lognormal"])
    use_day_type = param.Boolean(default=True)
    use_month = param.Boolean(default=True)
    default_filename = param.Path(
        default=os.path.join(_utils.DATA_FOLDER_PATH, "miso-daily-demand.csv")
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipeline = _helper.CustomPipeline(debug=True, inherit_params=True)

        self.input_data = TimeseriesInput(
            label=self.param.label,
            unit=self.param.unit,
            default_filename=self.param.default_filename,
            regression_kind=self.param.regression_kind,
            use_day_type=self.param.use_day_type,
            use_month=self.param.use_month,
        )

        self.prediction_model = TimeseriesPredictionModel

        self.pipeline.add_stage(
            f"Step 1: Input Historical {self.label}",
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


class AR1Process(pn.viewable.Viewer):
    θ = param.Number(default=0.1, bounds=(0, 1))
    μ = param.Number(default=50, bounds=(0, 100))
    σ = param.Number(default=0.1)
    dt = param.Number(default=1)
    T = param.Integer(default=100)
    x0 = param.Number(default=90, bounds=(0, 100))
    X = param.Array()

    def __init__(self, **params):
        super().__init__(**params)

    @param.depends("θ", "μ", "σ", "dt", "T", "x0", watch=True, on_init=True)
    def _update_process(self):
        np.random.seed(42)
        X = np.zeros(self.T)
        X[0] = self.x0

        for t in range(1, self.T):
            dW = np.random.normal(0, 1)
            X[t] = (1 - self.θ) * X[t - 1] + self.θ * self.μ + self.σ * dW

        self.X = X

    @param.depends("X")
    def _update_plot(self):
        fig = mpl.figure.Figure(figsize=(10, 5))
        axs = fig.subplots(1, 1)
        ax = axs
        ax.plot(self.X, label="Simulated Mean-Reverting Process")
        ax.axhline(self.μ, color="red", linestyle="--", label="Long-Run Mean")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.set_ylim(0, 100)
        ax.set_title("AR1 Mean Reversion Process")
        return fig

    def __panel__(self):
        return pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    r"""
Here is an interactive pane to gain intuition about this process.
                """
                ),
                pn.Column(
                    pn.Row(
                        pn.widgets.FloatInput.from_param(
                            self.param.θ,
                            name="Mean Reversion Rate",
                            step=0.01,
                        ),
                        pn.widgets.FloatInput.from_param(
                            self.param.μ,
                            name="Long-Run Mean",
                        ),
                        pn.widgets.FloatInput.from_param(
                            self.param.σ,
                            name="Standard Error of the Residuals",
                        ),
                        pn.widgets.FloatInput.from_param(
                            self.param.x0,
                            name="Initial Value",
                        ),
                    ),
                ),
                pn.pane.Matplotlib(
                    self._update_plot, sizing_mode="stretch_width", height=400
                ),
            ),
            title="AR(1) Mean Reversion Process",
        )


class RegressionManual(pn.viewable.Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self.input_data = TimeseriesInput()
        self.input_data.load_example_data()
        self.output_data = TimeseriesPredictionModel(
            input_df=self.input_data.input_df,
            use_month=True,
            n_samples=10,
        )
        self.output_data.panel()

    def __panel__(self):
        drp = pn.widgets.DateRangePicker(
            value=(
                self.output_data.subset_start_date,
                self.output_data.subset_end_date,
            ),
            start=self.output_data.output_df.index[0].date(),
            end=self.output_data.output_df.index[-1].date(),
        )

        @param.depends(self.output_data.param.output_df, watch=True)
        def _update_date_range_picker_bounds(df):
            drp.start = df.index[0].date()
            drp.end = df.index[-1].date()

        def _watch_date_range_picker(event):
            self.output_data.subset_start_date = drp.value[0]
            self.output_data.subset_end_date = drp.value[1]

        drp.param.watch(_watch_date_range_picker, "value")

        return pn.Column(
            pn.pane.Markdown(
                r"""
# Mean-Reverting Process Parameter Fitting

A "mean-reverting process" is defined as a type of stochastic process whose value tends to move back toward a long-term average (or "mean") over time. Whenever the process deviates from this average, it is pulled back toward it at a rate determined by a "mean-reversion" parameter.

This user guide explains how to fit the parameters for a mean-reverting process, using historical load data as an example. The same approach applies to many other inputs in a stochastic analysis (e.g., natural gas prices, coal prices).

## Input Historical Data

For this example, it is assumed that the input historical data (e.g., historical load) is normally distributed. For normally distributed data, the parameters of the mean-reverting process can be estimated by performing a linear regression.

For a mean reversion process, you should provide historical data with minimal long term trends, and data that does not have any extreme conditions. Historical data that has seasonal or day-of-week patterns are best suited for the current implementation of the mean-reverting process.

In this regression, the explanatory variable is daily load, and the response variable is the the day-to-day change in load. Visualized below are the historical load data used in this example.
""",
                max_width=800,
            ),
            pn.pane.Matplotlib(
                self.input_data.plot,
                sizing_mode="stretch_width",
                height=400,
            ),
            pn.pane.Markdown(
                r"""

## Sample Generation Parameters

Below is a table of sample generation parameters: slope, intercept, standard deviation of the residuals, mean reversion rate, long-run mean, and volatility. Mean reversion rate, long-run mean, and volatility are parameters used to generate the probabilistic samples, as described in the next section of this guide.

Load patterns (and the behaviors of other IRP inputs, like natural gas price) often vary by season and day of the week. For instance, load may be higher in the summer than in the spring and fall. To capture these differences, separate mean reverting processes can be fit for each month and day of the week (classified here as weekdays or weekends). The “index” column in the table below identifies these temporal groupings. Users can enable or disable these groupings based on the patterns in the historical data for their own system.
""",
                max_width=800,
            ),
            pn.Card(
                pn.Column(
                    pn.pane.Markdown(
                        "Try changing the 'Consider day-of-week effects', 'Consider seasonal effects' and/or 'Probability Distribution' options to see their impacts on the fitted sample generation parameters.",
                        max_width=800,
                    ),
                    pn.Row(
                        pn.Column(
                            pn.widgets.Checkbox.from_param(
                                self.output_data.param.use_day_type,
                                name="Consider day-of-week effects",
                                sizing_mode=None,
                            ),
                            align="center",
                        ),
                        pn.Column(
                            pn.widgets.Checkbox.from_param(
                                self.output_data.param.use_month,
                                name="Consider seasonal effects",
                                sizing_mode=None,
                            ),
                            align="center",
                        ),
                        pn.Column(
                            pn.widgets.Select.from_param(
                                self.output_data.param.regression_kind,
                                name="Probability Distribution",
                                sizing_mode=None,
                            ),
                            align="center",
                        ),
                        align="center",
                    ),
                    pn.widgets.Tabulator.from_param(
                        self.output_data.param.prediction_df,
                        page_size=5,
                        pagination="remote",
                        sizing_mode="stretch_width",
                        selectable=False,
                        disabled=True,
                        loading=pn.state.param.busy,
                    ),
                ),
                title="Mean-Reverting Process Parameters",
            ),
            pn.pane.Markdown(
                r"""


The equation for modeling a linear mean-reverting process is:

$$x_{t+1} - x_t = - \theta \cdot x_t + \theta \cdot \mu + \sigma \cdot Z_t$$

The above AR1 process can also be written like this:

$$x_{t} = (1 - \theta) \cdot x_{t-1} + \theta \cdot \mu + \sigma \cdot Z_t$$

where:

- $$x_{t}$$ is the value of the time series at time $$t$$
- $$x_{t-1}$$ is the value of the time series at time $$t-1$$
- $$\theta$$ is the mean reversion parameter ($$-\alpha$$)
- $$\mu$$ is the long-run mean of the process
- $$\sigma$$ is standard error of the regression
- $$Z_t$$ is a standard normal sample from [-1, 1]

For the lognormal mean-reverting process, the equation is:

$$ln(x_{t+1}) - ln(x_t) = - \theta \cdot ln(x_t) + \theta \cdot \mu + \sigma \cdot Z_t$$

The parameters $$\theta$$, $$\mu$$ and $$\sigma$$ can be calculated by performing a regression of $$x_t$$  against $$x_{t+1}-x_t$$.

                """,
                max_width=800,
            ),
            pn.pane.PNG(
                os.path.join(_utils.IMAGE_FOLDER_PATH, "load-fitting-parameters.png"),
                max_width=800,
            ),
            pn.pane.Markdown(
                r"""
The slope of the regression line is equal to $$-\theta$$, the intercept is equal to $$\theta \times \mu$$, and the standard deviation of the residuals is equal to $$\sigma$$.

Using the parameters calculated from the regression, the mean-reverting process is simulated to generate the probabilistic samples. The figure below shows the historical load data and the probabilistic samples generated by the mean-reverting process.

""",
                max_width=800,
            ),
            AR1Process(),
            pn.pane.Markdown("""
## Probabilistic Samples Generation

With those parameters, the output samples can be generated.
                             """),
            pn.Card(
                pn.Row(
                    pn.pane.Matplotlib(
                        self.output_data._update_plot,
                        sizing_mode="stretch_width",
                        height=400,
                    ),
                    max_height=300,
                ),
                pn.Row(
                    pn.pane.Matplotlib(
                        self.output_data._update_subset_plot,
                        max_height=400,
                    ),
                    pn.Column(
                        pn.pane.Markdown("""
Select a subset of the sample set data to view by changing the start and end dates below.
                                         """),
                        drp,
                        pn.pane.Markdown("""
Select a sample to view by changing the dropdown menu below.
                                         """),
                        pn.widgets.Select.from_param(
                            self.output_data.param.scenario,
                            name="Sample",
                            options=self.output_data.param.scenario.objects,
                        ),
                        pn.pane.Markdown("""
Select the regression kind, whether to use day category for training, and whether to use month for training below.
                                         """),
                        pn.Column(
                            pn.widgets.Select.from_param(
                                self.output_data.param.regression_kind,
                                name="Probability Distribution",
                                sizing_mode=None,
                            ),
                            pn.Row(
                                pn.widgets.Checkbox.from_param(
                                    self.output_data.param.use_day_type,
                                    name="Use day category for training",
                                ),
                                pn.widgets.Checkbox.from_param(
                                    self.output_data.param.use_month,
                                    name="Use month for training",
                                ),
                            ),
                        ),
                    ),
                ),
                title="Observed and Generated Samples, 2015-2039",
            ),
            pn.pane.Markdown(
                r"""
With the sample set generated, the output data can be downloaded as a CSV file by clicking on the button below. The CSV file contains the sample set data. The table below shows a preview of that sample set.
""",
                max_width=800,
            ),
            pn.Card(
                pn.Row(
                    pn.pane.Markdown(
                        "Change the slider to increase or decrease the number of samples in the output csv file."
                    ),
                    pn.widgets.IntSlider.from_param(
                        self.output_data.param.n_samples,
                        start=1,
                        end=50,
                        throttled=True,
                    ),
                ),
                pn.widgets.Tabulator.from_param(
                    self.output_data.param.output_df,
                    pagination="remote",
                    page_size=5,
                    frozen_columns=["date"],
                    selectable=False,
                    disabled=True,
                ),
                pn.Row(
                    self.output_data.download_scenarios_button,
                    align="center",
                    width=250,
                    margin=(5, 8, 10, 8),
                ),
                title="Output Data",
                collapsed=False,
                collapsible=True,
            ),
            pn.pane.Markdown(
                rf"""

## Conclusion

This user guide demonstrates the process of fitting the parameters of a mean-reverting process to historical {self.input_data.label} data. This calibrated mean-reverting process is then used to generate probabilistic {self.input_data.label} samples. This guide also allows users to explore how sample generation settings impact the probabilistic samples generated by SPI-Tool.

## References

1. Blanco, Carlos, and Soronow, David. "Mean reverting processes-energy price processes used for derivatives pricing & risk management." Commodities now 5.2 (2001): 68-72.

2. PacifiCorp 2023 IRP, Volume II, Appendix H. https://www.pacificorp.com/content/dam/pcorp/documents/en/pacificorp/energy/integrated-resource-plan/2023-irp/2023_IRP_Volume_II_Final_5-31-23.pdf

3. Stochastic Modeling Practices for Integrated Resource Planning, 2024. https://www.epri.com/research/products/000000003002030746
""",
                max_width=800,
            ),
        )
