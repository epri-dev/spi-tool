from spi_tool.models.regression import TimeseriesInput, TimeseriesPredictionModel
import os
import pandas as pd

CURRENT_DIRECTORY = os.path.dirname(__file__)


def test_get_data_with_duplicates():
    input_data = TimeseriesInput()
    filename = os.path.join(CURRENT_DIRECTORY, "data/historical_demand.csv")
    input_data.load_data(filename)
    assert len(input_data.input_df.index) == 366


def test_regression_using_lag_1():
    input_data = TimeseriesInput()
    input_data.load_example_data()
    model = TimeseriesPredictionModel(
        input_df=input_data.output(), regression_y_term="lag_1"
    )
    model.compute_sample_set()

    pd.testing.assert_frame_equal(
        model.output_df,
        pd.read_csv(
            os.path.join(CURRENT_DIRECTORY, "data/samples.csv"),
            index_col=0,
            parse_dates=True,
        ),
    )


def test_lognormal_regression_using_lag_1():
    input_data = TimeseriesInput()
    input_data.load_example_data()
    model = TimeseriesPredictionModel(
        input_df=input_data.output(),
        regression_y_term="lag_1",
        regression_kind="lognormal",
    )
    model.compute_sample_set()

    pd.testing.assert_frame_equal(
        model.output_df,
        pd.read_csv(
            os.path.join(CURRENT_DIRECTORY, "data/lognormal_samples.csv"),
            index_col=0,
            parse_dates=True,
        ),
    )
