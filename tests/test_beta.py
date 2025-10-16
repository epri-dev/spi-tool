import pandas as pd
import pytest
from spi_tool.models.beta import TimeseriesScenarioInput
import os

CURRENT_DIRECTORY = os.path.dirname(__file__)


def test_get_data_valid_file():
    scenario_input = TimeseriesScenarioInput()
    filename = os.path.join(CURRENT_DIRECTORY, "data/valid_carbon_prices.csv")
    df = scenario_input.get_data(filename)
    assert not df.empty
    assert "date" in df.index.name
    assert scenario_input.unit == "2022 $/MTCO2e"


def test_get_data_with_duplicates():
    scenario_input = TimeseriesScenarioInput()
    filename = os.path.join(CURRENT_DIRECTORY, "data/duplicated_dates.csv")
    df = scenario_input.get_data(filename)
    assert scenario_input.has_warning
    assert "Duplicated dates found" in scenario_input.warning_message


def test_get_data_invalid_file():
    scenario_input = TimeseriesScenarioInput()
    filename = "tests/data/invalid_file.csv"
    with pytest.raises(Exception):
        scenario_input.get_data(filename)
