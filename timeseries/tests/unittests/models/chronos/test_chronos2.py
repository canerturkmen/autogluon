import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.chronos import Chronos2Model

from ...common import (
    DATAFRAME_WITH_COVARIATES,
    DUMMY_TS_DATAFRAME,
    get_data_frame_with_item_index,
)


@pytest.fixture
def chronos2_model():
    return Chronos2Model(
        prediction_length=5,
        hyperparameters={"model_path": "amazon/chronos-2", "device": "cpu"},
    )


def test_chronos2_init(chronos2_model):
    assert chronos2_model.model_path == "amazon/chronos-2"
    assert chronos2_model.prediction_length == 5


def test_chronos2_fit_predict_basic(chronos2_model):
    data = DUMMY_TS_DATAFRAME
    chronos2_model.fit(train_data=data)
    predictions = chronos2_model.predict(data)

    assert isinstance(predictions, TimeSeriesDataFrame)
    assert len(predictions) == data.num_items * chronos2_model.prediction_length
    assert "mean" in predictions.columns
    assert all(str(q) in predictions.columns for q in chronos2_model.quantile_levels)


def test_chronos2_predict_with_known_covariates(chronos2_model):
    data = DATAFRAME_WITH_COVARIATES
    # Split into past and future for simulation
    past_data = data.slice_by_timestep(None, -5)
    # Known covariates should not contain target
    known_covariates = data.drop(columns=["target"])

    chronos2_model.fit(train_data=past_data)
    predictions = chronos2_model.predict(past_data, known_covariates=known_covariates)

    assert len(predictions) == past_data.num_items * chronos2_model.prediction_length
    assert not predictions.isna().any().any()


def test_chronos2_predict_with_future_covariates_splitting():
    # Test the logic where known_covariates are split into past (merged to df) and future (future_df)
    model = Chronos2Model(prediction_length=3, hyperparameters={"model_path": "amazon/chronos-2", "device": "cpu"})

    # Create data with covariates
    df = pd.DataFrame(
        {
            "item_id": ["A"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "target": np.random.rand(10),
            "cov1": np.random.rand(10),
        }
    )
    data = TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")

    # Train on first 7
    train_data = data.slice_by_timestep(None, 7)
    # Known covariates covers full range (past + future)
    known_covariates = data[["cov1"]]

    model.fit(train_data=train_data)
    predictions = model.predict(train_data, known_covariates=known_covariates)

    assert len(predictions) == 3  # 1 item * 3 steps
    assert not predictions.isna().any().any()


def test_chronos2_persistance(chronos2_model, tmp_path):
    data = DUMMY_TS_DATAFRAME
    chronos2_model.fit(train_data=data)
    chronos2_model.save(path=str(tmp_path))

    loaded_model = Chronos2Model.load(path=str(tmp_path))
    predictions = loaded_model.predict(data)

    assert len(predictions) == data.num_items * chronos2_model.prediction_length
