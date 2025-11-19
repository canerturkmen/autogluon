import logging
from typing import Any, Optional, Union

import pandas as pd

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class Chronos2Model(AbstractTimeSeriesModel):
    """Chronos-2 forecasting model.

    This model wraps the Chronos-2 pipeline from the `chronos-forecasting` library.
    It supports both zero-shot forecasting and fine-tuning.
    """

    _supports_known_covariates = True
    _supports_past_covariates = True

    default_model_path = "amazon/chronos-2"

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        hyperparameters = hyperparameters if hyperparameters is not None else {}

        # Handle model path
        self.model_path = hyperparameters.get("model_path", self.default_model_path)

        name = name if name is not None else "Chronos2"

        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )

        self._pipeline = None

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        **kwargs,
    ) -> None:
        from chronos.chronos2.pipeline import Chronos2Pipeline

        self._check_fit_params()

        # Load the pipeline
        # TODO: Handle device placement more intelligently if needed,
        # but Chronos2Pipeline.from_pretrained handles it reasonably well.
        self._pipeline = Chronos2Pipeline.from_pretrained(self.model_path)

        # Check if fine-tuning is requested
        if self._get_model_params().get("fine_tune", False):
            logger.info(f"Fine-tuning {self.name}...")

            # Prepare inputs for fit
            # We need to convert TimeSeriesDataFrame to the format expected by fit
            # But wait, fit takes the same input format as predict?
            # The docstring says: "The allowed formats of inputs are the same as `Chronos2Pipeline.predict()`."
            # However, predict_df is a wrapper around predict that handles dataframe conversion.
            # fit does NOT have a fit_df equivalent in the pipeline class I saw.
            # So we might need to use the internal conversion logic or just pass the dataframe if supported?
            # Looking at pipeline.py, fit takes `inputs`. predict_df calls `convert_df_input_to_list_of_dicts_input`.
            # We should probably use that helper.

            from chronos.df_utils import convert_df_input_to_list_of_dicts_input

            # Prepare training data
            train_inputs, _, _ = convert_df_input_to_list_of_dicts_input(
                df=train_data.reset_index(),
                id_column="item_id",
                timestamp_column="timestamp",
                target_columns="target",  # TODO: Handle multivariate if needed, but AG TS is mostly univariate target for now
                prediction_length=self.prediction_length,
            )

            val_inputs = None
            if val_data is not None:
                val_inputs, _, _ = convert_df_input_to_list_of_dicts_input(
                    df=val_data.reset_index(),
                    id_column="item_id",
                    timestamp_column="timestamp",
                    target_columns="target",
                    prediction_length=self.prediction_length,
                )

            # Get fine-tuning parameters
            fine_tune_kwargs = {
                k: v
                for k, v in self._get_model_params().items()
                if k in ["learning_rate", "num_steps", "batch_size", "context_length", "min_past"]
            }

            # Perform fine-tuning
            # Note: fit returns a NEW pipeline
            self._pipeline = self._pipeline.fit(
                inputs=train_inputs,
                prediction_length=self.prediction_length,
                validation_inputs=val_inputs,
                output_dir=self.path,
                **fine_tune_kwargs,
            )
        else:
            # If not fine-tuning, we just use the pretrained model.
            # We might want to save it to the local path to ensure persistence?
            # Or just rely on loading it again from the model_path.
            pass

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if self._pipeline is None:
            from chronos.chronos2.pipeline import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(self.model_path)

        # Pad short series if necessary (Chronos2 requires at least 3 observations)
        # We prepend the first observation with shifted timestamps
        min_len = 3
        counts = data.num_timesteps_per_item()
        short_items = counts[counts < min_len].index

        if not short_items.empty:
            # We need to modify data, so copy it
            data = data.copy()
            freq_offset = pd.tseries.frequencies.to_offset(self.freq)

            new_rows = []
            for item_id in short_items:
                item_df = data.loc[item_id]
                curr_len = len(item_df)
                needed = min_len - curr_len

                first_row = item_df.iloc[0]
                first_ts = item_df.index.get_level_values("timestamp")[0]

                for i in range(needed):
                    new_ts = first_ts - freq_offset * (needed - i)
                    new_row = first_row.copy()
                    # Create a DataFrame with MultiIndex to match structure
                    new_df = pd.DataFrame([new_row])
                    new_df["item_id"] = item_id
                    new_df["timestamp"] = new_ts
                    new_df = new_df.set_index(["item_id", "timestamp"])
                    new_rows.append(new_df)

            if new_rows:
                padding_df = pd.concat(new_rows)
                data = pd.concat([padding_df, data]).sort_index()

        # Prepare future_df for known covariates
        future_df = None
        if known_covariates is not None:
            # Merge past known covariates into df
            # We use a left join to keep only the past part corresponding to data
            # Only join columns that are not already in data to avoid overlap
            cols_to_join = [c for c in known_covariates.columns if c not in data.columns]
            if cols_to_join:
                df = data.join(known_covariates[cols_to_join], how="left")
            else:
                df = data

            # Extract future part for future_df
            # We assume future_df should contain rows NOT in data
            # and limit to prediction_length
            future_df = known_covariates[~known_covariates.index.isin(data.index)]
            future_df = future_df.groupby(level="item_id").head(self.prediction_length)

            # Ensure future_df does not contain target
            if "target" in future_df.columns:
                future_df = future_df.drop(columns=["target"])

            future_df = future_df.reset_index()
            future_df = pd.DataFrame(future_df)
        else:
            df = data

        # Ensure df is a plain pandas DataFrame
        df = df.reset_index()
        df = pd.DataFrame(df)

        forecast_df = self._pipeline.predict_df(
            df=df,
            future_df=future_df,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            batch_size=self._get_model_params().get("batch_size", 256),
        )

        # Convert back to TimeSeriesDataFrame
        # The output has columns: item_id, timestamp, target, predictions (mean), q1, q2...
        # We need to conform to AG's expected output format

        # Rename columns to match AG expectation
        rename_map = {"predictions": "mean"}
        # Quantiles in AG are usually strings like "0.1", "0.5", etc.
        # Chronos predict_df returns columns like 0.1, 0.5 (floats) or strings?
        # Let's assume they might be floats or strings depending on implementation,
        # but AG expects column names to be strings of the quantiles.

        # We need to set index back to (item_id, timestamp)
        forecast_df = forecast_df.set_index(["item_id", "timestamp"])

        # Select only the relevant columns (mean and quantiles)
        required_cols = ["mean"] + [str(q) for q in self.quantile_levels]

        # Ensure column names are strings for quantiles
        forecast_df.columns = [str(c) if not isinstance(c, str) else c for c in forecast_df.columns]

        # Rename 'predictions' to 'mean' if needed (Chronos might return 'predictions' as point forecast)
        if "predictions" in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={"predictions": "mean"})

        return TimeSeriesDataFrame(forecast_df[required_cols])

    def _get_model_params(self) -> dict:
        return self._get_default_hyperparameters() | self._hyperparameters

    def _get_default_hyperparameters(self) -> dict:
        return {
            "model_path": self.default_model_path,
            "batch_size": 256,
            "fine_tune": False,
            "learning_rate": 1e-5,
            "num_steps": 1000,
        }

    def _more_tags(self) -> dict[str, Any]:
        do_fine_tune = self._get_model_params().get("fine_tune", False)
        return {
            "allow_nan": True,
            "can_use_train_data": do_fine_tune,
            "can_use_val_data": do_fine_tune,
        }
