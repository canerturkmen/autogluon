# Forecasting Time-Series - Quick Start
:label:`sec_forecastingquick`

Via a simple `fit()` call, AutoGluon can train models produce forecasts for time series data. This tutorial demonstrates how to quickly use AutoGluon to produce forecasts of Covid-19 cases in a country given historical data.

To start, import AutoGluon's ForecastingPredictor and TabularDataset classes:

```{.python .input}
from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset
```

We load the time-series data to use for training from a CSV file into an AutoGluon Dataset object. This object is essentially equivalent to a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and the same methods can be applied to both.

```{.python .input}
train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
print(train_data[50:60])
```

Note that we loaded data from a CSV file stored in the cloud ([AWS s3 bucket](https://aws.amazon.com/s3/)), but you can you specify a local file-path instead if you have already downloaded the CSV file to your own machine (e.g., using [wget](https://www.gnu.org/software/wget/)). Our goal is to train models on this data that can forecast Covid-19 case counts in each country at future dates. This corresponds to a forecasting problem with many related individual time-series (one per country). Each row in the table `train_data` corresponds to one observation of one time-series at a particular time.

The dataset you use for `autogluon.forecasting` should usually contain three columns: a `date_column` with the time information (here "Date"), an `index_column` with a categorical ID specifying which (out of multiple) time-series is being observed (here "name", where each country corresponds to a different time-series in our example), and a `target_column` with the observed value of this particular time-series at this particular time (here "ConfirmedCases"). When forecasting future values of one particular time-series, AutoGluon may rely on historical observations of not only this time-series but also all of the other time-series in the dataset. You can use `NA` to represent missing observations in the data.

Now let's use AutoGluon to train some forecasting models. Below we instruct AutoGluon to fit models that can forecast up to 19 time-points into the future (`prediction_length`) and save them in the folder `save_path`. Because of the inherent uncertainty involved in this prediction problem, these models are trained to probabilistically forecast 3 different quantiles of the "ConfirmedCases" distribution: the central 0.5 quantile (median), a low 0.1 quantile, and a high 0.9 quantile. The first of these can be used as our forecasted value, while the latter two can be used as a prediction interval for this value (we are 80% confident the true value lies within this interval).

```{.python .input}
save_path = "agModels-covidforecast"

predictor = ForecastingPredictor(path=save_path).fit(train_data, prediction_length=19,
            index_column="name", target_column="ConfirmedCases", time_column="Date",
                                                         quantiles=[0.1, 0.5, 0.9],
                                                         presets="low_quality"  # last  argument is just for quick demo here, omit it in real applications!
                                                    )
```

**Note:** We use `presets = low_quality` above to ensure this example runs quickly, but this is NOT a good setting!  To obtain good performance in real applications you should either delete this argument or set `presets` to be one of: `"high_quality", "good_quality", "medium_quality"`. Higher quality presets will generally produce superior forecasting accuracy but take longer to train and may produce less efficient models (`low_quality` is intended just for quickly verifying that AutoGluon can be run on your data).

Now let's load some more recent test data to examine the forecasting performance of our trained models:

```{.python .input}
test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
```

The below code is unnecessary here, but is just included to demonstrate how to reload a trained Predictor object from file (for example in a new Python session):

```{.python .input}
predictor = ForecastingPredictor.load(save_path)
```

We can view the test performance of each model AutoGluon has trained via the `leaderboard()` function, where higher scores correspond to better predictive performance:

```{.python .input}
predictor.leaderboard(test_data)
```

Omit the `test_data` argument above to display performance scores on the validation data instead. By default, AutoGluon will score probabilistic forecasts of multiple time-series via the [weighted quantile loss](https://docs.aws.amazon.com/forecast/latest/dg/metrics.html#metrics-wQL), but you can specify a different `eval_metric` in `fit()` to instruct AutoGluon to optimize for a different evaluation metric instead (eg. `eval_metric="MAPE")`. For more details about the individual time-series models that AutoGluon can train, you can view the [GluonTS documentation](https://ts.gluon.ai/) or the source code folder `autogluon/forecasting/models/`.

We can also make forecasts further into the future based on the most recent data. When we call `predict()`, AutoGluon automatically forecasts with the model that had the best validation performance during training. The predictions returned by `predict()` form a dictionary whose keys index each time series (in this example, country) and whose values are DataFrames containing quantile forecasts for each time series (in this example, predicted quantiles of the case counts in each country at future subsequent dates to those observed in the test_data).

```{.python .input}
predictions = predictor.predict(test_data)
print(predictions['Afghanistan_'])  # quantile forecasts for the Afghanistan time-series
```

Instead of forecasting with the model that had the best validation score, you can instead specify which model to use for prediction, as well as that AutoGluon should only predict  certain time-series of interest:

```{.python .input}
model_touse = "MQCNN"
time_series_to_predict = ["Germany_", "Zimbabwe_"]
predictions = predictor.predict(test_data, model=model_touse, time_series_to_predict=time_series_to_predict)
```

Finally we can also print a summary of what happened during fit():

```{.python .input}
predictor.fit_summary()
```