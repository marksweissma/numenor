# numenor

This project is in active development - quick start & pip installable wheel available in the coming weeks!

numenor is an SDK for managing and improving ML training workflows.
Components are designed to be extensible and composeable enabling independent
integration, testing and development - not a DSL. Getting the most out of the framwork
does require buying into its basic tenants and structures, but does not block usage such as

1. Independent configurable column and row based data management 
2. The wrapper around sklearn's pipeline enabling applying transforms to validation data for early stopping (xgboost) or row-based resampling(i.e SMOTE).
3. A `Schema` transformer enabling no additional code to stand up a FastAPI app
with an automated pydantic Model for valiating the request

### patterns

1. One of the core tenants is separation of `IO <> Configuration <> Execution`
as a result numenor attempts to enable configuration with a focus on development speed.
This pattern appears often through the use of the `as_factory` utility. `as_factory` uses type inference to decide whether to pass the argument through at runtime or pass the argument(s) 
to a _handler_ and return the result of the handler. What does that look like in practice
```python
@define
class Config:
    an_arg: str
    another_arg: str


def load_config(config: Union[Dict[str, str], Config]) -> Config:
    return as_factory(Config)(config)

first_config = load_config({'an_arg': 'a', 'another_arg': 'b'})
second_config = load_config(Config('a', 'b'))

```
`first_config` and `second_config` are equivalent. While we could offload the dict unpacking
to a classmethod, we then need to maintain the logic to handle the types and offloading.
With `as_factory` we get the expected behavior of "if it's the object we want, pass it through, if not create it" this is most commonly used with `attrs` `fields` via `converters` 
What this pattern unblocks is the ability to pass nested configuration down through
dependencies while also enabling the same interfaces to accept the objects themselves without 
writing or maintaing the code.


2. Dispatch by value. 
 Dispatch by value enables a user to interact
with core code and modules. Extend them without ever leaving their development area. Dispatch by value is built on the `variants` package. A lightweight wrapper that allows functions to maintain _variants_ that can be invoked explicitly as an attribute.

```python
import variants

@variant.primary
def function(variant='base', *args, **kwargs):
    return getattr(function, variant)(*args, **kwargs)

```
the variant can be specified by configuration and if a user wants to extend functionality they can do so explicitly.

... elsewhere ..

```python
@function.variant('my_experiment')
def function(*args, **kwargs):
    return 'my_experiment'

assert 'my_experiment' == function.my_experiment()
assert 'my_experiment' == function('my_experiment')

```

## Examples

### without-config

In this example we'll walk through incorporating elements of a typical sklearn 
workflow with components from numenor

#### Boilerplate

```python
from sklearn import datasets
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, average_precision_score
from xgboost import XGBClassifier, XGBRegressor

from functools import lru_cache
import pandas as pd
import structlog
import variants

from numenor import data, estimate
from numenor.pipeline import (make_pipeline_xgb_resample,
                              package_terminal_estimator_params)

LOG = structlog.get_logger()
RANDOM_STATE = 42


@lru_cache(None)
def load_sklearn_dataset(handle):
    dataset = handle()
    X = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
    y = pd.Series(dataset['target'], name='label')
    return X, y

@variants.primary
def regression_example(variant=None, **kwargs):
    variant = variant if variant else 'base'
    return getattr(regression_example, variant)(**kwargs)

```

we'll go through the imports and the application as we go along

`load_sklearn_dataset` is a fixture generator for sklearn datasets and converting them to a pandas df


Here we're going to use the function `regression_example` as our entrypoint.
We'll start with a `base` case and build out variants to show components of numenor

For regression we're going to usethe `diabetes` dataset.
In this example we fit a L1 regularized linear model with scaled features.


```python
@regression_example.variant('base')
def regression_example():
    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE)

    model = make_pipeline(RobustScaler(), Lasso())
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    LOG.msg('regression performance',
            r2_score=r2_score(y_test, predictions),
            mean_absolute_error=mean_absolute_error(y_test, predictions))
```

```bash
>>> 2022-08-29 09:54.31 [info] regression performance mean_absolute_error=41.417137883143404 r2_score=0.49930007687526934
```

Next we'll wrap this the model in `numenor.estimate.Transformer` in this
example the `Transformer` doesn't give us much, but it provides
instrumentation of the pipeline and an intercace for callbacks
on the object itself

```python
@regression_example.variant('transformer')
def regression_example():
    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE)

    model = estimate.Transformer(make_pipeline(RobustScaler(), Lasso()))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    LOG.msg('regression performance',
            r2_score=r2_score(y_test, predictions),
            mean_absolute_error=mean_absolute_error(y_test, predictions))
    plt.scatter(y_test, predictions)
    plt.show()
```

```bash
>>> 2022-08-29 09:54.31 [info] regression performance mean_absolute_error=41.417137883143404 r2_score=0.49930007687526934
```

```python
@regression_example.variant('pipeline_with_early_stopping')
def regression_example():

    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE)

    # Yes, scaling -> trees isn't "correct" but this is for illustration purposes
    # Using numnenor's pipeline wrapper enables support for preprocessing transformers
    # and samplers before fitting without hacks / boilerplate / duplication
    model = make_pipeline_xgb_resample(
        RobustScaler(),
        XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=.3))

    # split training data into "fitting" and "validating" for early stopping
    X_train_fit, X_train_validate, X_train_fit, y_train_validate = train_test_split(
        X_train, y_train)

    early_stopping_params = {
        'xgbregressor__early_stopping_rounds': 5,
        'xgbregressor__eval_metric': ['rmse'],
        'xgbregressor__eval_set': [(X_train_validate, y_train_validate)]
    }

    model.fit(X_train, y_train, **early_stopping_params)

    predictions = model.predict(X_test)

    LOG.msg('regression performance',
            r2_score=r2_score(y_test, predictions),
            mean_absolute_error=mean_absolute_error(y_test, predictions))
```


```python
@regression_example.variant('data__column')
def regression_example():

    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE)

    metadata = pd.Series(['metadata'] * len(X), name='metadata')
    full_data = data.Data.from_components(features=X,
                                          target=y,
                                          metadata=metadata)

    model = make_pipeline(RobustScaler(), Lasso())
    model.fit(full_data.features, full_data.target)
    LOG.msg('All columns available', columns=full_data.columns)
    LOG.msg('Features', features=full_data.feature_keys)
    LOG.msg('metadata',
            metadata={
                'name': full_data.metadata.name,
                'type': full_data.metadata.__class__
            })


```

```python

@regression_example.variant('data__column_name')
def regression_example():

    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE)

    metadata = pd.Series(['my_metadata'] * len(X), name='metadata')
    table = pd.concat([X, y, metadata], axis=1)

    full_data = data.Data(table,
                          target_column='label',
                          metadata_column='metadata')

    model = make_pipeline(RobustScaler(), Lasso())
    model.fit(full_data.features, full_data.target)
    LOG.msg('All columns available', columns=full_data.columns)
    LOG.msg('Features', features=full_data.feature_keys)
    LOG.msg('metadata',
            metadata={
                'name': full_data.metadata.name,
                'type': full_data.metadata.__class__
            })

```


```python

@regression_example.variant('dataset')
def regression_example():

    X, y = load_sklearn_dataset(datasets.load_diabetes)

    full_dataset = data.Dataset.from_method('from_components',
                                            features=X,
                                            target=y)

    model = make_pipeline(RobustScaler(), Lasso())

    train_dataset, test_dataset = full_dataset.split()

    model.fit(train_dataset.features, train_dataset.target)

    predictions = model.predict(test_dataset.features)

    LOG.msg('regression performance',
            r2_score=r2_score(test_dataset.target, predictions),
            mean_absolute_error=mean_absolute_error(test_dataset.target,
                                                    predictions))
```

```python
@regression_example.variant('dataset__early_stopping')
def regression_example():

    X, y = load_sklearn_dataset(datasets.load_diabetes)

    full_dataset = data.Dataset.from_method('from_components',
                                            features=X,
                                            target=y)

    model = make_pipeline_xgb_resample(
        RobustScaler(),
        XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=.3))

    train_dataset, test_dataset = full_dataset.split()
    train_dataset_fitting, train_dataset_validating = train_dataset.split()

    fit_params = package_terminal_estimator_params(
        model, {
            'early_stopping_rounds':
            5,
            'eval_metric': ['rmse'],
            'eval_set': [(train_dataset_validating.features,
                          train_dataset_validating.target)]
        })

    model.fit(train_dataset.features, train_dataset.target, **fit_params)

    predictions = model.predict(test_dataset.features)

    LOG.msg('dataset lengths',
            full=len(full_dataset),
            train=len(train_dataset),
            train_fitting=len(train_dataset_fitting),
            train_validating=len(train_dataset_validating))

    LOG.msg('regression performance',
            r2_score=r2_score(test_dataset.target, predictions),
            mean_absolute_error=mean_absolute_error(test_dataset.target,
                                                    predictions))
```
