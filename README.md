# numenor

This project is in active development - quick start & pip installable wheel available in the coming weeks!

numenor is an SDK for managing and improving ML training workflows.
Components are designed to be extensible and composeable enabling independent
integration, testing and development - not a DSL. Getting the most out of the framwork
does require buying into its basic tenants and structures, but does not block usage such as

1. Independent configurable column and row based data management
2. The wrapper around sklearn's pipeline enabling applying transforms to validation data for early stopping (xgboost) or row-based resampling(i.e SMOTE).
3. A `Schema` transformer enabling no additional code to stand up a FastAPI app
with an automated pydantic Model for validating the request

## Patterns

### Dispatch by type
One of the core tenants is separation of `IO <> Configuration <> Execution`
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

assert first_config == Config(an_arg='a', another_arg='b')
assert second_config == Config(an_arg='a', another_arg='b')

```
While we could offload the dict unpacking
to a classmethod, we then need to maintain the logic to handle the types and offloading.
With `as_factory` we get the expected behavior of "if it's the object we want, pass it through, if not create it" this is most commonly used with `attrs` `fields` via `converters`
What this pattern unblocks is the ability to pass nested configuration down through
dependencies while also enabling the same interfaces to accept the objects themselves without
writing or maintaing the code.

### Dispatch by value
Dispatch by value enables users to drive workflows from configuration.
In addition, a user can interact
with core code and modules, extend them without ever leaving their development area. Dispatch by value is built on the `variants` package. A lightweight wrapper that allows functions to maintain _variants_ that can be invoked explicitly as an attribute.

```python
import variants

@variant.primary
def my_function(variant: str ='base', *args, **kwargs) -> str:
    return getattr(function, variant)(*args, **kwargs)

@my_function.variant('base')
def my_function(*args, **kwargs) -> str:
    return 'base'

```
the variant can be specified by configuration and if a user wants to extend functionality they can do so explicitly.

... elsewhere ..

```python
@my_function.variant('my_experiment')
def my_function__my_experiment(*args, **kwargs) -> str:
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
from functools import lru_cache

import pandas as pd
import structlog
import variants
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier, XGBRegressor

from numenor import data, estimate
from numenor.pipeline import make_pipeline, package_params

LOG = structlog.get_logger()
RANDOM_STATE = 42


@lru_cache(None)
def load_sklearn_dataset(handle):
    dataset = handle()
    X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    y = pd.Series(dataset["target"], name="label")
    return X, y

```

we'll go through the imports and the application as we go along

`load_sklearn_dataset` is a fixture generator for sklearn datasets and converting them to a pandas df


Here we're going to use the function `regression_example` as our entrypoint.
We'll start with a `base` case and build out variants to show components of numenor

For regression we're going to usethe `diabetes` dataset.
For the `base` case we fit a L1 regularized linear model with scaled features.


```python
@variants.primary
def regression_example(variant=None, **kwargs):
    variant = variant if variant else "base"
    return getattr(regression_example, variant)(**kwargs)


@regression_example.variant("base")
def regression_example_base(plot=True, **kwargs):
    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    model = make_pipeline(RobustScaler(), Lasso())
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    LOG.msg(
        "regression performance",
        r2_score=r2_score(y_test, predictions),
        mean_absolute_error=mean_absolute_error(y_test, predictions),
    )
    if plot:
        plt.scatter(y_test, predictions)
        plt.show()

```

```bash
>>> 2022-08-29 07:54.31 [info] regression performance mean_absolute_error=41.417137883143404 r2_score=0.49930007687526934
```

Next we'll wrap this the model in `numenor.estimate.Transformer` in this
example the `Transformer` doesn't give us much, but it provides
instrumentation of the pipeline and an interface for callbacks
on the object itself. The `Transformer` also helps
if we want to deploy this predictor in a fast api app - more on that in that in
the app example. While it doesn't give us much, it also shouldn't cost anything
it has the same interfaces and support as the sklearn model - setting params
just requires adding the `executor__` prefix to the specification.

```python
@regression_example.variant("transformer")
def regression_example_transformer(plot=True, **kwargs):
    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    model = estimate.Transformer(make_pipeline(RobustScaler(), Lasso()))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    LOG.msg(
        "regression performance",
        r2_score=r2_score(y_test, predictions),
        mean_absolute_error=mean_absolute_error(y_test, predictions),
    )
    if plot:
        plt.scatter(y_test, predictions)
        plt.show()

```

While xgboost provides an interface for sklearn - the way it handles
early stopping prevents transforms from being applied to early stopping and
metric evaluation datasets. A core "feature"  of sklearn pipeline's has always
been the inability to mutate rows - whether that's outlier removal
or upsamling in the pipeline. The numenor helpers `PipelineXGBResample`
and `make_pipeline` in `numenor.pipeline` solve this issue.
Transforms before the terminal estimator in the pipeline will be applied
to the `eval_set`s. In addition, a _sampler_ with the interface specified
by `imbalanced-learn` can be specified - it is called during fit between
the penultimate and ultimate steps of the pipeline.

```bash
>>> 2022-08-29 07:56.23 [info] regression performance mean_absolute_error=41.417137883143404 r2_score=0.49930007687526934
```

```python
@regression_example.variant("pipeline_with_early_stopping")
def regression_example_pipeline_with_early_stopping(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    # Using numnenor's pipeline wrapper enables support for preprocessing transformers
    # and samplers before fitting without hacks / boilerplate / duplication
    model = make_pipeline(
        RobustScaler(),
        KMeans(n_clusters=3),
        XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.3),
    )

    # split training data into "fitting" and "validating" for early stopping
    X_train_fit, X_train_validate, X_train_fit, y_train_validate = train_test_split(
        X_train, y_train
    )

    early_stopping_params = {
        "xgbregressor__early_stopping_rounds": 5,
        "xgbregressor__eval_metric": ["rmse"],
        "xgbregressor__eval_set": [(X_train_validate, y_train_validate)],
    }

    model.fit(X_train, y_train, **early_stopping_params)

    predictions = model.predict(X_test)

    LOG.msg(
        "regression performance",
        r2_score=r2_score(y_test, predictions),
        mean_absolute_error=mean_absolute_error(y_test, predictions),
    )

```

#### data

Numenor aims to enable, automate and better support data management.
There are two abstraction to support this goal. Having two instead of one
was a tough choice and is one of the few places numenor may have skewed towards "complex".
The rational was to provide _self describing_ objects that can generate their children
without additional information and **separate** column management from row management.

1. `numenor.data.Dataset` owns `nuemnor.data.Data` objects and maintains `numenor.data.Split` objects which control _row_ based management. `Dataset` objects also own child configuration (more on this later - i.e first split by time (9 months train and 3 months evaluation - then within train use stratified sampling to optimize parameters)

2. `numenor.data.Data` owns the direct reference to the underling data (DataFrame, array etc.) and manages column based selection (feautres, target, metadata, prediction, index)



More on data abstractions and getting leverage out of workflows in the User Guide,
but back to examples we'll start with `numenor.data.Data`.

The focus of the `Data` object is to manage the columns of the dataset.
To achieve this components
1. features
2. target
3. prediction
4. metadata

are managed by `numnor.data.Column` objects which "match" keys of the underlying data (i.e. column names of `pandas.DataFrame`). For automation purposes a `None` or empty
`[]` specificaton of `features` is assumed to select all not specified by another column selector

`Data` can be instantied directly (next example) or from their components (this example)


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
Here the keys and shapes  associated with features, target and metadata are inferred
from the inputs of the classmethod

Alternatively you can explicitly specify what keys belong to which groups of a table

```python
@regression_example.variant("data__column_name")
def regression_example_data__column_name(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    metadata = pd.Series(["my_metadata"] * len(X), name="metadata")
    table = pd.concat([X, y, metadata], axis=1)

    full_data = data.Data(table, target_column="label", metadata_column="metadata")

    model = make_pipeline(RobustScaler(), Lasso())
    model.fit(full_data.features, full_data.target)
    LOG.msg("All columns available", columns=full_data.columns)
    LOG.msg("Features", features=full_data.feature_keys)
    LOG.msg(
        "metadata",
        metadata={
            "name": full_data.metadata.name,
            "type": full_data.metadata.__class__,
        },
    )
```

Data objects maintain their columns (key selector) and are decoupled
from row-wise configuration which is defined through _splitting_

`numenor.data.Split` objets wrap `sklearn` `Splitter`s and `KFolder`s
they enable declarative definition of stratification, groups and
explicit setting of chained (children) datasets. For example,

1. split train, test based on time (i.e. 6 months train, 2 months test)
2. split train into fitting and validation for hyperparameter optimzation based on stratification of the target class. If children configurations are not specified
children inherit the configuration their parent. Children can be set up as levels
(i.e. all children will have this config, or as branches train has conf1, test has conf2)


Here we will not mutate the config for the children and illustrate defaulting
to a `ShuffleSplit`

```python
@regression_example.variant("dataset")
def regression_example_dataset(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)

    full_dataset = data.Dataset.from_method("from_components", features=X, target=y)

    model = make_pipeline(RobustScaler(), Lasso())

    train_dataset, test_dataset = full_dataset.split()

    model.fit(train_dataset.features, train_dataset.target)

    predictions = model.predict(test_dataset.features)

    LOG.msg(
        "regression performance",
        r2_score=r2_score(test_dataset.target, predictions),
        mean_absolute_error=mean_absolute_error(test_dataset.target, predictions),
    )


```

Automated splitting enables automated early stopping with a controllable shape

```python
@regression_example.variant("dataset__early_stopping")
def regression_example_dataset__early_stopping(**kwargs):

    X, y = load_sklearn_dataset(datasets.load_diabetes)

    full_dataset = data.Dataset.from_method("from_components", features=X, target=y)

    model = make_pipeline(
        RobustScaler(),
        KMeans(n_clusters=3),
        XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.3),
    )

    train_dataset, test_dataset = full_dataset.split()
    train_dataset_fitting, train_dataset_validating = train_dataset.split()

    fit_params = package_params(
        model,
        {
            "early_stopping_rounds": 5,
            "eval_metric": ["rmse"],
            "eval_set": [
                (train_dataset_validating.features, train_dataset_validating.target)
            ],
        },
    )

    model.fit(train_dataset.features, train_dataset.target, **fit_params)

    predictions = model.predict(test_dataset.features)

    LOG.msg(
        "dataset lengths",
        full=len(full_dataset),
        train=len(train_dataset),
        train_fitting=len(train_dataset_fitting),
        train_validating=len(train_dataset_validating),
    )

    LOG.msg(
        "regression performance",
        r2_score=r2_score(test_dataset.target, predictions),
        mean_absolute_error=mean_absolute_error(test_dataset.target, predictions),
    )
```
