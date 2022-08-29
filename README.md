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

### no-config

#### Boilerplate

```python
from sklearn import datasets
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, average_precision_score
from xgboost import XGBClassifier, XGBRegressor

from matplotlib import pyplot as plt
import variants
import pandas as pd

import structlog
from functools import lru_cache

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
