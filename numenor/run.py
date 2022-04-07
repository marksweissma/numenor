from attr import define, field, Factory
import pandas as pd
from numenor import data, estimate, measure
from typing import *
from sklearn.base import clone

@variants.primary
def generate_columns(variant, predictions, name='prediction' **kwargs):
    variant = variant if variant else 'base'
    return getattr(generate_columns, variant)(predictions, name, **kwargs)

@generate_columns.variant('base')
def generate_columns_base(predictions, name='prediction' **kwargs):
    if 

def package_predictions(dataset,
                        transformer,
                        column_packager=None,
                        column_packager_kwargs=None,
                        **extras):
                        ):
    column_packager_kwargs = column_packager_kwargs if column_packager_kwargs else {}
    predictions = transformer.predict(dataset.anchor.features)
    columns = generate_columns(column_packager, predictions,
                               **column_packager_kwargs)
    df = pd.DataFrame(index=dataset.anchor.raw_index)
    for column, constant in extras.items():
        df[column] = constant
    return df


@define
class Run:
    transformer: estimate.Transformer
    dataset: data.Dataset = None
    metrics: List[measure.Metric] = Factory(list)
    evaluation_kwargs: Dict = Factory(dict)

    def train(self, dataset=None, transformer=None):
        dataset = dataset if dataset is not None else self.dataset
        transformer = transformer if transformer is not None else self.transformer
        anchor, index_location_generator = dataset.get_anchor_table_and_split_generator(
        )
        transformer.fit(data=anchor, cv=index_location_generator)
        return transformer

    def train_and_evaluate_test(self, dataset=None):
        dataset = dataset if dataset is not None else self.dataset
        train_dataset, test_dataset = dataset.split()
        self.train(train_dataset)
        return self.evaluate(test_dataset)

    def evaluate_cross_val(self, dataset=None):
        dataset = dataset if dataset is not None else self.dataset
        predictions = []
        for train_dataset, test_dataset in dataset.split_iterator():
            transformer = clone(self.transformer)
            transformer.fit(data=train_dataset.anchor)
            predictions.append(package_predictions(test_dataset, transformer))
        return dataset.anchor.append_columns(pd.concat(predictions),
                                             'prediction')

    def evaluate(self, dataset=None):
        dataset = self.dataset if dataset is None else dataset
