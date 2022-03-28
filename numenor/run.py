import attr
import pandas as pd
from numenor import data, estimate, measure
from typing import *
from sklearn.base import clone


def package_predictions(dataset, transformer):
    return pd.Series(transformer.predict(dataset.anchor.features),
                     index=dataset.anchor.raw_index)


@attr.s(auto_attribs=True)
class Run:
    transformer: estimate.Transformer
    dataset: data.Dataset = None
    metrics: List[measure.Metric] = attr.Factory(list)
    evaluation_kwargs: Dict = attr.Factory(dict)

    def train(self, dataset=None, transformer=None):
        dataset = self.dataset if dataset is None else dataset
        transformer = self.transformer if transformer is None else transformer
        anchor, index_location_generator = dataset.get_anchor_table_and_split_generator(
        )
        transformer.fit(data=anchor, cv=index_location_generator)
        return transformer

    def train_and_evaluate_test(self, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        train_dataset, test_dataset = dataset.split()
        self.train(train_dataset)
        return self.evaluate(test_dataset)

    def evaluate_cross_val(self, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        predictions = []
        for train_dataset, test_dataset in dataset.split_iterator():
            transformer = clone(self.transformer)
            transformer.fit(data=train_dataset.anchor)
            predictions.append(package_predictions(test_dataset, transformer))
        return dataset.anchor.append_columns(pd.concat(predictions),
                                             'prediction')

    def evaluate(self, dataset=None):
        dataset = self.dataset if dataset is None else dataset
