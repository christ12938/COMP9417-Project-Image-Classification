from typing import Callable

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras

from nn_tensorflow_dataset import *


def train_model(model_builder: Callable,
                img_height: int,
                img_width: int,
                classes: list,
                metrics: list[keras.metrics.Metric | str],
                dataset: Dataset,
                epochs: int):
    model = model_builder(img_height, img_width, classes, metrics)
    history = model.fit(
        dataset.train_set,
        validation_data=dataset.valid_set,
        epochs=epochs
    )
    return model, history


def tune_and_train_model(model_builder: Callable,
                         img_height: int,
                         img_width: int,
                         classes: list,
                         metrics: list[keras.metrics.Metric | str],
                         dataset: Dataset,
                         epochs: int,
                         tuning_dir: str,
                         model_id: int):
    if len(metrics) == 0:
        raise Exception("Tuning must specify a primary objective metric.")

    primary_metric = metrics[0]
    primary_val_metric = "val_" + (
        primary_metric.name if isinstance(primary_metric, keras.metrics.Metric) else primary_metric)
    tuner = kt.Hyperband(lambda hp: model_builder(hp, img_height, img_width, classes, metrics),
                         objective=kt.Objective(primary_val_metric, "max"),
                         max_epochs=epochs + 10,
                         factor=3,
                         directory=tuning_dir,
                         project_name=str(model_id))
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    tuner.search(dataset.train_set, validation_data=dataset.valid_set, epochs=epochs, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        dataset.train_set,
        validation_data=dataset.valid_set,
        epochs=epochs
    )
    return model, history


def train_model_with_class_weights(model_builder: Callable,
                                   img_height: int,
                                   img_width: int,
                                   classes: list,
                                   metrics: list[keras.metrics.Metric | str],
                                   dataset: Dataset,
                                   epochs: int):
    model = model_builder(img_height, img_width, classes, metrics)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7)
    history = model.fit(
        dataset.train_set,
        validation_data=dataset.valid_set,
        epochs=epochs,
        class_weight=dataset.class_weights,
        callbacks=[reduce_lr]
    )
    return model, history
