from typing import Callable

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras


def train_model(model_builder: Callable,
                img_height: int,
                img_width: int,
                classes: list,
                metrics: list[keras.metrics.Metric | str],
                train_dataset,
                validation_dataset,
                epochs: int):
    model = model_builder(img_height, img_width, classes, metrics)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs
    )
    return model, history


def tune_and_train_model(model_builder: Callable,
                         img_height: int,
                         img_width: int,
                         classes: list,
                         metrics: list[keras.metrics.Metric | str],
                         train_dataset,
                         validation_dataset,
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
    tuner.search(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs
    )
    val_metric_per_epoch = history.history[primary_val_metric]
    best_epoch = val_metric_per_epoch.index(max(val_metric_per_epoch)) + 1
    model = tuner.hypermodel.build(best_hps)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=best_epoch
    )
    return model, history
