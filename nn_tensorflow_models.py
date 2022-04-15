import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def create_model_1(img_height: int, img_width: int, classes: list, metrics: list[keras.metrics.Metric | str]):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(classes))
    ],
        name="m1_tutorial_baseline")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=metrics)
    return model


def create_model_2(hyper_params, img_height: int, img_width: int, classes: list,
                   metrics: list[keras.metrics.Metric | str]):
    hp_rot = hyper_params.Float("rot_factor", min_value=0, max_value=0.5, step=0.1)
    hp_zoom = hyper_params.Float("zoom_factor", min_value=-0.5, max_value=0, step=0.1)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip(input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(hp_rot),
            layers.RandomZoom((0, hp_zoom)),
        ]
    )
    hp_conv2D_1_filters = hyper_params.Int("conv2D_1_filters", min_value=8, max_value=24, step=8)
    hp_conv2D_2_filters = hyper_params.Int("conv2D_2_filters", min_value=8, max_value=48, step=8)
    hp_conv2D_3_filters = hyper_params.Int("conv2D_3_filters", min_value=8, max_value=96, step=8)
    hp_dropout_rate = hyper_params.Float("dropout_rate", min_value=0, max_value=0.5, step=0.1)
    hp_dense_units = hyper_params.Int("dense_units", min_value=32, max_value=128, step=32)
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(hp_conv2D_1_filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_2_filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_3_filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(hp_dropout_rate),
        layers.Flatten(),
        layers.Dense(hp_dense_units, activation='relu'),
        layers.Dense(len(classes))
    ],
        name="m2_auto_tune_on_m1")
    hp_learning_rate = hyper_params.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=metrics)
    return model


def create_model_3(img_height: int, img_width: int, classes: list, metrics: list[keras.metrics.Metric | str]):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip(input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.5),
            layers.RandomZoom((0, -0.5)),
        ]
    )
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(4, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(12, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(20, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(24, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(28, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(classes))
    ],
        name="m3_many_layers")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=metrics)
    return model


def create_model_4(img_height: int, img_width: int, classes: list, metrics: list[keras.metrics.Metric | str]):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip(input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.5),
            layers.RandomZoom((0, -0.5)),
        ]
    )
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(40, activation='relu'),
        layers.Dense(len(classes))
    ],
        name="m4_large_filters")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=metrics)
    return model


def create_model_5(img_height: int, img_width: int, classes: list,
                   metrics: list[keras.metrics.Metric | str]):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip(input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.5),
            layers.RandomZoom((0, -0.1)),
        ]
    )
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(24, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(80, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(96, activation='relu'),
        layers.Dense(len(classes))
    ],
        name="m5_best_m2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=metrics)
    return model


def create_model_6(hyper_params, img_height: int, img_width: int, classes: list,
                   metrics: list[keras.metrics.Metric | str]):
    hp_rot = hyper_params.Float("rot_factor", min_value=0, max_value=0.5, step=0.1)
    hp_zoom = hyper_params.Float("zoom_factor", min_value=-0.5, max_value=0, step=0.1)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip(input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(hp_rot),
            layers.RandomZoom((0, hp_zoom)),
        ]
    )
    hp_conv2D_filters = [
        hyper_params.Int("conv2D_1_filters", min_value=4, max_value=8, step=4),
        hyper_params.Int("conv2D_2_filters", min_value=4, max_value=16, step=4),
        hyper_params.Int("conv2D_3_filters", min_value=4, max_value=32, step=4),
        hyper_params.Int("conv2D_4_filters", min_value=4, max_value=64, step=4),
        hyper_params.Int("conv2D_5_filters", min_value=4, max_value=128, step=4),
        hyper_params.Int("conv2D_6_filters", min_value=4, max_value=256, step=4),
        hyper_params.Int("conv2D_7_filters", min_value=4, max_value=512, step=4),
        hyper_params.Int("conv2D_8_filters", min_value=4, max_value=1024, step=4)
    ]
    hp_dropout_rate = hyper_params.Float("dropout_rate", min_value=0, max_value=0.5, step=0.1)
    hp_dense_units = hyper_params.Int("dense_units", min_value=32, max_value=128, step=32)
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(hp_conv2D_filters[0], 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_filters[1], 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_filters[2], 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_filters[3], 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_filters[4], 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_filters[5], 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_filters[6], 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv2D_filters[7], 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(hp_dropout_rate),
        layers.Flatten(),
        layers.Dense(hp_dense_units, activation='relu'),
        layers.Dense(len(classes))
    ],
        name="m6_auto_tune_on_m3")
    hp_learning_rate = hyper_params.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=metrics)
    return model


def create_model_7(img_height: int, img_width: int, classes: list,
                   metrics: list[keras.metrics.Metric | str]):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip(input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.5),
            layers.RandomZoom((0, -0.1)),
        ]
    )
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(28, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(88, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(120, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(316, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(100, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(classes))
    ],
        name="m7_best_m6")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=metrics)
    return model
