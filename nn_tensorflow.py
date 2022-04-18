import argparse
from pathlib import Path
from time import time_ns
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.utils import image_dataset_from_directory, img_to_array, load_img
from tensorflow_addons.metrics import F1Score

from nn_tensorflow_dataset import *
from nn_tensorflow_models import *
from nn_tensorflow_train import *

img_height: int = 1024
img_width: int = 1024
classes: list[int] = [0, 1, 2, 3]


class WeightedF1(keras.metrics.Metric):
    def __init__(self, n_classes: int, name="weighted_f1"):
        super(WeightedF1, self).__init__(name=name)
        self.f1score = F1Score(num_classes=n_classes, average="weighted")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(tf.one_hot(y_true, self.f1score.num_classes))
        self.f1score.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.f1score.result()

    def reset_state(self):
        self.f1score.reset_state()


def create_datasets(data_dir: Path, validation_split: float, batch_size: int, seed: Optional[int]) -> Dataset:
    seed = seed or time_ns() // 1_000_000_000
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    y_train = [int(Path(f).parent.name) for f in train_ds.file_paths]
    class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

    # configure prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return Dataset(train_set=train_ds, valid_set=val_ds, class_weights=dict(enumerate(class_weights)))


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Error: " + str(e))


def predict_model(model, data_dir: Path):
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    img_paths = dict((int(i.stem), i) for i in data_dir.glob("*.png"))
    predictions = np.full(max(img_paths.keys()) + 1, -1)

    for idx, img_path in img_paths.items():
        img_array = tf.expand_dims(img_to_array(load_img(img_path)), 0)
        pred_prob = probability_model(img_array)
        predictions[idx] = classes[np.argmax(pred_prob)]

    np.save(str(data_dir / Path("y_test")), predictions)
    np.savetxt(str(data_dir / Path("y_test.csv")), predictions.T, fmt="%d", delimiter=",")


def plot_training_history(hist, metric_name):
    metric = hist.history[metric_name]
    val_metric = hist.history["val_" + metric_name]

    epochs_range = range(len(metric))

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, metric, label='Training ' + metric_name)
    plt.plot(epochs_range, val_metric, label='Validation ' + metric_name)
    plt.legend(loc='upper left')
    plt.title('Training and Validation ' + metric_name)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", metavar="img_dir", help="training images dir")
    parser.add_argument("-p", "--predict", metavar="img_dir", help="predict using images from dir")
    parser.add_argument("-m", "--model", metavar="model_id", help="choose a model", type=int, default=0)
    parser.add_argument("-s", "--save", metavar="model_dir", help="save trained model to dir")
    parser.add_argument("-l", "--load", metavar="model_dir", help="load pre-trained model from dir")
    parser.add_argument("-ht", "--hyper_tune", metavar="hpt_dir", help="hyper-parameters tuning dir", default=".")
    parser.add_argument("-b", "--batch_size", metavar="batch_size", help="number of images in each training batch",
                        type=int, default=8)
    parser.add_argument("-e", "--epochs", metavar="epochs", help="number of training epochs",
                        type=int, default=30)
    parser.add_argument("-v", "--validation_split", metavar="proportion",
                        help="proportion of training data reserved for validation",
                        type=float, default=0.2)
    parser.add_argument("-r", "--rng_seed", metavar="seed_num",
                        help="seed used for randomisation",
                        type=int, default=None)
    args = parser.parse_args()

    if args.rng_seed is not None:
        tf.random.set_seed(args.rng_seed)

    configure_gpu()
    models_and_training = {
        1: (create_model_1, train_model),
        2: (create_model_2, lambda *a: tune_and_train_model(*a, tuning_dir=args.hyper_tune, model_id=2)),
        3: (create_model_3, train_model),
        4: (create_model_4, train_model),
        5: (create_model_5, train_model),
        6: (create_model_6, lambda *a: tune_and_train_model(*a, tuning_dir=args.hyper_tune, model_id=6)),
        7: (create_model_7, train_model),
        8: (create_model_8, train_model_with_class_weights),
    }
    model_builder, model_trainer = models_and_training.get(args.model, (None, None))
    model = None

    if model_builder is not None \
            and model_trainer is not None \
            and args.train is not None:
        dataset = create_datasets(Path(args.train), args.validation_split, args.batch_size, args.rng_seed)
        f1 = WeightedF1(n_classes=len(classes), name="weighted_f1")
        metric_accuracy = "accuracy"
        model, history = model_trainer(model_builder,
                                       img_height,
                                       img_width,
                                       classes,
                                       [f1, metric_accuracy],
                                       dataset,
                                       args.epochs)
        plot_training_history(history, f1.name)
        plot_training_history(history, metric_accuracy)
        plot_training_history(history, "loss")

        if args.save is not None:
            save_dir = Path(args.save)
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(save_dir))
    elif args.load is not None:
        model = tf.keras.models.load_model(args.load)
    else:
        print("Must either create and train a new model from dataset or load from saved models")
        exit()

    print(model.summary())

    if args.predict is not None:
        predict_model(model, Path(args.predict))

    exit()
