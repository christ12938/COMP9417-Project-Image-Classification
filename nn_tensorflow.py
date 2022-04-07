import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory, img_to_array, load_img

from nn_tensorflow_models import *

img_height = 1024
img_width = 1024
classes = [0, 1, 2, 3]
batch_size_train = 16
epochs = 20
models = {1: create_model_1}
validation_split = 0.1


def create_datasets(data_dir: Path, batch_size: int):
    seed = 1
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

    # configure prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def train_model(model, train_dataset, validation_dataset, epochs: int):
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs
    )
    return history


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
    predictions = np.empty(len(img_paths))

    for idx, img_path in img_paths.items():
        img_array = tf.expand_dims(img_to_array(load_img(img_path)), 0)
        pred_prob = probability_model(img_array)
        predictions[idx] = classes[np.argmax(pred_prob)]

    np.save(str(data_dir / Path("y_test")), predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", metavar="img_dir", help="training images dir")
    parser.add_argument("-p", "--predict", metavar="img_dir", help="predict using images from dir")
    parser.add_argument("-m", "--model", metavar="model_id", help="choose a model", type=int, default=0)
    parser.add_argument("-s", "--save", metavar="model_dir", help="save trained model to dir")
    parser.add_argument("-l", "--load", metavar="model_dir", help="load pre-trained model from dir")
    args = parser.parse_args()

    configure_gpu()
    create_model_func = models.get(args.model)
    model = None

    if create_model_func is not None and args.train is not None:
        model = create_model_func(img_height, img_width, classes)
        train_ds, val_ds = create_datasets(Path(args.train), batch_size_train)
        train_model(model, train_ds, val_ds, epochs)

        if args.save is not None:
            save_dir = Path(args.save)
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(save_dir))
    elif args.load is not None:
        model = tf.keras.models.load_model(args.load)
    else:
        print("Must either create and train a new model from dataset or load from saved models")
        exit()

    if args.predict is not None:
        predict_model(model, Path(args.predict))

    exit()
