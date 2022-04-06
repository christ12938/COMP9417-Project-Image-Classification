import argparse
import shutil
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("x_train", help="x_train_png dir")
    parser.add_argument("y_train", help="y_train.npy path")
    args = parser.parse_args()

    x_train_dir = Path(args.x_train)
    y_train_path = args.y_train
    train_sorted_dir = x_train_dir.parent / Path("train_png_sorted")
    y_data = np.load(y_train_path)

    for img in x_train_dir.glob("*.png"):
        idx = int(img.stem)
        class_dir = train_sorted_dir / Path(f"{y_data[idx]:.0f}")
        class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, class_dir)

    exit()
