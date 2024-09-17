import glob
import os
from functools import partial

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from kaggle_imgclassif.birdclef.data import convert_and_export


def _color_means(img_path):
    img = plt.imread(img_path)
    if np.max(img) > 1.5:
        img = img / 255.0
    clr_mean = np.mean(img) if img.ndim == 2 else {i: np.mean(img[..., i]) for i in range(3)}
    clr_std = np.std(img) if img.ndim == 2 else {i: np.std(img[..., i]) for i in range(3)}
    return clr_mean, clr_std


def main(
    path_dataset: str, reduce_noise: bool = False, img_extension: str = ".png", img_size: int = 256, n_jobs: int = 12
):
    train_meta = pd.read_csv(os.path.join(path_dataset, "train_metadata.csv")).sample(frac=1)
    print(train_meta.head())

    _convert_and_export = partial(
        convert_and_export,
        path_in=os.path.join(path_dataset, "train_audio"),
        path_out=os.path.join(path_dataset, "train_images"),
        reduce_noise=reduce_noise,
        img_extension=img_extension,
        img_size=img_size,
    )

    _ = Parallel(n_jobs=n_jobs)(delayed(_convert_and_export)(fn) for fn in tqdm(train_meta["filename"]))
    # _= list(map(_convert_and_export, tqdm(train_meta["filename"])))

    images = glob.glob(os.path.join(path_dataset, "train_images", "*", "*" + img_extension))
    clr_mean_std = Parallel(n_jobs=n_jobs)(delayed(_color_means)(fn) for fn in tqdm(images))
    img_color_mean = pd.DataFrame([c[0] for c in clr_mean_std]).describe()
    print(img_color_mean.T)
    img_color_std = pd.DataFrame([c[1] for c in clr_mean_std]).describe()
    print(img_color_std.T)
    img_color_mean = list(img_color_mean.T["mean"])
    img_color_std = list(img_color_std.T["mean"])
    print(f"MEAN: {img_color_mean}\n STD: {img_color_std}")


if __name__ == "__main__":
    fire.Fire(main)
