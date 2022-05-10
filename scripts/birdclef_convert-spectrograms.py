import glob
import os
from functools import partial
from math import ceil

import fire
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import delayed, Parallel
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.no_grad()
def create_spectrogram(
    fname: str,
    reduce_noise: bool = False,
    frame_size: int = 5,
    frame_step: int = 2,
    channel: int = 0,
    device="cpu",
    batch_size=5,
) -> list:
    waveform, sample_rate = torchaudio.load(fname)
    transform = torchaudio.transforms.Spectrogram(n_fft=1800, win_length=512).to(device)
    if reduce_noise:
        waveform = torch.tensor(
            nr.reduce_noise(
                y=waveform,
                sr=sample_rate,
                win_length=transform.win_length,
                use_tqdm=False,
                n_jobs=2,
            )
        )

    step = int(frame_step * sample_rate)
    size = int(frame_size * sample_rate)
    frames = []
    for i in range(ceil((waveform.size()[-1] - size) / step)):
        begin = i * step
        end = begin + size
        frame = waveform[channel][begin:end]
        if len(frame) < size:
            if i == 0:
                rep = round(float(size) / len(frame))
                frame = frame.repeat(int(rep))
            elif len(frame) < (size * 0.33):
                continue
            else:
                frame = waveform[channel][-size:]
        frames.append(frame)
    if not frames:
        return []

    dl = DataLoader(frames, batch_size=batch_size)
    spectrograms = []
    for batch in dl:
        sgs = torch.log(transform(batch.to(device))).cpu()
        sgs = np.nan_to_num(sgs.numpy())
        spectrograms += [sgs[i, ...] for i in range(sgs.shape[0])]
    return spectrograms


def convert_and_export(
    fn: str,
    path_in: str,
    path_out: str,
    reduce_noise: bool = False,
    frame_size: int = 5,
    frame_step: int = 2,
    device="cpu",
    batch_size=2,
    img_extension: str = ".png",
    img_size: int = 512,
) -> None:
    path_audio = os.path.join(path_in, fn)
    try:
        sgs = create_spectrogram(
            path_audio,
            reduce_noise=reduce_noise,
            frame_size=frame_size,
            frame_step=frame_step,
            device=device,
            batch_size=batch_size,
        )
    except Exception as ex:
        print(f"Failed conversion for audio: {path_audio}\n with {ex}")
        return
    if not sgs:
        print(f"Too short audio for: {path_audio}")
        return
    path_npz = os.path.join(path_out, fn + ".npz")
    os.makedirs(os.path.dirname(path_npz), exist_ok=True)
    np.savez_compressed(path_npz, np.array(sgs, dtype=np.float16))
    for i, sg in enumerate(sgs):
        path_img = os.path.join(path_out, fn + f".{i:03}" + img_extension)
        try:
            if img_extension == ".png":
                sg = (sg + 70) / 90.0
                sg = np.clip(sg, a_min=0, a_max=1) * 255
                img = Image.fromarray(sg.astype(np.uint8))
                if img_size:
                    img = img.resize((img_size, img_size))
                img.save(path_img)
            else:
                plt.imsave(path_img, sg, vmin=-70, vmax=20)
        except Exception as ex:
            print(f"Failed exporting for image: {path_img}\n with {ex}")
            continue


def _color_means(img_path):
    img = plt.imread(img_path)
    if np.max(img) > 1.5:
        img = img / 255.0
    clr_mean = np.mean(img) if img.ndim == 2 else {i: np.mean(img[..., i]) for i in range(3)}
    clr_std = np.std(img) if img.ndim == 2 else {i: np.std(img[..., i]) for i in range(3)}
    return clr_mean, clr_std


def main(path_dataset: str, img_extension: str = ".png", img_size: int = 512, use_gpu: bool = False, n_jobs: int = 12):
    train_meta = pd.read_csv(os.path.join(path_dataset, "train_metadata.csv")).sample(frac=1)
    print(train_meta.head())

    _convert_and_export = partial(
        convert_and_export,
        path_in=os.path.join(path_dataset, "train_audio"),
        path_out=os.path.join(path_dataset, "train_images"),
        reduce_noise=True,
        device="cuda" if use_gpu else "cpu",
        batch_size=3 if use_gpu else 1,
        img_extension=img_extension,
        img_size=img_size,
    )

    _ = Parallel(n_jobs=n_jobs)(delayed(_convert_and_export)(fn) for fn in tqdm(train_meta["filename"]))
    # _= list(map(_convert_and_export, tqdm(train_meta["filename"])))

    images = glob.glob(os.path.join(path_dataset, "train_images", "*", "*" + img_extension))
    clr_mean_std = Parallel(n_jobs=os.cpu_count())(delayed(_color_means)(fn) for fn in tqdm(images))
    img_color_mean = pd.DataFrame([c[0] for c in clr_mean_std]).describe()
    print(img_color_mean.T)
    img_color_std = pd.DataFrame([c[1] for c in clr_mean_std]).describe()
    print(img_color_std.T)
    img_color_mean = list(img_color_mean.T["mean"])
    img_color_std = list(img_color_std.T["mean"])
    print(f"MEAN: {img_color_mean}\n STD: {img_color_std}")


if __name__ == "__main__":
    fire.Fire(main)
