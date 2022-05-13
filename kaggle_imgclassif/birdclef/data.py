import os
from math import ceil

import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import noisereduce
except ImportError:
    noisereduce = object

SPECTROGRAM_PARAMS = dict(
    sample_rate=32_000, hop_length=640, n_fft=800, n_mels=128, fmin=20, fmax=16_000, win_length=512
)
PCEN_PARAS = dict(
    time_constant=0.06,
    eps=1e-6,
    gain=0.8,
    power=0.25,
    bias=10,
)
SPECTROGRAM_RANGE = (-80, 0)


def create_spectrogram(
    fname: str,
    reduce_noise: bool = False,
    frame_size: int = 5,
    frame_step: int = 2,
    spec_params: dict = SPECTROGRAM_PARAMS,
) -> list:
    waveform, sample_rate = librosa.core.load(fname, sr=spec_params["sample_rate"], mono=True)
    if reduce_noise:
        waveform = noisereduce.reduce_noise(
            y=waveform,
            sr=sample_rate,
            time_constant_s=float(frame_size),
            time_mask_smooth_ms=250,
            n_fft=spec_params["n_fft"],
            use_tqdm=False,
            n_jobs=2,
        )

    frames = cut_frames(waveform, sample_rate, frame_size, frame_step)
    spectrograms = []
    for frm in frames:
        sg = librosa.feature.melspectrogram(
            y=frm,
            sr=sample_rate,
            n_fft=spec_params["n_fft"],
            win_length=spec_params["win_length"],
            hop_length=spec_params["hop_length"],
            n_mels=spec_params["n_mels"],
            fmin=spec_params["fmin"],
            fmax=spec_params["fmax"],
            power=1,
        )
        # sg = librosa.pcen(sg, sr=sample_rate, hop_length=spec_params["hop_length"], **PCEN_PARAS)
        sg = librosa.amplitude_to_db(sg, ref=np.max)
        spectrograms.append(np.nan_to_num(sg))
    return spectrograms


def cut_frames(
    waveform,
    sample_rate: int,
    frame_size: int = 5,
    frame_step: int = 2,
    min_frame_fraction: float = 0.2,
):
    step = int(frame_step * sample_rate)
    size = int(frame_size * sample_rate)
    count = ceil((len(waveform) - size) / float(step))
    frames = []
    for i in range(max(1, count)):
        begin = i * step
        end = begin + size
        frame = waveform[begin:end]
        if len(frame) < size:
            if i == 0:
                rep = round(float(size) / len(frame))
                frame = frame.repeat(int(rep))
            elif len(frame) < (size * min_frame_fraction):
                continue
            else:
                frame = waveform[-size:]
        frames.append(frame)
    return frames


def convert_and_export(
    fn: str,
    path_in: str,
    path_out: str,
    reduce_noise: bool = False,
    frame_size: int = 5,
    frame_step: int = 2,
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
        )
    except Exception as ex:
        print(f"Failed conversion for audio: {path_audio}\n with {ex}")
        return
    if not sgs:
        print(f"Too short audio for: {path_audio} with ")
        return
    path_npz = os.path.join(path_out, fn + ".npz")
    os.makedirs(os.path.dirname(path_npz), exist_ok=True)
    np.savez_compressed(path_npz, np.array(sgs, dtype=np.float16))
    for i, sg in enumerate(sgs):
        path_img = os.path.join(path_out, fn + f".{i:03}" + img_extension)
        try:
            if img_extension == ".png":
                sg = (sg - SPECTROGRAM_RANGE[0]) / float(SPECTROGRAM_RANGE[1] - SPECTROGRAM_RANGE[0])
                sg = np.clip(sg, a_min=0, a_max=1) * 255
                img = Image.fromarray(sg.astype(np.uint8))
                if img_size:
                    img = img.resize((img_size, img_size))
                img.save(path_img)
            else:
                plt.imsave(path_img, sg, vmin=SPECTROGRAM_RANGE[0], vmax=SPECTROGRAM_RANGE[1])
        except Exception as ex:
            print(f"Failed exporting for image: {path_img}\n with {ex}")
            continue
