"""
Simple StreamLit app fro plant classification

>> streamlit run streamlit-app.py
"""
import os

import gdown
import numpy as np
import streamlit as st
import torch

from kaggle_plantpatho.augment import TORCHVISION_VALID_TRANSFORM
from kaggle_plantpatho.data import PlantPathologyDM
from kaggle_plantpatho.models import LitPlantPathology, MultiPlantPathology
from PIL import Image

MODEL_PATH_GDRIVE = "https://drive.google.com/uc?id=1bynbFW0FpIt7fnqzImu2UIM1PHb9-yjw"
MODEL_PATH_LOCAL = "fgvc8_resnet50.pt"
UNIQUE_LABELS = ("scab", "rust", "complex", "frog_eye_leaf_spot", "powdery_mildew", "cider_apple_rust", "healthy")
LUT_LABELS = dict(enumerate(sorted(UNIQUE_LABELS)))


@st.cache(allow_output_mutation=True)
def get_model(model_path: str = MODEL_PATH_LOCAL) -> LitPlantPathology:

    if not os.path.isfile(model_path):
        # download models if it missing locally
        gdown.download(MODEL_PATH_GDRIVE, model_path, quiet=False)

    net = torch.load(model_path)
    model = MultiPlantPathology(model=net)
    return model.eval()


def process_image(
    model: LitPlantPathology,
    img_path: str = "tests/data/test_images/8a0d7cad7053f18d.jpg",
    streamlit_app: bool = False,
):
    if not img_path:
        return

    img = Image.open(img_path)
    if streamlit_app:
        st.image(img)

    img = TORCHVISION_VALID_TRANSFORM(img)

    with torch.no_grad():
        encode = model(img.unsqueeze(0))[0]
    # process classification outputs
    binary = np.round(encode.detach().numpy(), decimals=2)
    labels = PlantPathologyDM.binary_mapping(encode, LUT_LABELS)

    if streamlit_app:
        st.write(", ".join(labels))
    else:
        print(f"Binary: {binary} >> {labels}")


if __name__ == "__main__":
    st.set_option("deprecation.showfileUploaderEncoding", False)

    # Upload an image and set some options for demo purposes
    st.header("Plant Pathology Demo")
    img_file = st.sidebar.file_uploader(label="Upload an image", type=["png", "jpg"])

    # load model and ideally use cache version to speedup
    model = get_model()

    # run the app
    process_image(model, img_file, streamlit_app=True)
    # process_image(model)  # dry rn with locals
