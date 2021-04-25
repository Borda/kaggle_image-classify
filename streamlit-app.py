"""
Simple StreamLit app fro plant classification

>> streamlit run streamlit-app.py
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np

from kaggle_plantpatho.augment import TORCHVISION_VALID_TRANSFORM
from kaggle_plantpatho.data import PlantPathologyDM
from kaggle_plantpatho.models import MultiPlantPathology

PATH_MODEL = 'assets/fgvc8_resnet50.pt'
UNIQUE_LABELS = ('scab', 'rust', 'complex', 'frog_eye_leaf_spot', 'powdery_mildew', 'cider_apple_rust', 'healthy')
LUT_LABELS = dict(enumerate(sorted(UNIQUE_LABELS)))


def process_image(img_path: str = 'tests/data/test_images/8a0d7cad7053f18d.jpg', model_path: str = PATH_MODEL, streamlit_app: bool = False):
    if not img_path:
        return

    img = Image.open(img_path)
    if streamlit_app:
        st.image(img)

    net = torch.load(model_path)
    model = MultiPlantPathology(model=net)
    model.eval()

    img = TORCHVISION_VALID_TRANSFORM(img)

    with torch.no_grad():
        onehot = model(img.unsqueeze(0))[0]
    onehot_bin = np.round(onehot.detach().numpy(), decimals=2)
    labels = PlantPathologyDM.onehot_mapping(onehot, LUT_LABELS)

    if streamlit_app:
        st.write(', '.join(labels))
    else:
        print(f"Onehot: {onehot_bin} >> {labels}")


st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Plant Pathology")
img_file = st.sidebar.file_uploader(label='Upload an image', type=['png', 'jpg'])

# run the app
process_image(img_file, streamlit_app=True)
