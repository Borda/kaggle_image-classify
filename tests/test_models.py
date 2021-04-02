from kaggle_plantpatho.models import LitPlantPathology, LitResnet


def test_create_resnet():
    LitResnet(arch='resnet18')


def test_create_model():
    net = LitResnet(arch='resnet18')
    LitPlantPathology(model=net)
