from cldc.models import LitCassava, LitMobileNet, LitResnet


def test_create_resnet():
    LitResnet(arch='resnet18')


def test_create_mobnet():
    LitMobileNet(arch='mobilenet_v3_small')


def test_create_model():
    net = LitMobileNet(arch='mobilenet_v3_small')
    LitCassava(model=net)
