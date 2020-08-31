# Classifier.py
from torch import nn
from torchvision import transforms, models


"""
    Generate 'Neural-Network' function.
"""


def create_network():
    # Read resnet18
    # It doesn't need 'pretrained=True'
    # because parameter set values
    # behind of model.
    net = models.resnet18()

    # Change Last-Layer by change it as
    # 2-outputs Linear-Layer.
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim, 2)
    return net
