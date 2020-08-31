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


"""
    Generate 'classifier.py' class.
"""


class Classifier(object):
    def __init__(self, params):
        # Create Discrimination-Network.
        self.net = create_network()

        # Set training-finished parameter.
        self.net.load_state_dict(params)

        # Set as evaluation-mode.
        self.net.eval()

        # A function that convert Image as Tensor.
        self.transformer = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        # Connect classification-ID and classification-Name.
        self.classes = ["burrito", "taco"]

    def predict(self, img):
        # Convert Image to Tensor.
        x = self.transformer(img)

        # Add batch's dimension to the front
        # because the Pytorch always deal with
        # Batch.
        x = x.unsqueeze(0)

        # Calculate Output of Neural-Network.
        out = self.net(x)
        out = out.max(1)[1].item()

        # Convert predicted-classification-name.
        return self.classes[out]
