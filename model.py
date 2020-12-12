from torchvision import models
from torch import nn


class Encoder(nn.Module):
    """
    Encoder: use RESNET pretrained model but
    fine tune it using our training dataset

    For dev purpose, using resnet-34,
    For prod, using resnet-152
    """

    def __init__(self):
        super(Encoder, self).__init__()
        pretrained_model = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.linear = nn.Linear(pretrained_model.fc.in_features, out_features=1000)
        self.batchnorm = nn.BatchNorm1d(num_features=1000, momentum=0.01)

        # init weights randomly with normal distribution
        nn.init.xavier_normal(self.linear.weight, gain=1)
        nn.init.constant_(self.linear.bias, 0)

