import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

class ModifiedResNet(nn.Module):
    """ResNet-50."""

    def __init__(self, make_model=True):
        super(ModifiedResNet, self).__init__()
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNet, self).train(mode)

        # set the BNs to eval mode
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""

        resnet = models.resnet50(pretrained=True) # get the pretrained model.
        self.datasets, self.classifiers = [], nn.ModuleList()

        # create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # add the first classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(resnet.fc)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(2048, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets

        self.classifier = self.classifiers[self.datasets.index(dataset)]
        #print("current classifier is :", self.classifier)

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
