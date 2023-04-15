import torch
import torch.nn as nn
import torchvision.models as models


class InceptionV3:
    def __init__(self, num_classes = 4, pretrained = False, model_checkpoint = None):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None
        self.model_checkpoint = model_checkpoint

    def load_model(self):
        self.model = models.inception_v3(pretrained = self.pretrained, progress = True, aux_logits = False)

        # Freeze the weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Change the last layer to output the number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

        # Load the model checkpoint if it exists
        if self.model_checkpoint is not None:
            self.model.load_state_dict(torch.load(self.model_checkpoint))

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model