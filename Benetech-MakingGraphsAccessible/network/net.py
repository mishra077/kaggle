import torch
import torch.nn as nn
import torchvision.models as models


class InceptionV3:
    def __init__(self, model_checkpoint, num_classes = 5, pretrained = False):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_checkpoint = model_checkpoint
        self.model = self.load_model()

    def load_model(self):
        print(self.model_checkpoint)
        model = models.inception_v3(pretrained = self.pretrained, progress = True, aux_logits = True)

        # Load the model checkpoint if it exists
        if self.model_checkpoint is not None:
            model.load_state_dict(torch.load(self.model_checkpoint))

        # Freeze the weights
        for param in model.parameters():
            param.requires_grad = False

        # Change the last layer to output the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)

        return model
        

    def get_model(self):
        return self.model