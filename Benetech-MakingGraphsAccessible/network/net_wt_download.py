import os
import torch
import torchvision.models as models


DIR_PATH = os.path.join(os.getcwd(), '..', 'weights')
WT_PATH = '../weights/inception_v3_wt.pth'

# Download the weights if it doesn't exist
def download_weights():
    # Create the directory if it doesn't exist
    if not os.path.exists(DIR_PATH):
        print("Creating directory for weights...")
        os.makedirs(DIR_PATH)
        print("Created directory for weights at {}".format(DIR_PATH))

    if not os.path.isfile(WT_PATH):
        print("Downloading pretrained weights...")
        
        model = models.inception_v3(pretrained=True)
        torch.save(model.state_dict(), WT_PATH)
        print("Downloaded pretrained weights to {}".format(WT_PATH))

    else:
        print("Pretrained weights already exist at {}".format(WT_PATH))


def load_weights(model):
    download_weights()
    model.load_state_dict(torch.load(WT_PATH))
    return model

# Test if model is loaded correctly
if __name__ == '__main__':
    model = models.inception_v3(pretrained=False)
    model = load_weights(model)
    print(model)