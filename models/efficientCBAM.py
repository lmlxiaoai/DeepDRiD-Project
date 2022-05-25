import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class Dgo_Cat_Net1(nn.Module):
    def __init__(self, num_classes=2):
        super(Dgo_Cat_Net1, self).__init__()


        model = EfficientNet.from_name('efficientnet-b1', include_top=False)
        model_dict = model.state_dict()
        pretrain_path = './efficientnet-b1-f1951068.pth'
        pretrained_dict = torch.load(pretrain_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict.pop('fc.weight', None)
        pretrained_dict.pop('fc.bias', None)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


        self.features = model
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x