from timm import create_model

import torch
import torch.nn as nn

class ModelResnet152dTimm(nn.Module):
    def __init__(self, output_count):
        super(ModelResnet152dTimm, self).__init__()
        self.model = create_model('resnet152d', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class ModelResnet50Timm(nn.Module):
    def __init__(self, output_count):
        super(ModelResnet50Timm, self).__init__()
        self.model = create_model('resnet152d', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class VisionTransformerTimm(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformerTimm, self).__init__()
        self.model = create_model('vit_base_patch32_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.model = create_model('vgg19', pretrained=True)
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)