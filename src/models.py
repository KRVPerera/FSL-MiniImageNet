from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights
import torch
import torch.nn as nn

class VGG11_BN_FC_Changed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = VGG11_BN_Weights.DEFAULT
        self.model = vgg11_bn(weights=weights, progress=False)
        self.transforms = weights.transforms(antialias=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transforms(x)
        return self.model(x)

class Resnet18_FC_Changed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False)
        self.transforms = weights.transforms(antialias=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transforms(x)
        return self.resnet18(x)
    

class EfficientNetB0_FC_Changed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights, progress=False)
        self.transforms = weights.transforms(antialias=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transforms(x)
        return self.model(x)
    
class Shufflenet_v2_x0_5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
        self.model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT, progress=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)