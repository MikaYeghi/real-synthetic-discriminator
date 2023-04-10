import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

class ResNet101(nn.Module):
    def __init__(self, n_classes=1, activation=True, device='cuda'):
        super(ResNet101, self).__init__()
        
        self.n_classes = 1
        self.backbone = self.get_backbone(n_classes).to(device)
        self.device = device
        
        if activation:
            if n_classes == 1:
                self.activation = nn.Sigmoid()
            elif n_classes > 1:
                raise NotImplementedError
            else:
                return ValueError(f"Number of output classes ({n_classes}) not valid")
        
    def get_backbone(self, n_classes):
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        backbone.fc = nn.Linear(
            in_features=backbone.fc.in_features, 
            out_features=n_classes,
            bias=True
        )
        
        return backbone
        
    def forward(self, inputs):
        return self.activation(self.backbone(inputs))