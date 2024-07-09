import torch
import lightning as L
from torchvision.models import \
    vgg11_bn, VGG11_BN_Weights, \
    vit_b_16, ViT_B_16_Weights, \
    resnet152, ResNet152_Weights, \
    densenet201, DenseNet201_Weights, \
    resnext101_32x8d, ResNeXt101_32X8D_Weights
from torch import nn

    
    
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from config import ALL_GENRES

class HouseXModel(L.LightningModule):
    def __init__(self, extractor_name = 'vit_b_16'):
        super(HouseXModel, self).__init__()
        
        if extractor_name == 'vit_b_16':
            self.extractor = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif extractor_name == 'vgg11_bn':
            self.extractor = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        elif extractor_name == 'resnet152':
            self.extractor = resnet152(weights=ResNet152_Weights.DEFAULT)
        elif extractor_name == 'densenet201':
            self.extractor = densenet201(weights=DenseNet201_Weights.DEFAULT)
        elif extractor_name == 'resnext101_32x8d':
            self.extractor = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
        else:
            raise NotImplementedError(f"Extractor {extractor_name} is not supported.")
        
        self.adapter = nn.Linear(1000, 768)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=6
        )
        

if __name__ == "__main__":
    for extractor_name in ['vit_b_16', 'vgg11_bn', 'resnet152', 'densenet201', 'resnext101_32x8d']:
        with open(f"{extractor_name}.txt", "w") as f:
            model = HouseXModel(extractor_name=extractor_name)
            for name, param in model.named_parameters():
                f.write(name + ' ' + str(param.shape) + '\n')