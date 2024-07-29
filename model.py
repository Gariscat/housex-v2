import torch
import lightning as L
from torchvision.models import \
    vgg11_bn, VGG11_BN_Weights, \
    vit_b_16, ViT_B_16_Weights, \
    resnet152, ResNet152_Weights, \
    densenet201, DenseNet201_Weights, \
    resnext101_32x8d, ResNeXt101_32X8D_Weights
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from config import ALL_GENRES, HOP_FRAME
from easydict import EasyDict as edict

class HouseXModel(L.LightningModule):
    def __init__(self,
        model_config: edict,
    ):
        super(HouseXModel, self).__init__()
        
        self.config = model_config
        if self.config.extractor_name == 'vit_b_16':
            backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif self.config.extractor_name == 'vgg11_bn':
            backbone = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        elif self.config.extractor_name == 'resnet152':
            backbone = resnet152(weights=ResNet152_Weights.DEFAULT)
        elif self.config.extractor_name == 'densenet201':
            backbone = densenet201(weights=DenseNet201_Weights.DEFAULT)
        elif self.config.extractor_name == 'resnext101_32x8d':
            backbone = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
        else:
            raise NotImplementedError(f"Extractor {self.config.extractor_name} is not supported.")
        
        self.extractor = nn.Sequential(
            backbone,
            nn.ReLU(),
            nn.Linear(1000, 768),
            nn.Tanh()
        )
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=model_config.transformer_num_layers
        )
        self.fc = nn.Linear(768, len(ALL_GENRES))
        
    def forward(self, x):
        b, c, h, w = x.shape
        assert h == 224
        
        features = []
        for i_frame in range(0, w-224, HOP_FRAME):
            cur_patch = x[:, :, :, i_frame:i_frame+224]
            if cur_patch.shape[-1] != 224:
                zero_paddings = torch.zeros(b, c, h, cur_patch.shape[-1]-224)
                cur_patch = torch.cat((cur_patch, zero_paddings), dim=-1)
            y = self.extractor(cur_patch)
            features.append(y)
            
        embedding = torch.stack(features) # (n, b, d)
        out = self.encoder(embedding)[0] # (0, d)
        out = nn.ReLU()(out)
        out = self.fc(out)
        
        return out
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.config.loss_weight is not None:
            self.config.loss_weight = self.config.loss_weight.to(x.device)
        loss = nn.CrossEntropyLoss(weight=self.config.loss_weight)(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.config.loss_weight is not None:
            self.config.loss_weight = self.config.loss_weight.to(x.device)
        loss = nn.CrossEntropyLoss(weight=self.config.loss_weight)(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        # sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=1)
        sch = optim.lr_scheduler.StepLR(optimizer, step_size=256, gamma=0.9)
        return [optimizer,], [sch,]
    
    
        
if __name__ == "__main__":
    for extractor_name in ['vit_b_16', 'vgg11_bn', 'resnet152', 'densenet201', 'resnext101_32x8d']:
        with open(f"misc/{extractor_name}.txt", "w") as f:
            model = HouseXModel(extractor_name=extractor_name)
            for name, param in model.named_parameters():
                f.write(name + ' ' + str(param.shape) + '\n')
