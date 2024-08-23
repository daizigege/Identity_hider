
import torch.nn as nn
import torchvision.models as models
import torch
device = torch.device('cuda')

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512, bias=True)
        # for param in self.resnet.parameters():
        #     param.requires_grad = True
        # # # 需欸
        # for param in self.resnet.fc.parameters():
        #     param.requires_grad = False
    def forward(self, input):
        output = self.resnet(input)
        return output


class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.mask_encoder=Encoder()

    def forward(self,X):
        X=self.mask_encoder(X)
        return X


