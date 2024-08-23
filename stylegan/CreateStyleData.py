
import torch
import random
import torchvision
import cv2
from stylegan.model import Generator
from stylegan.mymodel import mymodel
device='cuda'
# device='cpu'
class CreateStyleData():
    def __init__(self):
        self.generator = Generator(256, 512, 8, 2).to(device)
        self.generator.load_state_dict(torch.load('premodels/generator_ffhq.pt')["g_ema"])
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False
        self.G = mymodel().to(device)
        self.G.load_state_dict(torch.load('premodels/G_only.pth'))
        self.G.eval()
        for p in self.G.parameters():
            p.requires_grad = False
        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def return_style_data(self,X):
        with torch.no_grad():
            # Xs = Xs.unsqueeze(0).cuda()
            W=self.G(X.unsqueeze(0))
            W_plus=self.generator.get_latentplus([W],truncation=4096,truncation_latent=self.mean_latent,input_is_latent=True)
            Y, _ = self.generator(W_plus)
            return Y.squeeze(0)

    def return_diverse_style_data(self,X):
        with torch.no_grad():
            # Xs = Xs.unsqueeze(0).cuda()
            W=self.G(X.unsqueeze(0))
            W_plus=self.generator.get_latentplus([W],truncation=4096,truncation_latent=self.mean_latent,input_is_latent=True)
            # if random.random()>0.1:
            Z = torch.randn(1, 512, device=device)
            W = self.generator.get_latentplus([Z], truncation=4096, truncation_latent=self.mean_latent, input_is_latent=False)
            W_plus[:, 5:14] = W[:, 5:14]

            Y, _ = self.generator(W_plus)
            return Y.squeeze(0)














