from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
from stylegan.CreateStyleData import CreateStyleData
import random
import os

class FaceEmbed(TensorDataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = sorted(os.listdir(self.data_path))
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.style_data=CreateStyleData()
    def __getitem__(self, item):
        # 身份图像
        X = self.tf(Image.open(os.path.join(self.data_path, self.images[item]))).to('cuda')
        Xb=self.style_data.return_style_data(X)
        # 多样保护结果
        # Xb = self.style_data.return_diverse_style_data(X)
        return X,Xb,self.images[item]
    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    import torchvision.utils as vutils
    from torch.utils.data import DataLoader

    dataset = FaceEmbed('images/original/')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    from torchvision.transforms import Resize
    for x,y,z in dataloader:
        # vutils.save_image(x, 'imageA/'+z[0][:-4], nrow=4, normalize=True, range=(-1., 1.))
        vutils.save_image(y, 'images/virtual/'+z[0], nrow=4, normalize=True, range=(-1., 1.))
        print(x)
        # break




