import torch.nn.functional as F
import argparse

import numpy as np
import cv2


import torch
import torchvision.utils as vutils

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=110, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)# todo
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=32 , help='# of sample images')
    parser.add_argument('--gpu', dest='gpu', action='store_true',default=True)
    return parser.parse_args(args)


args = parse()
print(args)


from maske_model import BiSeNet
masknet = BiSeNet(n_classes=19).to('cuda')
masknet.load_state_dict(torch.load('premodels/79999_iter.pth'))
masknet.eval()
for p in masknet.parameters():
    p.requires_grad = False
from PIL import Image
import torchvision.transforms as transforms
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
import os

aaa=0
for item in sorted(os.listdir('images/protected/')):

    print(aaa)
    Xs_path = 'images/original/' + item  # 原始图像



    Xt_path = 'images/protected/'+item # 换身份图像

    img_a=tf(Image.open(Xs_path)).unsqueeze(0).cuda()
    img_b=tf(Image.open(Xt_path)).unsqueeze(0).cuda()


    # attgan.train()
    img_a = img_a.cuda() if args.gpu else img_a
    img_b = img_b.cuda() if args.gpu else img_b


    out_a, _, _ = masknet(F.interpolate(img_a, (512, 512), mode='bilinear', align_corners=True))
    parsing = out_a.argmax(1)
    parsing = parsing.unsqueeze(1).repeat(1, 3, 1, 1)
    a1 = torch.zeros((parsing.shape)).to('cuda')
    a1[parsing == 0] = 1
    # a[parsing == 14] = 1
    a1[parsing == 15] = 1
    a1[parsing == 16] = 1
    a1 = F.interpolate(a1, (256, 256), mode='bilinear', align_corners=True)
    #########################################################
    out_b, _, _ = masknet(F.interpolate(img_b, (512, 512), mode='bilinear', align_corners=True))
    parsing = out_b.argmax(1)
    parsing = parsing.unsqueeze(1).repeat(1, 3, 1, 1)
    a2 = torch.zeros((parsing.shape)).to('cuda')
    a2[parsing == 0] = 1
    # a[parsing == 14] = 1
    a2[parsing == 15] = 1
    a2[parsing == 16] = 1
    a2 = F.interpolate(a2, (256, 256), mode='bilinear', align_corners=True)

    back_mask = a1 * a2
    face_mask = (a2 + 1) % 2
    # if aaa==21:
    #     print('asadadd')
    img_a1 = img_a.mul(back_mask)  # 背景


    img_b2 = img_b.mul(face_mask)  # 人脸

    img=img_b2+img_a1
    img[img == 0] = -1
    mask=face_mask+back_mask
    mask[mask == 0] = -1


    vutils.save_image(img, 'images/D/'+item,nrow=1, normalize=True, range=(-1., 1.))
    vutils.save_image(mask, 'images/E/'+item, nrow=1, normalize=True, range=(-1., 1.))

    for item in sorted(os.listdir('images/D/')):
        Xs_path = 'images/D/' + item  # 原始A类别
        Xt_path = 'images/E/' + item  # 换身份图像
        img = cv2.imread(Xs_path)
        mask = cv2.imread(Xt_path, 0)
        mask = (((mask / 255) + 1) % 2)
        mask = mask.astype(np.uint8)
        # dst = cv2.inpaint(img,mask,9,cv2.INPAINT_TELEA)
        dst = cv2.inpaint(img, mask, 9, cv2.INPAINT_NS)
        cv2.imwrite('images/bg_preserved/' + item, dst)


