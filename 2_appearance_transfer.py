import sys
import torch
sys.path.append('./face_modules/')
import torchvision.transforms as transforms
import torch.nn.functional as F
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from network.DisenNet import *
from face_modules.mtcnn import *
import cv2
import PIL.Image as Image
import numpy as np
import configparser

detector = MTCNN()
device = torch.device('cuda')
# device = torch.device('cpu')
G = DisenNet(c_id=512)
G.eval()
G.load_state_dict(torch.load('premodels/G_latest.pth', map_location=torch.device('cpu')))
G = G.cuda()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()

arcface.load_state_dict(torch.load('premodels/model_ir_se50.pth', map_location=device), strict=False)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
import os
aaa=0
for item in sorted(os.listdir('images/original/')):
    # Xs_path = '/home/server1/wangtao/code/add/test_images/1.jpg' + item  # 原始A类别
    # aaa = aaa+1
    Xs_path = 'images/original/' +item # 原始A类别
    # if aaa>20:
    #     Xs_path = '/home/server1/wangtao/code/add/test_images/3.jpg'  #原始A类别

    Xt_path = 'images/virtual/'+item
    save_path = 'images/protected/'+item

    Xs_raw = cv2.imread(Xs_path)
    Xt_raw = cv2.imread(Xt_path)


    try:
        Xs = detector.align(Image.fromarray(Xs_raw[:, :, ::-1]), crop_size=(256, 256))
    except Exception as e:
        print('the source image is wrong, please change the image')
        cv2.imwrite(save_path, Xt_raw)
        continue
    print(item)
    if np.array(Xs).any()==None:
        continue
    Xs_raw = np.array(Xs)[:, :, ::-1]
    Xs = test_transform(Xs)
    Xs = Xs.unsqueeze(0).cuda()

    with torch.no_grad():
        embeds = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
        # embeds = arcface(F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True))

    # Xt_raw = cv2.imread(Xt_path)
    try:
        Xt, trans_inv = detector.align(Image.fromarray(Xt_raw[:, :, ::-1]), crop_size=(256, 256), return_trans_inv=True)
    except Exception as e:
        print('the target image is wrong, please change the image')
        cv2.imwrite(save_path, Xt_raw)
        continue
    Xt_raw = Xt_raw.astype(np.float)/255.0
    Xt = test_transform(Xt)
    Xt = Xt.unsqueeze(0).cuda()

    mask = np.zeros([256, 256], dtype=np.float)
    for i in range(256):
        for j in range(256):
            dist = np.sqrt((i-128)**2 + (j-128)**2)/128
            dist = np.minimum(dist, 1)
            mask[i, j] = 1-dist
    mask = cv2.dilate(mask, None, iterations=20)

    with torch.no_grad():
        Yt, _ = G(Xt, embeds)
        Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        Yt = Yt[:, :, ::-1]
        Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = cv2.warpAffine(mask,trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = np.expand_dims(mask_, 2)
        Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*Xt_raw
        # cv2.imshow('image',Yt_trans_inv)
        cv2.imwrite(save_path,Yt_trans_inv*255)
        # cv2.imshow('image',Yt)
        # cv2.imwrite(save_path,Yt*255)
        print("the result image has been saved")
        # cv2.waitKey(0)
