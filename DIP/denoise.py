from __future__ import print_function
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from models import *
import torch
from skimage.measure import compare_nrmse,compare_psnr
from utils.denoising_utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from models import get_net
import torch.optim
import random
import torch.nn as nn
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import imageio
from PIL import Image
import cv2
random.seed(2020)

from utils.inpainting_utils import *
unloader = transforms.ToPILImage()
def image_folder_custom_label(root, transform, idx2label,flag):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    if flag == "str":
        old_classes = old_data.classes
        label2idx = {}

        for i, item in enumerate(idx2label):
            label2idx[item] = i

        new_data = dsets.ImageFolder(root=root, transform=transform,
                                     target_transform=lambda x: idx2label.index(old_data.classes[x]))
        # new_data = dsets.ImageFolder(root=root, transform=transform)
        new_data.classes = idx2label
        new_data.class_to_idx = label2idx
        return new_data
    
    else:
        class_idx = json.load(open('imagenet_class_index.json'))
        
        old_classes = class_idx[str(int(old_data.classes[0]))][1]   
        old_data.classes = old_classes   
        label2idx = {}
        #print(old_data.tar)
        for i, item in enumerate(idx2label):
            label2idx[item] = i

        new_data = dsets.ImageFolder(root=root, transform=transform,
                                     target_transform=None)
        #new_data = dsets.ImageFolder(root=root, transform=transform)
        new_data.classes = idx2label
        new_data.class_to_idx = label2idx
        # print(old_data.imgs)
        return new_data, old_data.imgs



def FunctionName(f,save_path):
    global i, out_avg,psrn_noisy_last, last_net, net_input
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    dtype = torch.cuda.FloatTensor

    imsize =-1
    PLOT = True
    sigma = 0
    sigma_ = sigma/255.
    f = f

  

    img_pil = crop_image(get_image(f, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    


    #plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3,11);


    show_every=50
    figsize=5
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 0 # set to 1./20. for sigma=50
    LR = 0.01

    OPTIMIZER='adam' # 'LBFGS'
    show_every = 100
    exp_weight=0.99
            
    num_iter = 400
    input_depth = 32 
    figsize = 4 
    
    
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128, 
                  skip_n33u=128, 
                  skip_n11=4, 
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)
    
    # net = get_net(input_depth, 'UNet').type(dtype)

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
        
    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

    i = 0
    
    def closure():
        
        global i, out_avg, psrn_noisy_last, last_net, net_input
    
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
        out = net(net_input)
        
        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
                
        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()
            
        
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
        
        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        # if  PLOT and i % show_every == 0:
        #     out_np = torch_to_np(np.clip(torch_to_np(out_avg), 0, 1))
        #     plot_image_grid([np.clip(out_np, 0, 1), 
        #                     np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
            
            
        
        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy
                
        i += 1

        return total_loss

    # Init globals 
    
    last_net = None
    i = 0
        # net_input =net_input
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

        # Run
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    out_np = net(net_input)
        #q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);
    folder = os.path.exists(save_path+F[-34:-20])
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_path+F[-34:-20]) 
        print("creat success")        #makedirs 创建文件时如果路径不存在会创建这个路径
        image = out_np.cpu().clone() # we clone the tensor to not do changes on it
        image = image.squeeze(0) # remove the fake batch dimension
        image = unloader(image)
        image.save(save_path+F[-34:-4] + ".png")
    else:
        print("success")
        image = out_np.cpu().clone() # we clone the tensor to not do changes on it
        image = image.squeeze(0) # remove the fake batch dimension
        image = unloader(image)
        image.save(save_path+F[-34:-4] + ".png")


flag = "str"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class_idx = json.load(open('imagenet_class_index.json'))
idx2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    
ae_root = "G:/mix/ae_PGD-16"
save_path = "G:/123/re_PGD-16"
imagnet_data = image_folder_custom_label(root=ae_root, transform=transform, idx2label=idx2label, flag=flag)
data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
images, labels = iter(data_loader).next()
total = 0
Q=[]
for images, labels in data_loader:
    F = imagnet_data.imgs[total][0]
    total = total + 1
    print(F)
    print(save_path+F[-34:-4] + ".png")
    FunctionName(F,save_path)
    
ae_root = "G:/mix/ae_PGD-8"
save_path = "G:/123/re_PGD-8"
imagnet_data = image_folder_custom_label(root=ae_root, transform=transform, idx2label=idx2label, flag=flag)
data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
images, labels = iter(data_loader).next()
total = 0
Q=[]
for images, labels in data_loader:
    F = imagnet_data.imgs[total][0]
    total = total + 1
    print(F)
    print(save_path+F[-34:-4] + ".png")
    FunctionName(F,save_path)
    
# ae_root = "G:/ae/ae_FGSM-4"
# save_path = "G:/123/re_FGSM-4"
# imagnet_data = image_folder_custom_label(root=ae_root, transform=transform, idx2label=idx2label, flag=flag)
# data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
# images, labels = iter(data_loader).next()
# total = 0
# Q=[]
# for images, labels in data_loader:
#     F = imagnet_data.imgs[total][0]
#     total = total + 1
#     print(F)
#     print(save_path+F[-34:-4] + ".png")
#     FunctionName(F,save_path)
    
# ae_root = "G:/ae/ae_FGSM-2"
# save_path = "G:/123/re_FGSM-2"
# imagnet_data = image_folder_custom_label(root=ae_root, transform=transform, idx2label=idx2label, flag=flag)
# data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
# images, labels = iter(data_loader).next()
# total = 0
# Q=[]
# for images, labels in data_loader:
#     F = imagnet_data.imgs[total][0]
#     total = total + 1
#     print(F)
#     print(save_path+F[-34:-4] + ".png")
#     FunctionName(F,save_path)

# ae_root = "G:/ae/ae_FGSM-1"
# save_path = "G:/123/re_FGSM-1"
# imagnet_data = image_folder_custom_label(root=ae_root, transform=transform, idx2label=idx2label, flag=flag)
# data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
# images, labels = iter(data_loader).next()
# total = 0
# Q=[]
# for images, labels in data_loader:
#     F = imagnet_data.imgs[total][0]
#     total = total + 1
#     print(F)
#     print(save_path+F[-34:-4] + ".png")
#     FunctionName(F,save_path)
    
