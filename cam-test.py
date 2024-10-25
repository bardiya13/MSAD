from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import ast
from torch.nn import functional as F
import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudn
from torch.nn import functional as F
from unet_parts import *
# from scipy.misc import imsave
import torch.nn as nn
import ast
import sys
import imageio
# from skimage import img_as_ubyte
from sklearn.metrics import roc_curve, auc

import cv2

from rGAN import Generator, Discriminator
from dataset import TrainingDataset
from utils import createEpochData, roll_axis, loss_function, create_folder, prep_data, createEpochDataTest

# load functions from the training script for the finetuning of the model
from train import Load_Dataloader, overall_generator_pass, overall_discriminator_pass, meta_update_model
import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def overall_generator_pass_test(generator, discriminator, img, gt, valid):
    # print(len(img), gt.shape)
    recon_batch = generator(img)
    recon_batch = (recon_batch-recon_batch.min()) / (recon_batch.max() - recon_batch.min())
    gt = (gt-gt.min()) / (gt.max()-gt.min())
    msssim, f1, psnr = loss_function(recon_batch, gt)
    # print(msssim, f1, psnr)
    
    imgs = recon_batch.data.cpu().numpy()[0, :]
    imgs = roll_axis(imgs)
    
    return imgs, psnr

def frame_visualization(img):
    for frame in range(len(img)):
        one_img = np.squeeze(img[frame])
        one_img = one_img.cpu()
        one_img = np.transpose(one_img, (1, 2, 0))
        plt.imshow(one_img)
        plt.axis('off')
        plt.show()
        
def pred_frame_visualization(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


"""TEST SCRIPT"""


num_tasks = 3
adam_betas = (0.5, 0.999)
gen_lr = 2e-4
dis_lr = 1e-5
model_folder_path = "model"
torch.manual_seed(1)
batch_size = 1
anomaly_counts = 0
    
generator = Generator(batch_size=batch_size) 
discriminator = Discriminator()
generator.cuda()
discriminator.cuda()

path = r'C:\Users\liyun\Desktop\01_0015.npy'
loaded_array = np.load(path)
k_shots = len(loaded_array)

# define dataloader
tf = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

create_folder(model_folder_path)
generator_path = os.path.join(model_folder_path, str.format("Generator_finetuned_Final_shtech.pt"))
discriminator_path = os.path.join(model_folder_path, str.format("Discriminator_finetuned_Final_shtech.pt"))

# load the pre-trained model
print('- start loading pre-trained model')

generator.load_state_dict(torch.load(generator_path))
discriminator.load_state_dict(torch.load(discriminator_path))
# if you use CPU
#     generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
#     discriminator.load_state_dict(torch.load(discriminator_path, map_location=torch.device('cpu')))
    
print('- loading pretrained model done')
    
# this path must be the video frames for testing purposes
# Lei uses the fake dataset as an example here
frame_path = r'C:\Users\liyun\Desktop\testing\frames'
    
# the test dataloader
train_path_list = createEpochDataTest(frame_path, num_tasks, k_shots)
train_dataloader = Load_Dataloader(train_path_list, tf, batch_size)

    
# Meta-Validation
print ('\n Meta Validation/Test \n')

# forward pass
    
for i, epoch_of_tasks in enumerate(train_dataloader):
    epoch_results = 'results'# .format(epoch+1)
    create_folder(epoch_results)
    print(len(epoch_of_tasks))
        
    for tidx, task in enumerate(epoch_of_tasks):
        print(task)
        s_list = []
        scoreset = []
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
                
            # Meta-Validation
            print ('\n Meta Validation/Test \n')



            for vidx, val_frame_sequence in enumerate(task[-k_shots:]):
                # print(vidx)
                if vidx == 0:
                    dummy_frame_sequence = val_frame_sequence
                        
                if 1: # vidx % 2 == 0:
                        

                    img = val_frame_sequence[0]

                    # frame_visualization(img)

                    gt = val_frame_sequence[1]

                        

                    img, gt, valid, fake = prep_data(img, gt)

                    # k-Validation Generator
                    imgs, psnr = overall_generator_pass_test(generator, discriminator, img, gt, valid)
                    img_path = os.path.join(epoch_results,'{}-fig-val{}.png'.format(tidx+1, vidx+1))
                    # imsave(img_path , imgs)

                    # imgs = imgs.astype(np.uint8)
                    # imgs = (imgs-np.min(imgs))/(np.max(imgs) - np.min(imgs))
                    imgs = cv2.normalize(imgs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    # imageio.imwrite(img_path , img_as_ubyte(imgs))
                        
                    s_list.append(psnr)
                    print('Frame No.: ', vidx, ' ...')



normalized_psnr_list = []


y_score = np.array(s_list)
y_gt = loaded_array[3:]
print(y_gt)
    
for psnr in y_score:
    normalized_psnr = 1 - (psnr - np.min(y_score)) / (np.max(y_score) - np.min(y_score))
    normalized_psnr_list.append(normalized_psnr)
y_psnr_scores = np.array(normalized_psnr_list)
print(normalized_psnr_list)
fpr, tpr, thresholds = roc_curve(y_gt, y_psnr_scores)
print(thresholds)
    
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
    
    
    

x_values = range(len(y_psnr_scores))

plt.plot(x_values, y_psnr_scores, marker='o', linestyle='-')

plt.title('The line chart')
plt.xlabel('i-th frame')
plt.ylabel('anomaly score')
# plt.axhline(y=0.98, color='r', linestyle='--', label='y=0.98')
plt.show()