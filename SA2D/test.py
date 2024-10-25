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
from imageio import imwrite as imsave
# from skimage import img_as_ubyte
import rGAN
from rGAN import Generator, Discriminator
from dataset import TrainingDataset
from utils import createEpochData, roll_axis, loss_function, create_folder, prep_data, createEpochDataTest

import cv2

import warnings
warnings.filterwarnings("ignore")

# load functions from the training script for the finetuning of the model
from train import Load_Dataloader, overall_generator_pass, overall_discriminator_pass, meta_update_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def overall_generator_pass_test(generator: rGAN.Generator, discriminator, img: list[torch.Tensor], gt: torch.Tensor, valid) -> (list[torch.Tensor], float):
    # print(len(img), gt.shape)
    recon_batch = generator(img)
    msssim, f1, psnr = loss_function(recon_batch, gt)
    # print(msssim, f1, psnr)
    
    imgs = recon_batch.data.cpu().numpy()[0, :]
    imgs = roll_axis(imgs)
    
    return imgs, psnr


"""TEST SCRIPT"""
def main(k_shots: int, num_tasks: int, adam_betas: float, gen_lr: float, dis_lr:float, model_folder_path: str) -> None:
    torch.manual_seed(1)
    batch_size = 1
    
    generator = Generator(batch_size=batch_size) 
    discriminator = Discriminator()
    generator.to(device)
    discriminator.to(device)
    # generator.cuda()
    # discriminator.cuda()

    # finetune the Model
    # optimizer
    optimizer_G = optim.Adam (generator.parameters(), lr= gen_lr,  betas=adam_betas)
    optimizer_D = optim.Adam (discriminator.parameters(), lr= dis_lr,  betas=adam_betas)   

    # define dataloader
    tf = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

    create_folder(model_folder_path)
    generator_path = os.path.join(model_folder_path, str.format("Generator_Final.pt"))
    discriminator_path = os.path.join(model_folder_path, str.format("Discriminator_Final.pt"))
    # for saving the fine-tuned model
    generator_path_finetune = os.path.join(model_folder_path, str.format("Generator_finetuned.pt"))
    discriminator_path_finetune = os.path.join(model_folder_path, str.format("Discriminator_finetuned.pt"))
    
    # load the pre-trained model
    print('- start loading pre-trained model')
    # generator.load_state_dict(torch.load(generator_path))
    # discriminator.load_state_dict(torch.load(discriminator_path))
    # if you use CPU
    generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=torch.device('cpu')))
    print('- loading pretrained model done')
    
    # this path must be the video frames for testing purposes
    # Lei uses the fake dataset as an example here
    # frame_path = '/Users/wanglei/Desktop/mnist/'
    # frame_path = '/g/data/cp23/lw4988/few-shot-anomaly-detection/test_dataset/'
    frame_path = '/Users/leiwang/Desktop/fsl_AD-main/test_dataset/'
    # the test dataloader
    train_path_list = createEpochDataTest(frame_path, num_tasks, k_shots)
    # print(frame_path)
    train_dataloader = Load_Dataloader(train_path_list, tf, batch_size)
    # the loop here is just for later extension
    for _, epoch_of_tasks in enumerate(train_dataloader):
            
        # Create folder for saving the images/results
        epoch_results = 'results'# .format(epoch+1)
        create_folder(epoch_results)

        gen_epoch_grads = []
        dis_epoch_grads = []

        

        # Meta-Training
        for tidx, task in enumerate(epoch_of_tasks):
            # Copy rGAN
            
            # print("Memory Allocated: ",torch.cuda.memory_allocated()/1e9)
            # print("Memory Allocated: ",torch.memory_allocated()/1e9)
            
            inner_optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
            inner_optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
            
            for kidx, frame_sequence in enumerate(task[0:3]):
                for ii in range(30): 
                    # Configure input
                    img = frame_sequence[0]
                    print(len(img), ' ---')
                    # print('the len. of img: ', len(img), '------ finetuning Lei') 
                    gt = frame_sequence[1]

                    print('the len. of gt: ', len(gt), '------ gt lei')
                
                    img, gt, valid, fake = prep_data(img, gt)

                    # Train Generator
                    inner_optimizer_G.zero_grad()
                    imgs, g_loss, recon_batch, loss, msssim = overall_generator_pass(generator, discriminator, img, gt, valid)
                    img_path = os.path.join(epoch_results,'{}-te-fig-train{}.png'.format(ii+1, kidx+1))
                    # imsave(img_path , imgs)
                    
                    # imgs = imgs.astype(np.uint8)
                    # imgs = (imgs-np.min(imgs))/(np.max(imgs) - np.min(imgs))
                    imgs = cv2.normalize(imgs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    # imageio.imwrite(img_path , img_as_ubyte(imgs))
                    imsave(img_path , imgs)
                
                    # print(imgs.shape, '-----------')
                
                    g_loss.backward()
                    inner_optimizer_G.step()
                    
                    # Train Discriminator
                    inner_optimizer_D.zero_grad()
                    # Measure discriminator's ability to classify real from generated samples
                    d_loss = overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake)
                    d_loss.backward()
                    inner_optimizer_D.step()
                    print (kidx, ' | ', ii, ' - Reconstruction_Loss: {:.4f}, G_Loss: {:.4f}, D_loss: {:.4f},  msssim:{:.4f} '.format(loss.item(), g_loss, d_loss, msssim))
                    # break
        torch.save(generator.state_dict(), generator_path_finetune)
        torch.save(discriminator.state_dict(), discriminator_path_finetune)

        print('model finetuning done! Now applying in testing...')

    # Meta-Validation
    print ('\n Meta Validation/Test \n')
    
    # forward pass
   

    # ----------------------- the following codes are for the meta-validation/meta test--------
    s_list = []
    scoreset = []
    with torch.no_grad():
        generator.eval()
        discriminator.eval()
        
        for tidx, task in enumerate(epoch_of_tasks):
            # the len of task = k_shots x num_tasks x 2 (one for training and another for testing as the author states in utils.py)
            # print('tidx: ', tidx, ' | length: ', len(task))
            for vidx, val_frame_sequence in enumerate(task[-k_shots:]):
                # print(' === vidx: ', vidx, ' | val_frame_seq: ', len(val_frame_sequence))
                # if vidx == 0:
                #     dummy_frame_sequence = val_frame_sequence
                        
                img = val_frame_sequence[0]
                        
                gt = val_frame_sequence[1]
                img, gt, valid, fake = prep_data(img, gt)
                # len of img is 3 (input is a 3-frame sequence)
                # print('test --- len of img: ', len(img))                 
                # print('test --- len of gt: ', len(gt))
                # k-Validation Generator
                imgs, psnr = overall_generator_pass_test(generator, discriminator, img, gt, valid)
                img_path = os.path.join(epoch_results,'{}-te-fig-val{}.png'.format(tidx+1, vidx+1))
                # imsave(img_path , imgs)
                        
                # imgs = imgs.astype(np.uint8)
                # imgs = (imgs-np.min(imgs))/(np.max(imgs) - np.min(imgs))
                imgs = cv2.normalize(imgs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # imageio.imwrite(img_path , img_as_ubyte(imgs))
                imsave(img_path , imgs)

                s_list.append(psnr)

                maxPSNR = max(s_list)
                minPSNR = min(s_list)

                print('maxPSNR: ', maxPSNR, ' | minPSNR: ', minPSNR, ' | psnr: ', psnr)

                if maxPSNR == minPSNR:
                    print('min PSNR = max PSNR')
                    continue;
                else:
                    score = (psnr - minPSNR) / (maxPSNR - minPSNR)
                    print('normalized PSNR as the anomaly score: ', score)
                
                scoreset.append(score)
                        
                if score > 0.6:
                    print('Anomaly!')
                else:
                    print('Not Anomaly!')
    scorenp = np.array(scoreset)
    with open('test.npy', 'wb') as f:
        np.save(f, scorenp)

    print('anomaly scores saved.')


if __name__ == "__main__":
    if (len(sys.argv) == 8):
        """SYS ARG ORDER: 
        K_shots, num_tasks, adam_betas, generator lr, discriminator lr, total epochs, save model path
        """
        k_shots = int(sys.argv[1])
        num_tasks =  int(sys.argv[2])
        adam_betas = ast.literal_eval(sys.argv[3])
        gen_lr = float(sys.argv[4])
        dis_lr = float(sys.argv[5])
        model_folder_path = sys.argv[7]
    else:
        k_shots = 90
        num_tasks = 1
        adam_betas = (0.5, 0.999)
        gen_lr = 2e-4
        dis_lr = 1e-5
        model_folder_path = "model"
    main(k_shots, num_tasks, adam_betas, gen_lr, dis_lr, model_folder_path)
