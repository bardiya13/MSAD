import os
import random
import torch
import pytorch_msssim
import numpy as np
from math import log10
from torch.nn import functional as F
# from torch.cuda import FloatTensor as Tensor
from torch import Tensor

from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_k_shot_frames(video_folder:str, k_shots: int) -> (str, list[list]):
    k_video_sequences = []

    all_frames = list(os.walk(video_folder))[0][2]

    all_frames = sorted(all_frames, key=lambda x: int(x.replace('out', '').replace('.jpg', '')))
    video_length = len(all_frames)

    frame_samples = random.sample(range(3,video_length), k_shots * 2) # first k are meta-training, rest are meta-testing
    # this frame_samples are the last index of the 4-frame video clip
    # [444,...]
    # print(frame_samples, ' --- frame_samples')

    k_frame_sequences = [[all_frames[v_index - before] for before in reversed(range(0,4))] for v_index in frame_samples]
    # the k frame sequence is a list of images in the form of, e.g., 
    # [['441.jpg', '442.jpg', '443.jpg', '444.jpg'],...]
    # print('k_frame_seq: ', k_frame_sequences)
    return video_folder, k_frame_sequences

def generate_k_shot_frames_test(video_folder: str, k_shots: int) -> (str, list[list]):
    k_video_sequences = []

    all_frames = list(os.walk(video_folder))[0][2]

    all_frames = sorted(all_frames, key=lambda x: int(x.replace('out', '').replace('.jpg', '')))
    video_length = len(all_frames)

    # frame_samples = random.sample(range(3,video_length), k_shots * 2) # first k are meta-training, rest are meta-testing
    frame_samples = [x for x in range(3, video_length)][:k_shots * 2]
    # this frame_samples are the last index of the 4-frame video clip
    # [444,...]
    # print(frame_samples, ' --- frame_samples')

    k_frame_sequences = [[all_frames[v_index - before] for before in reversed(range(0,4))] for v_index in frame_samples]
    # the k frame sequence is a list of images in the form of, e.g., 
    # [['441.jpg', '442.jpg', '443.jpg', '444.jpg'],...]
    # print('k_frame_seq: ', k_frame_sequences)
    return video_folder, k_frame_sequences

def createEpochData(frame_path: str, numTasks:int, k_shots:int) -> list[str]:
    dirs = os.listdir(frame_path)
    dirs = [d for d in dirs if d != '.DS_Store']  # Exclude '.DS_Store'
    dirs.sort(key=int)
    
    # Selected Tasks (videos that are being used)
    selected_videos = []

    for task in range(numTasks):
        print(numTasks)
        print(task)
        sample = random.sample(list(os.listdir(os.path.join(frame_path, dirs[task]))), 1)
        print(sample)
        selected_videos.append(os.path.join(frame_path, dirs[task], sample[0]))

    print('----------- selected videos: ', selected_videos)

    train_path_list = []

    # task_order = [0, 2, 3, 4, 7, 12]
    train_curr_paths = []
    # for task in range(len(task_order)):
    for task in range(numTasks):
        video = selected_videos[task]
        video_folder, k_shot_frames = generate_k_shot_frames(video, k_shots)        
        train_curr_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in k_shot_frames])
    train_path_list.append(train_curr_paths)

    return train_path_list

def createEpochDataTest(frame_path: str, numTasks: int, k_shots:int) -> list[str]:
    dirs = os.listdir(frame_path)
    dirs = [d for d in dirs if d != '.DS_Store']  # Exclude '.DS_Store'
    dirs.sort(key=int)

    # Selected Tasks (videos that are being used)
    selected_videos = []

    for task in range(numTasks):
        sample = random.sample(list(os.listdir(os.path.join(frame_path, dirs[task]))), 1)
        selected_videos.append(os.path.join(frame_path, dirs[task]))   # change here to adjust the structure of testing

    print('----------- selected videos: ', selected_videos)

    train_path_list = []

    # task_order = [0, 2, 3, 4, 7, 12]
    train_curr_paths = []
    # for task in range(len(task_order)):
    for task in range(numTasks):
        video = selected_videos[task]
        video_folder, k_shot_frames = generate_k_shot_frames_test(video, k_shots)
        train_curr_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in k_shot_frames])
    train_path_list.append(train_curr_paths)

    return train_path_list


def loss_function(recon_x:torch.Tensor, x:torch.Tensor) -> (torch.Tensor, torch.Tensor, float):
    msssim = ((1-pytorch_msssim.msssim(x,recon_x)))/2
    f1 =  F.l1_loss(recon_x, x)
    # psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_batch))))))
    # psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_x))))))
    # Lei's psnr
    # recon_x = (recon_x - recon_x.min()) / (recon_x.max() - recon_x.min())
    # print(recon_x.min(), ' | ', recon_x.max(), ' ====== ', x.min(), ' | ', x.max(),)
    
    psnr_error = 10 * log10(torch.pow((torch.max(recon_x)), 2)/(1/(256*256) * torch.sum(torch.pow((x-recon_x), 2))))
    # print(psnr_error, '------')

    # print(type(msssim), type(f1), type(psnr_error))

    return msssim, f1, psnr_error

def roll_axis(img: np.ndarray) -> np.ndarray:
    img = np.rollaxis(img, -1, 0)
    img = np.rollaxis(img, -1, 0)
    return img

def create_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

def prep_data(img: list[torch.Tensor], gt: torch.Tensor, gen_labels=True) -> (list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor):
    if gen_labels:
        # Adversarial ground truths
        valid = Variable(Tensor(1, 1).fill_(0.9), requires_grad=False)
        fake = Variable(Tensor(1, 1).fill_(0.1), requires_grad=False)
        # valid.cuda()
        # fake.cuda()
        valid.to(device)
        fake.to(device)
    # print(len(img), ' ======')
    for x in range(len(img)):
        # img[x] = Variable(img[x].cuda())
        img[x] = Variable(img[x].to(device))
        # print(img[x].shape, 'xxx')
    # gt = Variable(gt.cuda())
    gt = Variable(gt.to(device))
    return img, gt, valid, fake

