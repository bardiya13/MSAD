def folder_to_list(folder_path, reference_list_file, output_list_file):
    # Read reference folder names (without paths)
    with open(reference_list_file, 'r') as ref_file:
        ref_folders = [line.strip() for line in ref_file]
    ref_folders = [os.path.basename(line).replace('_i3d.npy', '') for line in ref_folders]
    # print("Processed reference folder names:", ref_folders, "\n")

    # Get all folder paths in the given directory
    folder_paths = {}
    for root, subdirs, _ in os.walk(folder_path):
        # print(f"Root: {root}, Subdirs: {subdirs}\n")
        for subdir in subdirs:
            if subdir in ref_folders:
                # print(f"Match found: {subdir}\n")
                folder_paths[subdir] = os.path.join(root, subdir)

    # Order folder paths based on reference list
    ordered_folders = [folder_paths[folder] for folder in ref_folders if folder in folder_paths]

    # Write the ordered folder paths to the output .list file
    with open(output_list_file, 'w') as f:
        f.write('\n'.join(ordered_folders) + '\n')
    print(f"List file created at '{output_list_file}' with {len(ordered_folders)} entries.")


folder_path = '/home/mnafez/msad_dataset/test_frames'
reference_list_file = '/home/mnafez/RTFM/MSAD/RTFM/list/msad-i3d-test.list'
output_list_file = './msad-frame-test.list'
folder_to_list(folder_path, reference_list_file, output_list_file)

import os
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self,
                 order_list="./msad-frame-test.list",
                 frames_dir="/home/mnafez/msad_dataset/test_frames",
                 gt_dir="/home/mnafez/msad_dataset/gt_new.npy"):
        self.order_list = order_list
        self.frames_dir = frames_dir
        self.gt_dir = gt_dir

        self.video_list = [line.strip() for line in open(self.order_list)]
        self.gt_data = np.load(self.gt_dir, allow_pickle=True)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        # Get the video ID from the list
        video_id = self.video_list[index]
        video_folder = os.path.join(self.frames_dir, video_id)

        # Get sorted list of frame files
        frame_files = natsorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])

        # Get full paths of the frames
        frame_paths = [os.path.join(video_folder, frame_file) for frame_file in frame_files]

        # Get the corresponding labels (same length as frame_files)
        start_idx = sum(((len(natsorted(os.listdir(os.path.join(self.frames_dir, v))))) // 16) * 16
                        for v in self.video_list[:index])
        end_idx = start_idx + (len(frame_files) // 16) * 16
        labels = self.gt_data[start_idx:end_idx]

        return frame_paths, labels