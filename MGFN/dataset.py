# #######################
# import torch.utils.data as data
# import numpy as np
# from utils.utils import process_feat
# import torch
# import os
#
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# import option
#
# args = option.parse_args()
#
#
# class Dataset(data.Dataset):
#     def __init__(self, args, is_normal=True, transform=None, test_mode=False, is_preprocessed=False, shangatic=False):
#
#         self.modality = args.modality
#         self.is_normal = is_normal
#         self.shangatic = shangatic
#         if shangatic:
#             if test_mode:
#                 self.feature_address = args.test_feature_address
#                 self.label_dir = args.test_label_address
#             else:
#                 self.feature_address = args.train_feature_address
#                 self.label_dir = args.train_label_address
#
#
#
#         if test_mode:
#             self.rgb_list_file = args.test_rgb_list
#         else:
#             self.rgb_list_file = args.rgb_list
#         self.tranform = transform
#         self.test_mode = test_mode
#         self._parse_list()
#         self.num_frame = 0
#         self.labels = None
#         self.is_preprocessed = args.preprocessed
#
#     def _parse_list(self):
#         self.list = list(open(self.rgb_list_file))
#         self.list = [x.strip("\n").strip() for x in self.list]
#         # self.frame_indices = [os.path.splitext(os.path.basename(item.strip()))[0] for item in self.list]
#         if self.test_mode is False:
#             if args.datasetname == 'UCF':
#                 if self.is_normal:
#                     self.list = self.list[810:]  # ucf 810; sht63; xd 9525
#                     # print('normal list')
#                     # print(self.list)
#                 else:
#                     self.list = self.list[:810]  # ucf 810; sht 63; 9525
#                     # print('abnormal list')
#                     # print(self.list)
# #
#             if args.datasetname == 'MSAD':
#                 if self.is_normal:
#                     self.list = self.list[120:]
#                     # print('normal list')
#                     # print(self.list)
#                 else:
#                     self.list = self.list[:120]
#                     # print('abnormal list')
#                     # print(self.list)
#
#             elif args.datasetname == 'XD':
#                 if self.is_normal:
#                     self.list = self.list[9525:]
#                     # print('normal list')
#                     # print(self.list)
#                 else:
#                     self.list = self.list[:9525]
#                     # print('abnormal list')
#                     # print(self.list)
#
#             elif args.datasetname == 'SH':
#                 if self.is_normal:
#                     self.list = [self.list[i] for i in range(len(self.list))
#                                  if self.get_label(i) == 0]
#
#                 else:
#                     self.list = [self.list[i] for i in range(len(self.list))
#                                  if self.get_label(i) == 1]
#
#
#
#     def __getitem__(self, index):
#         label = self.get_label(index)  # get video level label 0/1
#         if args.datasetname == 'UCF':
#             features = np.load(self.list[index].strip('\n'), allow_pickle=True)
#             features = np.array(features, dtype=np.float32)
#             name = self.list[index].split('/')[-1].strip('\n')[:-4]
#         elif args.datasetname == 'XD':
#             features = np.load(self.list[index].strip('\n'), allow_pickle=True)
#             features = np.array(features, dtype=np.float32)
#             name = self.list[index].split('/')[-1].strip('\n')[:-4]
#
#         elif args.datasetname == 'MSAD' or args.datasetname == 'CUHK':
#             features = np.load(self.list[index].strip('\n'), allow_pickle=True)
#             features = np.array(features, dtype=np.float32)
#             name = self.list[index].split('/')[-1].strip('\n')[:-4]
#
#         elif args.datasetname == 'Ped2':
#             features = np.load(self.list[index].strip('\n'), allow_pickle=True)
#             features = np.array(features, dtype=np.float32)
#             name = self.list[index].split('/')[-1].strip('\n')[:-4]
#
#         elif args.datasetname == 'SH':
#
#             features = np.load(os.path.join(self.feature_address, self.list[index].strip('\n')+".npy"), allow_pickle=True)
#             features = np.array(features, dtype=np.float32)
#             name = self.list[index].split('/')[-1].strip('\n')[:-4]
#
#         if self.tranform is not None:
#             features = self.tranform(features)
#         if self.test_mode:
#             if args.datasetname == 'UCF':
#                 # ------------ I3D --------------
#                 # mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
#                 # features = np.concatenate((features,mag),axis = 2)
#
#                 # ------------ Swin ------------
#                 mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
#                 features = np.concatenate((features, mag), axis=1)
#
#             elif args.datasetname == 'MSAD' or args.datasetname == 'CUHK':
#                 # ------------ I3D --------------
#                 mag = np.linalg.norm(features, axis=2)[:, :, np.newaxis]
#                 features = np.concatenate((features, mag), axis=2)
#
#                 # ------------ Swin ------------
#                 # mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
#                 # features = np.concatenate((features, mag), axis=1)
#
#             elif args.datasetname == 'XD':
#                 mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
#                 features = np.concatenate((features, mag), axis=1)
#             elif args.datasetname == 'Ped2':
#                 mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
#                 features = np.concatenate((features, mag), axis=1)
#             elif args.datasetname == 'SH':
#                 mag = np.linalg.norm(features, axis=2)[:, :, np.newaxis]
#                 features = np.concatenate((features, mag), axis=2)
#             return features, name
#         else:
#             if args.datasetname == 'UCF':
#                 if self.is_preprocessed:
#                     return features, label
#                 # ------------------------ for I3D ------------------------------
#                 # features = features.transpose(1, 0, 2)  # [10, T, F]
#                 # divided_features = []
#
#                 # divided_mag = []
#                 # for feature in features:
#                 #     feature = process_feat(feature, args.seg_length) #ucf(32,2048)
#                 #     divided_features.append(feature)
#                 #     divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
#                 # divided_features = np.array(divided_features, dtype=np.float32)
#                 # divided_mag = np.array(divided_mag, dtype=np.float32)
#                 # divided_features = np.concatenate((divided_features,divided_mag),axis = 2)
#                 # return divided_features, label
#
#                 # ------------------------ for Swin --------------------------------
#
#                 feature = process_feat(features, 32)
#                 # if args.add_mag_info == True:
#                 feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
#                 feature = np.concatenate((feature, feature_mag), axis=1)
#                 return feature, label
#
#             if args.datasetname == 'MSAD':
#                 if self.is_preprocessed:
#                     return features, label
#                 # ------------------------ for I3D ------------------------------
#                 features = features.transpose(1, 0, 2)  # [10, T, F]
#                 divided_features = []
#
#                 divided_mag = []
#                 for feature in features:
#                     feature = process_feat(feature, args.seg_length)  # ucf(32,2048)
#                     divided_features.append(feature)
#                     divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
#                 divided_features = np.array(divided_features, dtype=np.float32)
#                 divided_mag = np.array(divided_mag, dtype=np.float32)
#                 divided_features = np.concatenate((divided_features, divided_mag), axis=2)
#                 return divided_features, label
#
#                 # ------------------------ for Swin --------------------------------
#
#                 # feature = process_feat(features, 32)
#                 # # if args.add_mag_info == True:
#                 # feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
#                 # feature = np.concatenate((feature,feature_mag),axis = 1)
#                 # return feature, label
#
#
#
#             elif args.datasetname == 'XD':
#                 feature = process_feat(features, 32)
#                 if args.add_mag_info == True:
#                     feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
#                     feature = np.concatenate((feature, feature_mag), axis=1)
#                 return feature, label
#
#             if args.datasetname == 'SH':
#                 if self.is_preprocessed:
#                     return features, label
#                 features = features.transpose(1, 0, 2)  # [10, T, F]
#                 divided_features = []
#
#                 divided_mag = []
#                 for feature in features:
#                     feature = process_feat(feature, args.seg_length)
#                     divided_features.append(feature)
#                     divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
#                 divided_features = np.array(divided_features, dtype=np.float32)
#                 divided_mag = np.array(divided_mag, dtype=np.float32)
#                 divided_features = np.concatenate((divided_features, divided_mag), axis=2)
#                 return divided_features, label
#
#     def get_label(self, index):
#         if self.shangatic:
#             # Load the label file for the current frame
#
#             label_path = os.path.join(self.label_dir, f"{self.list[index]}_labels.npy")
#
#             # Load the numpy array
#             label_array = np.load(label_path)
#
#             # Return 1 if sum > 0, else 0
#             if np.sum(label_array) > 0:
#                 return torch.tensor(1.0)
#             else:
#                 return torch.tensor(0.0)
#         else:
#             # Original logic for non-shangatic mode
#             if self.is_normal:
#                 # label[0] = 1
#                 label = torch.tensor(0.0)
#             else:
#                 label = torch.tensor(1.0)
#                 # label[1] = 1
#             return label
#
#     def __len__(self):
#         return len(self.list)
#
#     def get_num_frames(self):
#         return self.num_frame
# ##################################
import torch.utils.data as data
import numpy as np
from utils.utils import process_feat
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')
import option

args = option.parse_args()


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, is_preprocessed=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.is_preprocessed = args.preprocessed

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if args.datasetname == 'UCF':
                if self.is_normal:
                    self.list = self.list[180:]  # ucf 810; sht63; xd 9525
                    # print('normal list')
                    # print(self.list)
                else:
                    self.list = self.list[:180]  # ucf 810; sht 63; 9525
                    # print('abnormal list')
                    # print(self.list)

            if args.datasetname == 'MSAD':
                if self.is_normal:
                    self.list = self.list[120:]
                    # print('normal list')
                    # print(self.list)
                else:
                    self.list = self.list[:120]
                    # print('abnormal list')
                    # print(self.list)

            elif args.datasetname == 'XD':
                if self.is_normal:
                    self.list = self.list[9525:]
                    # print('normal list')
                    # print(self.list)
                else:
                    self.list = self.list[:9525]
                    # print('abnormal list')
                    # print(self.list)

            elif args.datasetname == 'SH':
                if self.is_normal:
                    self.list = self.list[63:]
                    # print('normal list')
                    # print(self.list)
                else:
                    self.list = self.list[:63]
                    # print('abnormal list')
                    # print(self.list)

    def __getitem__(self, index):
        label = self.get_label(index)

        file_path = (self.list[index].strip('\n'))

        if self.test_mode is False:
            file_path = "/kaggle/input/tad-train-feauter/feauter_train/" + file_path
        else:

            file_path = "/kaggle/input/tad-feauter-test-1/output_folder/" + file_path
        features = np.load(file_path, allow_pickle=True)
        features = np.array(features, dtype=np.float32)





        # get video level label 0/1
        if args.datasetname == 'UCF':
            if self.test_mode is False:
                file_path = "/kaggle/input/tad-train-feauter/feauter_train/" + file_path
            else:

                file_path = "/kaggle/input/tad-feauter-test-1/output_folder/" + file_path
            features = np.load(file_path, allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]
        elif args.datasetname == 'XD':
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]

        elif args.datasetname == 'MSAD' or args.datasetname == 'CUHK':

            if "MSAD-I3D-abnormal-testing" in self.list[index].strip('\n'):
                features = np.load(
                    self.list[index].strip('\n').replace('/scratch/kf09/lz1278/MSAD-I3D-WS/', '/kaggle/input/abnrmal-msad-test/'),
                    allow_pickle=True)


            # features = self.list[index].strip('\n').replace('/scratch/kf09/lz1278/MSAD-I3D-WS/', '/kaggle/input/')
            else:
                features = np.load(
                    self.list[index].strip('\n').replace('/scratch/kf09/lz1278/MSAD-I3D-WS/', '/kaggle/input/msad-normal-test/'))
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]

        elif args.datasetname == 'Ped2':

            features = np.load(
                self.list[index].strip('\n').replace('/scratch/kf09/lz1278/TEVAD/save/UCSDped2/ped2_ten_crop_i3d/', '/kaggle/working/featur_zip_all_in TEST_p2/'),
                allow_pickle=True)
#/kaggle/working/featur_zip_all_in TEST_p2
#/kaggle/input/ooooooo/featur_zip_all_in TEST_p2
#/scratch/kf09/lz1278/TEVAD/save/UCSDped2/ped2_ten_crop_i3d/Test003_i3d.npy

            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]

        elif args.datasetname == 'SH':
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            if args.datasetname == 'UCF':
                # ------------ I3D --------------
                # mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
                # features = np.concatenate((features,mag),axis = 2)

                # ------------ Swin ------------
                mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
                features = np.concatenate((features, mag), axis=1)

            elif args.datasetname == 'MSAD' or args.datasetname == 'CUHK':
                # ------------ I3D --------------
                mag = np.linalg.norm(features, axis=2)[:, :, np.newaxis]
                features = np.concatenate((features, mag), axis=2)

                # ------------ Swin ------------
                # mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
                # features = np.concatenate((features, mag), axis=1)

            elif args.datasetname == 'XD':
                mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
                features = np.concatenate((features, mag), axis=1)
            elif args.datasetname == 'Ped2':
                mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
                features = np.concatenate((features, mag), axis=1)
            elif args.datasetname == 'SH':
                mag = np.linalg.norm(features, axis=2)[:, :, np.newaxis]
                features = np.concatenate((features, mag), axis=2)
            return features, name
        else:
            if args.datasetname == 'UCF':
                if self.is_preprocessed:
                    return features, label
                # ------------------------ for I3D ------------------------------
                # features = features.transpose(1, 0, 2)  # [10, T, F]
                # divided_features = []

                # divided_mag = []
                # for feature in features:
                #     feature = process_feat(feature, args.seg_length) #ucf(32,2048)
                #     divided_features.append(feature)
                #     divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                # divided_features = np.array(divided_features, dtype=np.float32)
                # divided_mag = np.array(divided_mag, dtype=np.float32)
                # divided_features = np.concatenate((divided_features,divided_mag),axis = 2)
                # return divided_features, label

                # ------------------------ for Swin --------------------------------

                feature = process_feat(features, 32)
                # if args.add_mag_info == True:
                feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                feature = np.concatenate((feature, feature_mag), axis=1)
                return feature, label

            if args.datasetname == 'MSAD':
                if self.is_preprocessed:
                    return features, label
                # ------------------------ for I3D ------------------------------
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []

                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, args.seg_length)  # ucf(32,2048)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate((divided_features, divided_mag), axis=2)
                return divided_features, label

                # ------------------------ for Swin --------------------------------

                # feature = process_feat(features, 32)
                # # if args.add_mag_info == True:
                # feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                # feature = np.concatenate((feature,feature_mag),axis = 1)
                # return feature, label



            elif args.datasetname == 'XD':
                feature = process_feat(features, 32)
                if args.add_mag_info == True:
                    feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                    feature = np.concatenate((feature, feature_mag), axis=1)
                return feature, label

            if args.datasetname == 'SH':
                if self.is_preprocessed:
                    return features, label
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []

                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, args.seg_length)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate((divided_features, divided_mag), axis=2)
                return divided_features, label

    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def __len__(self):

        return len(self.list)

    def get_num_frames(self):
        return self.num_frame