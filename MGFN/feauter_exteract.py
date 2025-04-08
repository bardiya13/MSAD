import os
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 2 (index 1)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrozenBN(nn.Module):
	def __init__(self, num_channels, momentum=0.1, eps=1e-5):
		super(FrozenBN, self).__init__()
		self.num_channels = num_channels
		self.momentum = momentum
		self.eps = eps
		self.params_set = False

	def set_params(self, scale, bias, running_mean, running_var):
		self.register_buffer('scale', scale)
		self.register_buffer('bias', bias)
		self.register_buffer('running_mean', running_mean)
		self.register_buffer('running_var', running_var)
		self.params_set = True

	def forward(self, x):
		assert self.params_set, 'model.set_params(...) must be called before the forward pass'
		return torch.batch_norm(x, self.scale, self.bias, self.running_mean, self.running_var, False, self.momentum, self.eps, torch.backends.cudnn.enabled)

	def __repr__(self):
		return 'FrozenBN(%d)'%self.num_channels

def freeze_bn(m, name):
	for attr_str in dir(m):
		target_attr = getattr(m, attr_str)
		if type(target_attr) == torch.nn.BatchNorm3d:
			frozen_bn = FrozenBN(target_attr.num_features, target_attr.momentum, target_attr.eps)
			frozen_bn.set_params(target_attr.weight.data, target_attr.bias.data, target_attr.running_mean, target_attr.running_var)
			setattr(m, attr_str, frozen_bn)
	for n, ch in m.named_children():
		freeze_bn(ch, n)

#-----------------------------------------------------------------------------------------------#

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm3d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		outplanes = planes * 4
		self.nl = NonLocalBlock(outplanes, outplanes, outplanes//2) if use_nl else None

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		if self.nl is not None:
			out = self.nl(out)

		return out

class NonLocalBlock(nn.Module):
	def __init__(self, dim_in, dim_out, dim_inner):
		super(NonLocalBlock, self).__init__()

		self.dim_in = dim_in
		self.dim_inner = dim_inner
		self.dim_out = dim_out

		self.theta = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
		self.phi = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		self.g = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

		self.out = nn.Conv3d(dim_inner, dim_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		self.bn = nn.BatchNorm3d(dim_out)

	def forward(self, x):
		residual = x

		batch_size = x.shape[0]
		mp = self.maxpool(x)
		theta = self.theta(x)
		phi = self.phi(mp)
		g = self.g(mp)

		theta_shape_5d = theta.shape
		theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view(batch_size, self.dim_inner, -1)

		theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
		theta_phi_sc = theta_phi * (self.dim_inner**-.5)
		p = F.softmax(theta_phi_sc, dim=-1)

		t = torch.bmm(g, p.transpose(1, 2))
		t = t.view(theta_shape_5d)

		out = self.out(t)
		out = self.bn(out)

		out = out + residual
		return out

#-----------------------------------------------------------------------------------------------#

class I3Res50(nn.Module):

	def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False):
		self.inplanes = 64
		super(I3Res50, self).__init__()
		self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
		self.bn1 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
		self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

		nonlocal_mod = 2 if use_nl else 1000
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.drop = nn.Dropout(0.5)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0]!=1:
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
				nn.BatchNorm3d(planes * block.expansion)
				)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i], i%nonlocal_mod==nonlocal_mod-1))

		return nn.Sequential(*layers)

	def forward_single(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool1(x)

		x = self.layer1(x)
		x = self.maxpool2(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		return x

	def forward(self, batch):
		if batch['frames'].dim() == 5:
			feat = self.forward_single(batch['frames'])
		return feat

#-----------------------------------------------------------------------------------------------#

def i3_res50(num_classes, pretrainedpath):
		net = I3Res50(num_classes=num_classes, use_nl=False)
		state_dict = torch.load(pretrainedpath, weights_only=True)
		net.load_state_dict(state_dict)
		print("Received Pretrained model..")
		# freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
		return net

def i3_res50_nl(num_classes, pretrainedpath):
		net = I3Res50(num_classes=num_classes, use_nl=True)
		state_dict = torch.load(pretrainedpath, weights_only=False)
		net.load_state_dict(state_dict)
		# freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
		return net

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import torch
import numpy as np
from PIL import Image
from natsort import natsorted
from torch.autograd import Variable

def load_frame(frame_file):
    data = Image.open(frame_file)
    data = data.resize((340, 256), Image.Resampling.LANCZOS)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)
    return data

def load_rgb_batch(rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))  # Shape: (batch_size, chunk_size, 256, 340, 3)
    for i in range(frame_indices.shape[0]):  # Iterate over batches
        for j in range(frame_indices.shape[1]):  # Iterate over frames in each batch
            batch_data[i, j, :, :, :] = load_frame(rgb_files[frame_indices[i][j]])  # Use full paths
    return batch_data

def oversample_data(data):
    # 19, 16, 256, 340, 3
    data_flip = np.array(data[:,:,:,::-1,:])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5,
      data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]
##########################
# I3D Model Weight Conversion
# This notebook converts Caffe2 weights to PyTorch format for I3D ResNet-50 model

import pickle
import torch
import re
import sys
import os

# Parameters (you can modify these directly in the notebook)
# Uncomment this line to use command-line arguments when running as a script
# c2_weights = sys.argv[1] if len(sys.argv) > 1 else 'pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl'
# pth_weights_out = sys.argv[2] if len(sys.argv) > 2 else 'pretrained/i3d_r50_kinetics.pth'

# For notebook use, define the paths directly
c2_weights = '/kaggle/working/pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl'
pth_weights_out = "/kaggle/working/pretrained/i3d_r50_kinetics.pth"

# Check if input file exists
if not os.path.exists(c2_weights):
    raise FileNotFoundError(f"Input weights file not found: {c2_weights}")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(pth_weights_out), exist_ok=True)

# Load Caffe2 weights
print(f"Loading Caffe2 weights from: {c2_weights}")
c2 = pickle.load(open(c2_weights, 'rb'), encoding='latin')['blobs']
c2 = {k:v for k,v in c2.items() if 'momentum' not in k}
print(f"Loaded {len(c2)} weight blobs")

# Regular expressions for matching layer names
downsample_pat = re.compile('res(.)_(.)_branch1_.*')
conv_pat = re.compile('res(.)_(.)_branch2(.)_.*')
nl_pat = re.compile('nonlocal_conv(.)_(.)_(.*)_.*')

# Mapping from module letters to numbers
m2num = dict(zip('abc',[1,2,3]))
suffix_dict = {'b':'bias', 'w':'weight', 's':'weight', 'rm':'running_mean', 'riv':'running_var'}

# Initialize key mapping dictionary
key_map = {}
key_map.update({
    'conv1.weight':'conv1_w',
    'bn1.weight':'res_conv1_bn_s',
    'bn1.bias':'res_conv1_bn_b',
    'bn1.running_mean':'res_conv1_bn_rm',
    'bn1.running_var':'res_conv1_bn_riv',
    'fc.weight':'pred_w',
    'fc.bias':'pred_b',
})

# Process regular convolution layers
for key in c2:
    # Match convolution layers
    conv_match = conv_pat.match(key)
    if conv_match:
        layer, block, module = conv_match.groups()
        layer, block, module = int(layer), int(block), m2num[module]
        name = 'bn' if 'bn_' in key else 'conv'
        suffix = suffix_dict[key.split('_')[-1]]
        new_key = f'layer{layer-1}.{block}.{name}{module}.{suffix}'
        key_map[new_key] = key

    # Match downsample layers
    ds_match = downsample_pat.match(key)
    if ds_match:
        layer, block = ds_match.groups()
        layer, block = int(layer), int(block)
        module = 0 if key[-1]=='w' else 1
        name = 'downsample'
        suffix = suffix_dict[key.split('_')[-1]]
        new_key = f'layer{layer-1}.{block}.{name}.{module}.{suffix}'
        key_map[new_key] = key

    # Match non-local layers
    nl_match = nl_pat.match(key)
    if nl_match:
        layer, block, module = nl_match.groups()
        layer, block = int(layer), int(block)
        name = f'nl.{module}'
        suffix = suffix_dict[key.split('_')[-1]]
        new_key = f'layer{layer-1}.{block}.{name}.{suffix}'
        key_map[new_key] = key

# Import the model
# Note: Make sure the 'models' directory with resnet.py is in your working directory
print("Loading PyTorch model")
import resnet
pth = resnet.I3Res50(num_classes=400, use_nl=True)
state_dict = pth.state_dict()

# Create new state dictionary with converted weights
print("Converting weights...")
new_state_dict = {key: torch.from_numpy(c2[key_map[key]]) for key in state_dict if key in key_map}

# Save converted weights
print(f"Saving converted weights to: {pth_weights_out}")
torch.save(new_state_dict, pth_weights_out)
torch.save(key_map, pth_weights_out+'.keymap')

# Verify weight dimensions match
print("\nVerifying weight dimensions match:")
print(f"{'Caffe2 Key':<35} --> {'PyTorch Key':<35} | {'Shape':<21}")
print("-" * 95)

mismatch_count = 0
matched_count = 0

for key in state_dict:
    if key not in key_map:
        continue

    c2_v, pth_v = c2[key_map[key]], state_dict[key]
    if str(tuple(c2_v.shape)) == str(tuple(pth_v.shape)):
        print(f"{key_map[key]:<35} --> {key:<35} | {str(tuple(c2_v.shape)):<21}")
        matched_count += 1
    else:
        print(f"{key_map[key]:<35} --> {key:<35} | {str(tuple(c2_v.shape)):<21} != {str(tuple(pth_v.shape))}")
        mismatch_count += 1

print(f"\nConverted {matched_count} weights successfully.")
if mismatch_count > 0:
    print(f"WARNING: Found {mismatch_count} dimension mismatches!")
else:
    print("All dimensions match correctly!")

############################
import os
import time
import torch
import shutil
import argparse
from pathlib import Path

import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import L1Loss
import torch.optim as optim
from torch.nn import MSELoss
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve, precision_recall_curve

torch.set_default_tensor_type('torch.FloatTensor')

###################################
import torch
from torch import nn, einsum
from utils.utils import FeedForward, LayerNorm, GLANCE, FOCUS
from MGFN import option

args = option.parse_args()


def exists(val):
    return val is not None


def attention(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)

    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)

    return out


def MSNSD(features, scores, bs, batch_size, drop_out, ncrops, k):
    # magnitude selection and score prediction

    features = features  # (B*10crop,32,1024)
    bc, t, f = features.size()

    if ncrops == 1:

        scroes = scores
        normal_features = features[0:batch_size]  # [b/2,32,1024]
        normal_scores = scores[0:batch_size]  # [b/2, 32,1]
        abnormal_features = features[batch_size:]
        abnormal_scores = scores[batch_size:]
        feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b, 32]

    elif ncrops == 10:

        scores = scores.view(bs, ncrops, -1).mean(1)  # (B,32)

        scores = scores.unsqueeze(dim=2)  # (B,32,1)

        normal_features = features[0:batch_size * 10]  # [b/2*ten,32,1024]
        normal_scores = scores[0:batch_size]  # [b/2, 32,1]

        abnormal_features = features[batch_size * 10:]
        abnormal_scores = scores[batch_size:]
        feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b*ten,32]

        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [b,32]

    nfea_magnitudes = feat_magnitudes[0:batch_size]  # [b/2,32]  # normal feature magnitudes
    afea_magnitudes = feat_magnitudes[batch_size:]  # abnormal feature magnitudes
    n_size = nfea_magnitudes.shape[0]  # b/2

    if nfea_magnitudes.shape[0] == 1:  # this is for inference

        afea_magnitudes = nfea_magnitudes
        abnormal_scores = normal_scores
        abnormal_features = normal_features

    select_idx = torch.ones_like(nfea_magnitudes).cuda()
    select_idx = drop_out(select_idx)

    afea_magnitudes_drop = afea_magnitudes * select_idx

    idx_abn = torch.topk(afea_magnitudes_drop, k, dim=1)[1]

    idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

    abnormal_features = abnormal_features.view(n_size, ncrops, t, f)

    abnormal_features = abnormal_features.permute(1, 0, 2, 3)

    total_select_abn_feature = torch.zeros(0)

    for i, abnormal_feature in enumerate(abnormal_features):
        feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)

        total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))  #

    idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  #

    score_abnormal = torch.gather(abnormal_scores, 1, idx_abn_score)

    score_abnormal = torch.mean(score_abnormal, dim=1)

    select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
    select_idx_normal = drop_out(select_idx_normal)

    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal = torch.topk(nfea_magnitudes_drop, k, dim=1)[1]

    idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

    normal_features = normal_features.view(n_size, ncrops, t, f)

    normal_features = normal_features.permute(1, 0, 2, 3)

    total_select_nor_feature = torch.zeros(0)
    for i, nor_fea in enumerate(normal_features):
        feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)

        total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])

    score_normal = torch.gather(normal_scores, 1, idx_normal_score)

    score_normal = torch.mean(score_normal, dim=1)

    abn_feamagnitude = total_select_abn_feature
    nor_feamagnitude = total_select_nor_feature

    return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores


class Backbone(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            heads,
            mgfn_type='gb',
            kernel=5,
            dim_headnumber=64,
            ff_repe=4,
            dropout=0.,
            attention_dropout=0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = FOCUS(dim, heads=heads, dim_head=dim_headnumber, local_aggr_kernel=kernel)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads=heads, dim_head=dim_headnumber, dropout=attention_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding=1),
                attention,
                FeedForward(dim, repe=ff_repe, dropout=dropout),
            ]))

    def forward(self, x):

        for i, (scc, attention, ff) in enumerate(self.layers):
            x_scc = scc(x)

            x = x_scc + x

            x_attn = attention(x)

            x = x_attn + x

            x_ff = ff(x)

            x = x_ff + x

        return x


# main class

class mgfn(nn.Module):
    def __init__(
            self,
            *,
            classes=0,
            dims=(64, 128, 1024),
            depths=(args.depths1, args.depths2, args.depths3),
            mgfn_types=(args.mgfn_type1, args.mgfn_type2, args.mgfn_type3),
            lokernel=5,
            channels=2048,  # default
            # channels = 1024,
            ff_repe=4,
            dim_head=64,
            dropout=0.,
            attention_dropout=0.
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv1d(channels, init_dim, kernel_size=3, stride=1, padding=1)

        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Backbone(
                    dim=stage_dim,
                    depth=depth,
                    heads=heads,
                    mgfn_type=mgfn_types,
                    ff_repe=ff_repe,
                    dropout=dropout,
                    attention_dropout=attention_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride=1),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(last_dim)
        )
        self.batch_size = args.batch_size
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(args.dropout_rate)

        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, video):

        k = 2
        # (100,10,2048)
        if len(video.size()) == 4:
            bs, ncrops, t, c = video.size()

            x = video.view(bs * ncrops, t, c).permute(0, 2, 1)

        elif len(video.size()) == 3:
            bs, _, _ = video.size()
            ncrops = 1
            x = video.permute(0, 2, 1)

        if x.shape[1] == 2049:

            x_f = x[:, :2048, :]

            x_m = x[:, 2048:, :]

        elif x.shape[1] == 1025:

            x_f = x[:, :1024, :]

            x_m = x[:, 1024:, :]

        x_f = self.to_tokens(x_f)

        x_m = self.to_mag(x_m)

        x_f = x_f + args.mag_ratio * x_m

        for i, (backbone, conv) in enumerate(self.stages):

            x_f = backbone(x_f)

            if exists(conv):
                x_f = conv(x_f)

        x_f = x_f.permute(0, 2, 1)

        x = self.to_logits(x_f)

        scores = self.sigmoid(self.fc(x))

        score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = MSNSD(x, scores, bs, self.batch_size,
                                                                                         self.drop_out, ncrops, k)

        return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores
