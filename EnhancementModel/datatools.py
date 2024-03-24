import scipy.io as scio
import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

def Encode(tiff, bottom_cut=False, log_flag=True, inpainting_Ratio=0.5, SelRatio=6.5, ScaRatios=[5.25, 5.75, 6.25]):
	'''
	tiff :　原始的16位tiff切片
	Is ： 经过编码变换后的输入数据
	[tiff_max, tiff_min] : 编码变换过程中，归一操作时输入tiff的原界，用来解码网络输出
	'''

	# 直方图
	hist = cv2.calcHist([np.float32(tiff)], [0], None, [65536], [0, 65536])
	# 平滑直方图，防止误读峰值
	hist = hist.squeeze()
	hist = np.convolve(hist, np.ones((5,))/5, 'same')
	peak = np.argmax(hist)

	# 截顶
	MaxBound = np.floor((peak+1)*(SelRatio+1)) # 上界
	MaxBoundIndex = (tiff > MaxBound).astype(np.float)
	tiff = tiff * (1 - MaxBoundIndex) + MaxBoundIndex * MaxBound

	# 截底
	MinBound = np.floor( (peak+1) ) # 下界
	MinBoundIndex = (tiff < MinBound).astype(np.float)
	if bottom_cut:
		tiff = tiff * (1 - MinBoundIndex) + MinBoundIndex * MinBound

	# inpainting
	inpaintingBound = np.floor( (peak+1) * inpainting_Ratio )
	inpainting_mask = (tiff > inpaintingBound).astype(np.float) # 拒绝低灰度目标参与图像重构和损失计算

	# 界点值
	part_values = []
	for ScaRatio in ScaRatios:
		part_value = np.floor( (peak+1) * ScaRatio )
		part_values.append(part_value)

	# 归一至[0,1]
	tiff_max = np.max(tiff)
	tiff_min = np.min(tiff)
	tiff_norm = (tiff - tiff_min) / (tiff_max - tiff_min)

	if log_flag:
		# 拉伸至[1,e]
		tiff_exp = (np.e - 1) * tiff_norm + 1
		# 同态变换[0,1]
		I = np.log(tiff_exp)
	else:
		I = tiff_norm

	return np.float32(I), np.array([tiff_max, tiff_min, MaxBound, part_values[0], part_values[1], part_values[2]]), np.float32(inpainting_mask)

def Decode(T, tiff_region, log_flag = True, MultiChannel=False, Uint16=False):
	
	if log_flag:
		# 逆同态变换[1,e]
		tiff_exp = np.exp(T)
		# 线性变换[0,1]
		tiff_norm = (tiff_exp - 1) / (np.e -1)
	else:
		tiff_norm = T

	# 重映射至 Uint16
	tiff_recon = tiff_norm * (tiff_region[0] - tiff_region[1]) + tiff_region[1]
	tiff_recon = tiff_recon.astype(np.uint16)

	if Uint16:
		return tiff_recon

	# uint16 to uint8
	MaxBound = tiff_region[2]
	part_values = tiff_region[3:6]
	if MultiChannel:
		pass
	else:
		# return SingleChannel(tiff_recon, MaxBound, part_values[1])
		return np.uint8(255*tiff_norm)

def SingleChannel(tiff, MaxBound, part_value):
	
	# 截顶
	MaxBoundIndex = (tiff > MaxBound).astype(np.float)
	tiff = tiff * (1 - MaxBoundIndex) + MaxBoundIndex * MaxBound

	# 分界
	LowPartIndex = (tiff <= part_value).astype(np.float)
	HighPartIndex = 1 - LowPartIndex
	tiff_remap = (tiff*127.0/part_value)*LowPartIndex + ((tiff-part_value-1)*127.0/(MaxBound-part_value-1)+128.0)*HighPartIndex

	return np.uint8(tiff_remap)

def MultiChannel():
	pass

def pixel_shuffle_down_sampling(x, r=2):

	shuffle_kernel = nn.Parameter(torch.tensor( [[[1,0],[0,0]],[[0,1],[0,0]],[[0,0],[1,0]],[[0,0],[0,1]]] ).type(torch.FloatTensor).unsqueeze(1))
	b,c,w,h = x.shape
	unshuffled = F.conv2d(x, shuffle_kernel, bias=None, stride=int(r), padding=0)

	return unshuffled.view(b,c,r,r,w//r,h//r).permute(0,1,2,4,3,5).reshape(b,c,w,h)

def pixel_shuffle_up_sampling(x, r=2):

	b,c,w,h = x.shape
	before_shuffle = x.view(b,c,r,w//r,r,h//r).permute(0,1,2,4,3,5).reshape(b,c*r*r,w//r,h//r)

	return F.pixel_shuffle(before_shuffle, r)

class BasicDataset(Dataset):
	def __init__(self, datadir, inpainting_Ratio=0.5, log_flag=True):
		super(BasicDataset, self).__init__()

		self.datadir = datadir
		self.datafiles = os.listdir(self.datadir)
		self.length = len(self.datafiles)
		self.log_flag = log_flag
		self.inpainting_Ratio = inpainting_Ratio

	def __len__(self):
		return self.length

	def __getitem__(self, i):
		file_name = self.datafiles[i]
		file_path = os.path.join(self.datadir, file_name)

		tiff = cv2.imread(file_path, -1)
		I, tiff_region, inpainting_mask = Encode(tiff, log_flag=self.log_flag, inpainting_Ratio=self.inpainting_Ratio)

		I = torch.from_numpy(I).type(torch.FloatTensor).unsqueeze(0)
		tiff_region = torch.from_numpy(tiff_region).type(torch.FloatTensor)
		inpainting_mask = torch.from_numpy(inpainting_mask).type(torch.FloatTensor).unsqueeze(0)

		return I, tiff_region, inpainting_mask

if __name__ == '__main__':

	pass

