import os
import glob
import copy
import cv2
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler

from swin_transformer import *
from datatools import *

if __name__ == "__main__":
	
	# device 2 5 6
	os.environ['CUDA_VISIBLE_DEVICES'] = '3'

	# uint16 tiff dir
	tiff_dir = '/workspace/SAR_Aircraft/TiffSlice/20221012/JPEGImages/'

	# demo path
	demo_dir = '/workspace/SAR_Aircraft/EnhancementModel/demo/'
	demo_path = '/workspace/SAR_Aircraft/EnhancementModel/demo/GF3_MDJ_SL_008316_E121.3_N31.2_20180309_L2_VV_L20003093820_20480_13056.tiff'
	save_demo_dir = '/workspace/SAR_Aircraft/EnhancementModel/demo/20230216/shuffle-false_patchsize-4_p-15b16_inpainting-1b2/'
	os.makedirs(save_demo_dir, exist_ok=True)

	# model dir
	model_dir = '/workspace/SAR_Aircraft/EnhancementModel/model/20230216/shuffle-false_patchsize-4_p-15b16_inpainting-1b2/'
	os.makedirs(model_dir, exist_ok=True)



	# network and optimizer
	patch_size = 4
	log_flag = True
	pixel_shuffle_flag = False
	p = 15/16
	inpainting_Ratio = 1/2

	model = SwinTransformer(patch_size=patch_size)
	model = model.cuda()

	# train set
	TrainingDataset = BasicDataset(tiff_dir, inpainting_Ratio=inpainting_Ratio, log_flag=log_flag)
	TrainingLoader = DataLoader(dataset=TrainingDataset,
								num_workers=8,
								batch_size=2,
								shuffle=True,
								pin_memory=False,
								drop_last=True)

	learning_rate = 1e-4
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

	n_epoch = 500
	n_eval = 10
	n_inference = 100
	for epoch in range(1, n_epoch+1):

		for param_group in optimizer.param_groups:
			current_lr = param_group['lr']
		print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

		for iteration, (I, tiff_region, inpainting_mask) in enumerate(TrainingLoader):

			if pixel_shuffle_flag:
				I = pixel_shuffle_down_sampling(I)
				inpainting_mask = pixel_shuffle_down_sampling(inpainting_mask)
				I = I.detach()
				inpainting_mask = inpainting_mask.detach()
		
			# 伯努利采样矩阵
			p_mtx = torch.rand(I.size()).float()
			p_mtx = torch.FloatTensor(p_mtx)
			mask = p_mtx>p
			mask = torch.FloatTensor(mask.float())
			mask = mask * inpainting_mask # 拒绝低灰度像素参与图像重构和损失计算
	
			# input data
			data_in = copy.deepcopy(I)
			data_in = data_in.cuda()
			mask = mask.cuda()

			# Bernoulli probability
			data_in = data_in * mask
	
			# target
			target = copy.deepcopy(I)
			target = target.cuda()
	
			model.train()
			data_out = model(data_in) # 1 * 1 * 1024 * 1024
			
			# loss
			inpainting_mask = inpainting_mask.cuda()
			loss = torch.sum( (data_out-target)*(data_out-target)*(1-mask)*inpainting_mask ) / torch.sum( (1-mask)*inpainting_mask )
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			print('max value of output : ', torch.max(data_out).item(), 'max value of ground truth : ', torch.max(target).item())
			print("iteration %d, loss = %.4f" % (iteration+1, loss.item()*100))
		
		scheduler.step()

		if epoch%n_eval==0 or epoch==1:
			model.eval()

			# demo and input data
			demo = cv2.imread(demo_path, -1)
			I, tiff_region, inpainting_mask = Encode(demo, bottom_cut=False, log_flag=log_flag, inpainting_Ratio=inpainting_Ratio)
			inpainting_mask_3d = np.expand_dims(inpainting_mask, axis=0)
			inpainting_mask_4d = np.expand_dims(inpainting_mask_3d, axis=0)
			sum_preds = np.zeros((I.shape[0],I.shape[1])) #　1024 * 1024
			for i_inference in range(n_inference):
				# 伯努利采样
				p_mtx = np.random.uniform(size=[I.shape[0],I.shape[1]]) #　1024 * 1024
				mask = (p_mtx>p).astype(np.float32)
				mask = mask * inpainting_mask

				I_input = I * mask 

				# input tensor
				I_tensor = torch.from_numpy(I_input)
				I_tensor = I_tensor.unsqueeze(0).unsqueeze(0) # 1 * 1 * 1024 * 1024
				if pixel_shuffle_flag:
					I_tensor = pixel_shuffle_down_sampling(I_tensor)
				I_tensor = I_tensor.cuda() 
				
				T = model(I_tensor) # 1 * r**2 * 512//r * 512//r
				if pixel_shuffle_flag:
					T = pixel_shuffle_up_sampling(T)

				sum_preds += np.squeeze(T.detach().cpu().numpy()) #　1024 * 1024
			# 求平均且归一到 [0, 1]
			T_avg = np.float32(  np.clip(   (sum_preds-np.min(sum_preds)) / (np.max(sum_preds)-np.min(sum_preds)), 0, 1   )  ) 
			
			# Decode to uint16 tiff
			output_name = 'epoch_{:06d}.tiff'.format(epoch)
			save_output_path = os.path.join(save_demo_dir, output_name)

			T_avg = Decode(T_avg, tiff_region, log_flag=log_flag, MultiChannel=False, Uint16=True)
			
			tiff_result = Image.fromarray( T_avg )
			tiff_result.save(save_output_path)

			# save model
			model_name = 'epoch_{:06d}.pth'.format(epoch)
			save_model_path = os.path.join(model_dir, model_name)
			torch.save(model.state_dict(), save_model_path)