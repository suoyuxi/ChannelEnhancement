import os
import copy
from tqdm import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
import numpy as np

from swin_transformer import SwinTransformer
from datatools import *

if __name__ == "__main__":
	
	# device
	os.environ['CUDA_VISIBLE_DEVICES'] = '6'

	# uint16 tiff dir
	tiff_dir = '/workspace/SAR_Aircraft/TiffSlice/20230308/JPEGImages/'
	tiff_names = os.listdir(tiff_dir)

	# output path
	output_tiff_dir = '/workspace/SAR_Aircraft/TiffSlice/20230308/Enhanced_tiff_epoch500/'
	os.makedirs(output_tiff_dir, exist_ok=True)

	output_jpg_dir = '/workspace/SAR_Aircraft/TiffSlice/20230308/Enhanced_jpg_epoch500/'
	os.makedirs(output_jpg_dir, exist_ok=True)

	# load model
	model_path = '/workspace/SAR_Aircraft/EnhancementModel/model/20230211/shuffle-false_patchsize-2_p-15b16_inpainting-1b2/epoch_000500.pth'

	p = 15/16
	inpainting_Ratio = 1/2
	patch_size = 2
	model = SwinTransformer(patch_size=patch_size)
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model = model.cuda()
	model.eval()

	# inference iterations
	n_inference = 10

	for tiff_name in tqdm(tiff_names):

		torch.cuda.empty_cache()
		
		tiff_path = os.path.join(tiff_dir, tiff_name)
		tiff = cv2.imread(tiff_path, -1)
		
		I, tiff_region, inpainting_mask = Encode(tiff, inpainting_Ratio=inpainting_Ratio)

		sum_preds = np.zeros((I.shape[0],I.shape[1]))
		for i_inference in range(n_inference):
			
			p_mtx = np.random.uniform(size=[I.shape[0],I.shape[1]])
			mask = (p_mtx>p).astype(np.float32)
			mask = mask * inpainting_mask

			I_input = I * mask 

			I_tensor = torch.from_numpy(I_input)
			I_tensor = I_tensor.unsqueeze(0).unsqueeze(0).cuda()

			mask = torch.tensor(mask)
			mask = mask.unsqueeze(0).unsqueeze(0).cuda()
			
			T = model(I_tensor)
			sum_preds += np.squeeze(T.detach().cpu().numpy())

		# average and norm to [0, 1]
		T_avg = np.float32(  np.clip(   (sum_preds-np.min(sum_preds)) / (np.max(sum_preds)-np.min(sum_preds)), 0, 1   )  ) 
		# Decode to uint16 tiff
		T_avg = Decode(T_avg, tiff_region, MultiChannel=False, Uint16=True)

		# save tiff
		output_tiff_path = os.path.join(output_tiff_dir, tiff_name)
		result = Image.fromarray(T_avg)
		result.save(output_tiff_path)

		# save jpg
		jpg_name = tiff_name.replace('.tiff','.jpg')
		output_tiff_path = os.path.join(output_jpg_dir, jpg_name)
		result = Image.fromarray(np.uint8(255*T_avg/np.max(T_avg)))
		result.save(output_tiff_path)