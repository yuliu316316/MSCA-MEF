# -*- coding:utf-8 -*-

import os
import torch
from torch.autograd import Variable
from net import MSCA_autoencoder
import utils
from args_fusion import args
import numpy as np
import time


def load_model(path, deepsupervision):
	input_nc = 1
	output_nc = 1
	# nb_filter = [64, 112, 160, 208, 256]
	nb_filter = [32, 64, 128, 256, 208]
	# nb_filter = [32, 64, 128, 208, 160, 192, 256, 320]

	mef_model = MSCA_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
	mef_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in mef_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(mef_model._get_name(), para * type_size / 1000 / 1000))

	mef_model.eval()
	mef_model.cuda()

	return mef_model


def run_demo(mef_model, infrared_path, visible_path, output_path_root, index, f_type):
	img_ir, h, w, c = utils.get_test_image(infrared_path)
	img_vi, h, w, c = utils.get_test_image(visible_path)

	# dim = img_ir.shape
	if c is 1:
		if args.cuda:
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)
		# encoder
		en_r = mef_model.encoder(img_ir)
		en_v = mef_model.encoder(img_vi)
		# fusion
		f = mef_model.fusion(en_r, en_v, f_type)
		# decoder
		img_fusion_list = mef_model.decoder_eval(f)
	else:
		# fusion each block
		img_fusion_blocks = []
		for i in range(c):
			# encoder
			img_vi_temp = img_vi[i]
			img_ir_temp = img_ir[i]
			if args.cuda:
				img_vi_temp = img_vi_temp.cuda()
				img_ir_temp = img_ir_temp.cuda()
			img_vi_temp = Variable(img_vi_temp, requires_grad=False)
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)

			en_r = mef_model.encoder(img_ir_temp)
			en_v = mef_model.encoder(img_vi_temp)
			# fusion
			f = mef_model.fusion(en_r, en_v, f_type)
			# decoder
			img_fusion_temp = mef_model.decoder_eval(f)
			img_fusion_blocks.append(img_fusion_temp)
		img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

	############################ multi outputs ##############################################
	output_count = 0
	for img_fusion in img_fusion_list:
		file_name = str(index) + '.png'
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)
		print(output_path)


def main():
	# run demo
	#test_path = "images/IV_images/"
	path1 = "oe_ue/benchmark/o_Y/"
	path2 = "oe_ue/benchmark/un_Y/"
	deepsupervision = False  # true for deeply supervision
	fusion_type = ['mean']
	#strategy_type_list = ['addition']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

	with torch.no_grad():
		if deepsupervision:
			model_path = args.model_deepsuper
		else:
			model_path = args.model_default
		model = load_model(model_path, deepsupervision)
		for j in range(1):
			output_path = './outputs/'

			if os.path.exists(output_path) is False:
				os.mkdir(output_path)
			output_path = output_path
			f_type = fusion_type[j]
			print('Processing......  ' + f_type)
			start = time.time()

			for i in range(5):
				index = i + 1
				infrared_path = path1 + str(index) + '.png'
				visible_path = path2 + str(index) + '.png'
				run_demo(model, infrared_path, visible_path, output_path, index, f_type)

			end = time.time()
			time_cost = end - start
			print(time_cost)
	print('Done......')


if __name__ == '__main__':
	main()


