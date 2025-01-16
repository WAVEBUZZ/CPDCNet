from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
	work_dir = r'F:\PLT\1hrnet-pytorch-main\VOCdevkit\VOC2007\1JPEGImages'  # 图像所处文件夹

	Out_JPEGImages_path = r'F:\PLT\1hrnet-pytorch-main\VOCdevkit\VOC2007\JPEGImages'
	# os.chmod(Out_JPEGImages_path, stat.S_IRWXU)
	file_names = os.listdir(work_dir)
	for file_name in file_names:
		# print(file_name) # ISIC_0000000_Segmentation.png
		file_path = os.path.join(work_dir, file_name)

		image = Image.open(file_path)
		gray_image = image.convert('L')
		img = np.array(gray_image)
		img[img == 255] = 1

		# 重新保存
		image = Image.fromarray(img, 'L')
		new_name = file_name[:-4]
		# new_name = new_name.strip("_Segmentation")  # 文件名处理成和图像一样的名字
		image.save(os.path.join(Out_JPEGImages_path, new_name + '.png'))
