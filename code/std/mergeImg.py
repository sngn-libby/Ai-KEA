import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

file_dir = '/content/furniture/'

def imageMerge(file_path_1, file_path_2, debug=True): # format : '/content/furniture/chair'

	num = 0
	cls_1 = file_path_1.split(sep='/')[-1]
	cls_2 = file_path_2.split(sep='/')[-1]
	cls_name = cls_1+'_'+cls_2
	
	save_path = file_dir+cls_name
	os.makedirs(save_path, exist_ok=True)

	files1 = [ item for item in glob.glob(file_path_1 + '/*.jpg')][:4]
	files2 = [ item for item in glob.glob(file_path_2 + '/*.jpg')][:4]
	print(files1)
	for i in files1:
		img1 = image.load_img(i)
		if debug is True : plt.imshow(img1)
		for j in files2:
			img2 = image.load_img(j)
			if debug is True : plt.imshow(img2)
			w = min([img1.size[0], img2.size[0]])
			h = min([img1.size[1], img2.size[1]])

			if debug is True : print(w, h)
			resized_1 = img1.resize((w, h))
			resized_2 = img2.resize((w, h))
   
			if debug is True: 
						plt.imshow(resized_2)
			new_img = Image.new('RGB', (w*2, h), 'black')
			plt.imshow(new_img)
			
			if debug is True : 
						print(new_img.size)
						print(type(new_img), type(img2), type(resized_2))
			new_img.paste(resized_1, (0, 0))
			new_img.paste(resized_2, (w, 0))
			#save_path = 
			new_img.save(save_path+'/'+cls_name+'_'+str(num)+'.jpg')
			num+=1
			if debug is True : plt.imshow(new_img)
