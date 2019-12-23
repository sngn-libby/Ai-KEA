# import modules
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.keras.datasets import mnist

'''
print("Module Loaded.")
print("NumPy Version :{}".format(np.__version__))
#print("TensorFlow Version :{}".format(tf.__version__))
print("Matplotlib Version :{}".format(plt.matplotlib.__version__))
'''

# Accuracy
def Accuracy(y:np.ndarray, t:np.ndarray)->np.float32:
    return np.mean(np.equal(np.argmax(y, axis=1).reshape((-1, 1)),t).astype(np.float32))

'''
    결과 출력 함수
    Make_Result_Plot
    Arguments:
        suptitle:
        data:
        label:
        y_max:
'''
def Make_Result_Plot(suptitle:str, data:np.ndarray, label:np.ndarray, y_max:np.ndarray):
    fig_result, ax_result = plt.subplots(2,5,figsize=(18, 7))
    fig_result.suptitle(suptitle)
    for idx in range(10):
        ax_result[idx//5][idx%5].imshow(data[idx].reshape((28,28)),cmap="binary")
        ax_result[idx//5][idx%5].set_title("test_data[{}] (label : {} / y : {})".format(idx, 
                                                                    label[idx], y_max[idx]))


'''
    show image function
    input: 
        x: np.ndarray image data
        y: display image number
'''
def show_img(x, y=16):
    size_img = 28
    plt.figure(figsize=(8,7))
    num_images = y
    n_samples = x.shape[0]
    x = x.reshape(n_samples, size_img, size_img)
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(x[i], cmap='gray')
    plt.show()

def model_eval(model, n=230):
  #n = 230
  plt.figure(figsize=(10, 2))
  for i in range(5):
    img_idx = n+i
    predict = model.predict(x_test[img_idx:img_idx+1, :])
    img = x_test[img_idx]
    plt.subplot(1,5,i+1)
    plt.imshow(img.reshape((28,28)), cmap='gray')
    plt.title('pre: {}'.format(np.argmax(predict)))

  plt.show()

import IPython.display as display
from PIL import Image

# tf.keras.preprocessing.image.ImageDataGenerator()
## 가 생성한 결과의 일부를 확인
def show_batch(data_gen, class_l):
  # get image and label from data generator
  img_batch, l_batch = next(data_gen) 
  plt.figure(figsize=(10,10))
  for n in range(25):
    ax = plt.subplot(5,5,n+1)
    plt.imshow(img_batch[n])
    plt.title(class_l[l_batch[n]==1][0].title())
    plt.axis('off')
  return img_batch, l_batch

# 지정 폴더 아래에 있는 모든 *.jpg 파일의 수
#  및 폴더명 목록을 리턴  
def check_dir(d_path):
  img_count = len(list(d_path.glob('*/*.jpg')))
  c_name = np.array([item.name for item in d_path.glob('*') if item.name != "LICENSE.txt"])
  return img_count, c_name


# 지정 path 아래에 있는 폴더에서 이미지 두장씩을 확인
def check_image(d_path, class_list):
  for i in range(len(class_list)):
    class_temp = list(d_path.glob(str(class_list[i])+'/*'))
    for image_path in class_temp[:2]:
      display.display(Image.open(str(image_path)))
