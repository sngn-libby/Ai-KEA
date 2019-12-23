# -*- coding: utf-8 -*-
"""aidl_cam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e-Akg-I7FBQw7T7uUXEkc2gfdLsyK3QN

# [실습 예제 6 - 2] aidl cam module
"""

# Commented out IPython magic to ensure Python compatibility.

def aidl_cam(img_path):
  import tensorflow as tf

  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg

  import numpy as np

  from tensorflow.keras.applications import vgg16
  from tensorflow.keras.preprocessing import image
  from tensorflow.keras import backend as K
  K.clear_session()
  # vgg16 model import
  model = vgg16.VGG16(weights='imagenet')

  # load the image with the required shape
  img = image.load_img(img_path, target_size=(224, 224))
  # convert the image to an array
  img = image.img_to_array(img)
  # expand dimensions so that it represents a single 'sample'
  img = np.expand_dims(img, axis=0)
  # prepare the image (e.g. scale pixel values for the vgg)
  img = vgg16.preprocess_input(img)
  
  # prediction
  import pandas as pd
  from keras.applications.vgg16 import decode_predictions
  preds = model.predict(img)
  predictions = pd.DataFrame(decode_predictions(preds, top=3)[0],columns=['col1','category','probability']).iloc[:,1:]

  # get prediction and get index of prediction
  argmax = np.argmax(preds[0])
  output = model.output[:, argmax]

  # CAM
  # Model
  # tf.keras.models.Model -> functional model
  grad_model = tf.keras.models.Model(
    [model.inputs],                           # input: vgg16 inputs
    [model.get_layer('block5_conv3').output,  # outputs: vgg16 last layer
     model.output]                            # softmax out
  )

  # --- build CAM
  # GradientTape 설정
  with tf.GradientTape() as tape:
    conv_outputs, predict = grad_model(img)
    loss = predict[:, argmax]     

  output = conv_outputs[0]
  # Get gradient for image
  grads = tape.gradient(loss, conv_outputs)[0]

  # check grads' shape
  grads.shape, output.shape  

  # Average gradients spatially
  weights = tf.reduce_mean(grads, axis=(0, 1))

  # Build a map of filters according to gradients importance
  cam = np.ones(output.shape[0:2], dtype=np.float32)

  for index, w in enumerate(weights):
    cam += w * output[:, :, index]
  # --- 

  # --- blending CAM
  import cv2

  # image read
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2는 기본적으로 image를 GBR로 처리

  # build heat map
  # resize cam
  cam = cv2.resize(cam.numpy(), (img.shape[1], img.shape[0]))
  cam = np.maximum(cam, 0)
  # heatmap : minmax normalized one
  heatmap = (cam - cam.min()) / (cam.max() - cam.min())

  # convert heatmap to rgb
  heatmap = np.uint8(255 * heatmap)                       # RGB range (0~255)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # apply color map
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)      # colormap to RGB

  # blending heatmap: image: 100%, heatmap: 50%
  output_image = cv2.addWeighted(img.astype('uint8'), 1, 
                                heatmap, 0.5, 
                                0)
  # plot blended image
  plt.imshow(output_image)
  plt.axis('off')
  plt.title(predictions.loc[0,'category'])

"""## Sample Image"""

if __name__ == "__main__":
  from google.colab import drive
  drive.mount('/content/drive')

  aidl_cam('/content/drive/Shared drives/scsa_2019_e/z_data/pobee.jpeg')

