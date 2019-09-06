import os
from keras.applications import VGG16, ResNet50, ResNet152, DenseNet121
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Pre-trained Network model
NN_DICT = {
    'VGG16': {
        'model': VGG16(weights="imagenet"),
        'layer_name': 'block5_conv3',
    },
    'ResNet50': {
        'model': ResNet50(weights="imagenet"),
        'layer_name': 'activation_49',
    },
    'ResNet152': {
        'model': ResNet152(weights="imagenet"),
        'layer_name': 'conv5_block3_out',
    },
    'DenseNet121': {
        'model': DenseNet121(weights="imagenet"),
        'layer_name': 'relu',
    },
}

# Image Size to input the model
IMG_SIZE=(224, 224)
# frame numbers
FRAME_NUMBERS = [1, 24, 48, 72, 96, 120, 144, 168]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="path to the input image")
ap.add_argument("-m", "--model_name", required=True, help="model to be used")
args = vars(ap.parse_args())

image_path = args["image_path"]
print('[INFO] Loading {}...'.format(image_path))
image_name = image_path.split(os.path.sep)[-1].split('.')[0]
print('[INFO] Image Name: {}'.format(image_name))

model_name = args["model_name"]
print('[INFO] Model name: {}...'.format(model_name))

# Load model and include the densely connected classifer on top
model = NN_DICT[model_name]['model']
# model = VGG16(weights='imagenet')
print(model.summary())

# load image
image = load_img(image_path, target_size=IMG_SIZE)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

# predict
predictions = model.predict(image)
print('Predicted:', imagenet_utils.decode_predictions(predictions, top=3)[0])
index = np.argmax(predictions[0])
print('Index:', index)

# output of the model
image_output = model.output[:, index]
# output feature map of the block5_conv3 layer, the last convolutional layer in VGG16
layer_name = NN_DICT[model_name]['layer_name']
last_conv_layer = model.get_layer(layer_name)

# Gradient of the class with regard to the output feature map of block5_conv3
grads = K.gradients(image_output, last_conv_layer.output)[0]
# Vector of shape (512,), where each entry is the mean intensity of the gradient over a specific feature-map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))
# Lets you access the values of the quantities you just defined: pooled_grads and the output feature map of block5_conv3, given a sample image
iterate = K.function(
    [ model.input ],
    [ pooled_grads, last_conv_layer.output[0] ])
# Values of these two quantities, as Numpy arrays, given the sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([image])
# Multiplies each channel in the feature-map array by 'how important this channel is' with regard to the 'elephant' class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
# The channel-wise mean of the resulting feature map is the heatmap of the class activation.
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
plt.imsave('output/{}_heatmap.png'.format(image_name), heatmap)

# Uses cv2 to load the original image
img = cv2.imread(image_path)
# Resizes the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# Converts the heatmap to RGB
heatmap = np.uint8(255 * heatmap)
# Applies the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img
# Saves the image
cv2.imwrite('output/{}_CAM.jpg'.format(image_name), superimposed_img)

