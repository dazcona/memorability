# import the necessary packages
############ , , , 
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# from imutils import paths
import numpy as np
import progressbar
import os
import config
from loader import data_load
import h5py


# Image Size to input the model
IMG_SIZE=(224, 224)
# frame numbers
FRAME_NUMBERS = [1, 24, 48, 72, 96, 120, 144, 168]


# Pre-trained Network model
def get_model(model_name):

    if model_name == 'VGG16':

        from keras.applications import VGG16
        model = VGG16(weights="imagenet", include_top=False, pooling='avg')
        size = 512 # if pooling is 'avg', else  512 * 7 * 7 if pooling is 'None'

    elif model_name == 'ResNet50':

        from keras.applications import ResNet50
        model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
        size = 2048, # if pooling is 'avg',  2048 * 7 * 7 if pooling is 'None'

    elif model_name == 'ResNet152':

        from keras.applications import ResNet152
        model = ResNet152(weights="imagenet", include_top=False, pooling='avg')
        size = 2048 # if pooling is 'avg', 2048 * 7 * 7 # if pooling is 'None'

    elif model_name == 'DenseNet121':

        from keras.applications import DenseNet121
        model = DenseNet121(weights="imagenet", include_top=False, pooling='avg'),
        size = 1024 # if pooling is 'avg'

    elif model_name == 'Custom':

        ## CUSTOM MODEL
        
        from keras.models import load_model
        model = load_model(config.FINE_TUNED_MODEL)
        size = 2048 # our trained models are based on ResNet152

    else:

        raise ValueError("Model needs to be defined. Examples: VGG16 or ResNet50.")

    return model, size


def get_features_from_last_layer_pretrained_nn(videos, frames_path, NN_name, train_val_or_test):

    # model and size
    model, size = get_model(NN_name)

    # Output filename
    dev_or_test = frames_path.split(os.path.sep)[2] # devset or testset
    h5_filename = '{}/{}_{}_{}.h5'.format(config.MY_FEATURES_DIR, NN_name, dev_or_test, train_val_or_test)
    
    # videos
    videos = list(videos)

    if not os.path.isfile(h5_filename):

        print('[INFO] Creating features for pre-trained model...')
        
        # images
        image_paths = [ 
            os.path.join(frames_path, video.split('.webm')[0] + '-frame-{}.jpg'.format(frame))
            for video in videos
            for frame in FRAME_NUMBERS
        ]

        # initialize the progress bar
        widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

        batch_size = 32

        with h5py.File(h5_filename, 'w') as h5f: 

            # loop over the images in batches
            for i in np.arange(0, len(image_paths), batch_size):
                
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []

                # loop over the images in the current batch
                for image_path in batch_paths:
                    
                    # load the input image using the Keras helper utility
                    # while ensuring the image is resized to 224x224 pixels
                    image = load_img(image_path, target_size=IMG_SIZE)
                    image = img_to_array(image)
                    # preprocess the image by (1) expanding the dimensions and
                    # (2) subtracting the mean RGB pixel intensity from the
                    # ImageNet dataset
                    image = np.expand_dims(image, axis=0)
                    image = imagenet_utils.preprocess_input(image)
                    batch_images.append(image)

                # pass the images through the network and use the outputs as
                # our actual features
                batch_images = np.vstack(batch_images)
                features = model.predict(batch_images, batch_size=batch_size)
                # reshape the features so that each image is represented by
                # a flattened feature vector of the ‘MaxPooling2D‘ outputs
                features = features.reshape((features.shape[0], size))

                # add features per image
                for feature_index, batch_index in zip(np.arange(0, features.shape[0], len(FRAME_NUMBERS)), 
                                                    np.arange(i, i + len(FRAME_NUMBERS))):
                    # Concatenate all features per frame to one
                    image_features = np.concatenate( features[feature_index:feature_index + len(FRAME_NUMBERS)] )
                    # grab the video corresponding to these frames
                    image_index = batch_index // len(FRAME_NUMBERS) + batch_index % len(FRAME_NUMBERS)
                    video_name = videos[image_index]
                    # save it!
                    h5f.create_dataset(video_name, data=image_features)

                # update the progress bar
                pbar.update(i)

        # progress bar
        pbar.finish()

    # Reading file

    print('[INFO] Reading features from {}...'.format(h5_filename))

    with h5py.File(h5_filename) as h5f:

        features = np.array([ h5f[video_name][:] for video_name in videos ])

    return features


