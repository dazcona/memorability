import os
import numpy as np
import pandas as pd
from keras.applications import ResNet152, ResNet50, VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
# import h5py
import progressbar
from skimage.io import imread
from skimage.transform import resize
import config
from keras.models import load_model
# import pdb; pdb.set_trace()

## LOGDIR

MAIN_LOG = os.path.join(config.RUN_LOG_DIR, 'mainlog.txt')

# from tensorflow.python import debug as tf_debug
# import keras
# sess = keras.backend.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# keras.backend.set_session(sess)

# frame numbers
FRAME_NUMBERS = [1, 24, 48, 72, 96, 120, 144, 168]
# Image Size to input the model
# IMG_SIZE = (1080, 1920)

IMG_SIZE = (224, 224)

# Hyperparameters
LEARNING_RATE_WARMUP = 1e-2
EPOCHS_WARMUP = 5

# LAYER_TO_TRAIN_FROM = 453
LEARNING_RATE = 1e-2
EPOCHS = 5
# DECAY = LEARNING_RATE / EPOCHS


def get_data(videos, y, frames_path):

    # TEST one frame only
    # frame = 72
    image_paths = [
        os.path.join(frames_path, video.split('.webm')[0] + '-frame-{}.jpg'.format(frame))
        for video in videos
        for frame in FRAME_NUMBERS
    ]
    # y_frames = y 
    y_frames = np.repeat(y, len(FRAME_NUMBERS)) 
    print("[INFO] Number of images: {:,}, scores: {:,}...".format(len(image_paths), len(y_frames)))

    return image_paths, y_frames


from keras.utils import Sequence

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

class My_Generator(Sequence):

    def __init__(self, filenames, labels, batch_size):
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.filenames) / float(self.batch_size)).astype(int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
        # batch_x_images = np.array([ resize(imread(file_name), (224, 224)) for file_name in batch_x])
        batch_x_images = []
        for image_path in batch_x:
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = imagenet_utils.preprocess_input(image)
            # image = ((image / 255.) - means) / stds
            batch_x_images.append(image)
        batch_x_images = np.array(batch_x_images)
        return batch_x_images, batch_y


def train_fine_tuned_cnn(train_videos, val_videos, y_train, y_val, frames_path):

    print("[INFO] [FIRST PART] Fine-tuning CNN...")

    print('[INFO] Creating generators...')
    batch_size = 32

    print('[INFO] Training data...')
    train_images_paths, train_y = get_data(train_videos, y_train, frames_path)
    print('[INFO] Validation data...')
    val_images_paths, val_y = get_data(val_videos, y_val, frames_path)
    print('[INFO] Creating generators data...')
    my_training_batch_generator = My_Generator(train_images_paths, train_y, batch_size)
    my_validation_batch_generator = My_Generator(val_images_paths, val_y, batch_size)

    # NUM_TRAINING_SAMPLES = len(train_images_paths)
    # NUM_VALIDATION_SAMPLES = len(val_images_paths)

    if config.FINE_TUNED_MODEL == '':

        ## TRAINING MODEL

        model = train_base_model(my_training_batch_generator, my_validation_batch_generator)

    else:

        ## LOAD MODEL

        model_name = config.FINE_TUNED_MODEL
        print('[INFO] Loading model "{}"...'.format(model_name))
        model = load_model(model_name)

    if config.FINE_TUNE_EVALUATE_ONLY:

        ## EVALUATE MODEL

        return evaluate_model(model, my_validation_batch_generator)

    else:

        ## TRAIN FROM BASELINE MODEL

        return train_from_base_model(model, my_training_batch_generator, my_validation_batch_generator)


def train_base_model(my_training_batch_generator, my_validation_batch_generator):

    # Base Model
    print("[INFO] Defining base model...")
    base_model = ResNet152(
        weights='imagenet', 
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

    # Head Model
    head_model = base_model.output
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(256, activation="relu")(head_model)
    # head_model = Dropout(0.5)(head_model)
    head_model = Dropout(0.75)(head_model)
    head_model = Dense(1, activation='sigmoid')(head_model)

    # Combine models
    model = Model(inputs=base_model.input, outputs=head_model)

    print("[INFO] Freezing the model's layers...")
    # loop over all layers in the base model and freeze them so they
    # will *not* be updated during the training process
    for layer in base_model.layers:
        layer.trainable = False

    with open(MAIN_LOG, 'a') as f:
        print("Base model's layers are frozen", file=f)

    print("[INFO] Model layers...")
    for (i, layer) in enumerate(model.layers):
        print("[INFO] {}\t{}\t{}".format(i, layer.__class__.__name__, layer.trainable))

    print("[INFO] Model summary...")
    print(model.summary())

    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable
    print("[INFO] Compiling model...")
    # Optimizer
    opt = SGD(lr=LEARNING_RATE_WARMUP)
    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['mse', 'mae', 'mape'],
    )

    with open(MAIN_LOG, 'a') as f:
        print('Optimizer: {}, LR: {}'.format('SDG', LEARNING_RATE_WARMUP), file=f)


    # train the head of the network for a few epochs (all other
    # layers are frozen) -- this will allow the new FC layers to
    # start to become initialized with actual "learned" values
    # versus pure random
    print("[INFO] Training model's head...")

    part = 0

    tensorboard = TensorBoard(log_dir=config.RUN_LOG_FOLD_DIR.format(part))
    
    checkpoints = ModelCheckpoint(
    os.path.join(
        config.RUN_CHECKPOINT_DIR,
        'weights-part_' + str(part) + '-{epoch:02d}-{val_loss:.10f}.hdf5'),
    monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    H = model.fit_generator(
        generator=my_training_batch_generator,
        # steps_per_epoch=NUM_TRAINING_SAMPLES // batch_size,
        epochs=EPOCHS_WARMUP,
        validation_data=my_validation_batch_generator,
        # validation_steps=NUM_VALIDATION_SAMPLES // batch_size,
        use_multiprocessing=True,
        workers=8,
        max_queue_size=32,
        verbose=1,
        callbacks=[
            tensorboard,
            checkpoints,
            ])

    # save model
    model.save(os.path.join(config.RUN_LOG_DIR, 'fine_tuned_model_after_warmup.h5'))

    return model


def evaluate_model(model, my_validation_batch_generator):

    # evaluate the network after initialization
    print("[INFO] Predicting values after head warm up...")
    preds_frames = model.predict_generator(
        my_validation_batch_generator, 
        #steps=NUM_VALIDATION_SAMPLES // batch_size
    )

    # one frame
    # preds_frames = preds_frames.flatten()
    # np.savetxt(os.path.join(config.RUN_LOG_DIR, 'preds_frames.out'),
    #     preds_frames, delimiter=',')
    # print(preds_frames)
    ### #### #### #### #### #### #### #### #### #### #### 
    # return preds_frames
    ### #### #### #### #### #### #### #### #### #### #### #### #### 

    # Predictions for each video: average of the frames' predictions for each video
    preds_frames = preds_frames.flatten()
    # average for each video
    preds_videos = np.mean(preds_frames.reshape(-1, len(FRAME_NUMBERS)), axis=1)
    # save array as text
    np.savetxt(os.path.join(config.RUN_LOG_DIR, 'preds_videos.out'),
        preds_videos, delimiter=',')

    return preds_videos


def train_from_base_model(model, my_training_batch_generator, my_validation_batch_generator):


    print("[INFO] [SECOND PART] Fine-tuning CNN...")

    # now that the head FC layers have been trained/initialized, lets
    # unfreeze the final set of CONV layers and make them trainable
    
    ############################################################################
    #### TEST
    # for layer in model.layers[LAYER_TO_TRAIN_FROM:]:
    #     layer.trainable = True

    # with open(MAIN_LOG, 'a') as f:
    #     print('Layer to train it from: {}'.format(LAYER_TO_TRAIN_FROM), file=f)
    ############################################################################


    # loop over the layers in the network and display them to the console
    print("[INFO] Model layers...")
    for (i, layer) in enumerate(model.layers):
        print("[INFO] {}\t{}\t{}".format(i, layer.__class__.__name__, layer.trainable))

    # Re-compile model
    print("[INFO] Re-compiling model...")
    opt = SGD(lr=LEARNING_RATE)
    # opt = Adam(lr=LEARNING_RATE, decay=DECAY)
    # opt = RMSprop(lr=LEARNING_RATE)
    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['mse', 'mae', 'mape'],
    )

    with open(MAIN_LOG, 'a') as f:
        print('Optimizer: {}, LR: {:.2f}'.format('SGD', LEARNING_RATE), file=f)

    # ...
    print("[INFO] Fine-tuning model...")

    part = 1

    tensorboard = TensorBoard(log_dir=config.RUN_LOG_FOLD_DIR.format(part))

    checkpoints = ModelCheckpoint(
    os.path.join(
        config.RUN_CHECKPOINT_DIR,
        'weights-part_' + str(part) + '-{epoch:02d}-{val_loss:.10f}.hdf5'),
    monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    H = model.fit_generator(
        generator=my_training_batch_generator,
        # steps_per_epoch=NUM_TRAINING_SAMPLES // batch_size,
        epochs=EPOCHS,
        validation_data=my_validation_batch_generator,
        # validation_steps=NUM_VALIDATION_SAMPLES // batch_size,
        use_multiprocessing=True,
        workers=8,
        max_queue_size=32,
        verbose=1,
        callbacks=[
            tensorboard,
            checkpoints,
            ])

    # save model
    model.save(os.path.join(config.RUN_LOG_DIR, 'fine_tuned_model.h5'))

    return evaluate_model(model, my_validation_batch_generator)
