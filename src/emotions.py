import os
import config
import json
import numpy as np
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint


MAX_FACES = 5
NUM_FEATURES_FACE = 12
NUM_FEATURES_FRAME = MAX_FACES * NUM_FEATURES_FACE
NUM_FEATURES_VIDEO = NUM_FEATURES_FRAME * len(config.FRAME_NUMBERS)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
DECAY = 1e-4 / NUM_EPOCHS


def get_video_emotions(videos, emotions_path):

    print("[INFO] Number of videos: {:,}...".format(len(videos)))

    features = []

    for video in tqdm(videos):

        image_paths = [
            os.path.join(emotions_path, video.split('.webm')[0] + '-frame-{}.jpg.json'.format(frame))
            for frame in config.FRAME_NUMBERS
        ]

        video_features = []

        for image_path in image_paths:

            with open(image_path) as f:
                data = json.load(f)

            results = data['results']
            frame_features = []
            
            for face in results[:MAX_FACES]:

                frame_features.append([ 
                    float(face['emotion_prediction0']),
                    float(face['emotion_prediction1']),
                    float(face['emotion_prediction2']),
                    float(face['emotion_prediction3']),
                    float(face['emotion_prediction4']),
                    float(face['emotion_prediction5']),
                    float(face['emotion_prediction6']),
                    float(face['gender_prediction_male']),
                    float(face['x1']),
                    float(face['x2']),
                    float(face['y1']),
                    float(face['y2']),
                    # (float(face['x2']) - float(face['x1'])) * (float(face['y2']) - float(face['y1'])), # area
                ])

            # features per frame: pad until NUM_FEATURES_FRAME
            frame_features = np.array(frame_features).flatten()
            frame_features_padded = np.zeros(NUM_FEATURES_FRAME)
            frame_features_padded[:len(frame_features)] = frame_features

            video_features.append(frame_features_padded)

        # features per video
        video_features = np.array(video_features).flatten()
        features.append(video_features)
    
    features = np.array(features)

    return features


def get_emotions_data(train_videos, val_videos, emotions_path):

    print('[INFO] Training data...')
    X_emotions_train_filename = '{}/emotions_train.npy'.format(config.MY_FEATURES_DIR)

    if not os.path.isfile(X_emotions_train_filename):
        X_train = get_video_emotions(train_videos, emotions_path)
        np.save(X_emotions_train_filename, X_train)
    else:
        X_train = np.load(X_emotions_train_filename)

    print('[INFO] Validation data...')
    X_emotions_val_filename = '{}/emotions_val.npy'.format(config.MY_FEATURES_DIR)

    if not os.path.isfile(X_emotions_val_filename):
        X_val = get_video_emotions(val_videos, emotions_path)
        np.save(X_emotions_val_filename, X_val)
    else:
        X_val = np.load(X_emotions_val_filename)

    return X_train, X_val


def get_emotions_test_data(videos, emotions_path, dev_or_test):
    """ Get emotion features """

    print('[INFO] Testing data...')
    X_emotions_filename = '{}/emotions_{}.npy'.format(config.MY_FEATURES_DIR, dev_or_test)

    if not os.path.isfile(X_emotions_filename):
        features = get_video_emotions(videos, emotions_path)
        np.save(X_emotions_filename, features)
    else:
        features = np.load(X_emotions_filename)

    return features


def train_emotions(train_videos, val_videos, y_train, y_val, emotions_path):

    print("[INFO] Getting emotions...")
    print('Number of features per video: {}'.format(NUM_FEATURES_VIDEO))

    X_train, X_val = get_emotions_data(train_videos, val_videos, emotions_path)


    if config.EMOTIONS_MODEL == '':

        ## TRAINING MODEL

        print('[INFO] Training model...')

        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=(NUM_FEATURES_VIDEO,)))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))

        print('[INFO] Model\'s Summary')
        print(model.summary())

        # COMPILE

        print('[INFO] Compiling model...')

        # Optimizer
        opt = Adam(lr=LEARNING_RATE, decay=DECAY)

        model.compile(
            loss='mean_squared_error',
            optimizer=opt,
            metrics=['mse', 'mae', 'mape'],
        )

        # FIT

        print('[INFO] Fitting model...')

        fold = 0

        tensorboard = TensorBoard(log_dir=config.RUN_LOG_FOLD_DIR.format(fold))
        
        checkpoints = ModelCheckpoint(
        os.path.join(
            config.RUN_CHECKPOINT_DIR,
            'emotions-weights-fold_' + str(fold) + '-{epoch:02d}-{val_loss:.10f}.hdf5'),
        monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        
        H = model.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=NUM_EPOCHS,
            shuffle=False,
            batch_size=32,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                tensorboard,
                checkpoints,
                ]
        )

    else:

        ## LOAD MODEL

        model_name = config.EMOTIONS_MODEL
        print('[INFO] Loading model "{}"...'.format(model_name))
        model = load_model(model_name)

    ## EVALUATE

    print('[INFO] Predicting values...')
    predicted = model.predict(X_val).flatten()

    return predicted


if __name__ == "__main__":

    videos = [ 
        'video72', 
        'video3', 
        'video56',
    ]
    emotions_path = '/datasets/emotions/devset'
    features = get_video_emotions(videos, emotions_path)
    print(features.shape)
