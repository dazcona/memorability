# import libraries
from timeit import default_timer
import os
import pandas as pd
import numpy as np
import sys
import requests
import json
import config
from loader import data_load
from embeddings import fit_predict_with_embeddings
from pretrained_cnn import get_features_from_last_layer_pretrained_nn
from modelling import fit_predict
from aesthetics import videos_to_our_aesthetics
from emotions import get_emotions_test_data
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import ResNet152
import csv

## START

start = default_timer()

## SEED

np.random.seed(42)

## LOAD DATA

FEATURES_WEIGHTS = { 
    'C3D': 0,
    'AESTHETICS': 0,
    'HMP': 0,
    'ColorHistogram': 0,
    'LBP': 0,
    'InceptionV3': 0,
    'CAPTIONS': 1,
    'PRE-TRAINED NN': 1,
    'FINE-TUNED NN': 0,
    'Emotions': 1,
    'Our_Aesthetics': 0,
}
print('[INFO] Predicting {}-term memorability'.format(config.TARGET_SHORT_NAME))
print('[INFO] Loading training and validation data...')
dataframe = data_load(
    FEATURES_WEIGHTS, 
    config.DEV_GROUNDTRUTH, 
    config.DEV_CAPTIONS, 
    config.DEV_FEATURES_PATH,
    'train_val',
)
X = dataframe.drop(columns=config.TARGET_COLS)
Y = dataframe[config.TARGET]

# Special test data only consisting of images

images = []
directory = 'images/'
for _dir in os.listdir(directory):
    path = os.path.join(directory, _dir)
    if os.path.isdir(path):
        image_path = os.path.join(path, 'image.png')
        with open(os.path.join(path, 'caption.txt')) as f: 
            caption = f.read()
        images.append(
            {
                'video': _dir,
                'image_path': image_path,
                'caption': caption,
            }
        )

X_test = pd.DataFrame(images)
print(X_test.head()) 

print('[INFO] Number of training and validation samples: {:,}'.format(len(X)))
print('[INFO] Number of test samples: {:,}'.format(len(X_test)))

## FEATURES

## CAPTIONS, EMBEDDINGS, DEEP LEARNING

# Fit and predict
print('[INFO] Processing CAPTIONS with EMBEDDINGS...')
predictions_features = fit_predict_with_embeddings(X_test['caption'])
# Save predictions
preds_filename = 'images/{}_captions_embeddings'.format(config.TARGET)
np.save(preds_filename, predictions_features)

# PRE-TRAINED CNN AS FEATURE EXTRACTOR

pretrained_nn = config.PRE_TRAINED_NN
print('[INFO] {} as feature extractor...'.format(pretrained_nn))
# # Get features
X_features = get_features_from_last_layer_pretrained_nn(X['video'], config.DEV_FRAMES, 
                pretrained_nn, 'train_val')
# Get test features
IMG_SIZE=(224, 224)
batch_images = []
for image_path in X_test['image_path']:
    print('Loading image: {}'.format(image_path))
    image = load_img(image_path, target_size=IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    batch_images.append(image)
batch_images = np.vstack(batch_images)
model = ResNet152(weights="imagenet", include_top=False, pooling='avg')
size = 2048
features = model.predict(batch_images)
features = features.reshape((features.shape[0], size))
X_test_features = []
for feature in features:
    feature = np.stack((feature,) * len(config.FRAME_NUMBERS), axis=1).flatten()
    X_test_features.append(feature)
# Fit and predict
predictions_features = fit_predict(X_features, Y, X_test_features, pretrained_nn)
# Save predictions
preds_filename = 'images/{}_pretrained'.format(config.TARGET)
np.save(preds_filename, predictions_features)

# OUR PRE-COMPUTED EMOTION FEATURES

print('[INFO] Processing EMOTION test features and applying a traditional ML approach...')
# Get features
X_features = get_emotions_test_data(X['video'], config.EMOTIONS_DEV, 'train_val')
# Get image features
# Face classifier is running as a separate container
EMOTIONS_SERVER = 'http://face-classifier:8084'
filenames = []
for item in images:
    image_path = item['image_path'] 
    image = { 'img': open(image_path, 'rb') }
    response = requests.post('{}/classifyImage'.format(EMOTIONS_SERVER), files=image)
    features = json.loads(response.text)
    # write response: features
    image_name = item['video']
    filename = os.path.join('images', image_name, 'emotions.json')
    with open(filename, 'w') as outfile:
        json.dump(features, outfile)
    filenames.append(filename)
MAX_FACES = 5
NUM_FEATURES_FACE = 12
NUM_FEATURES_FRAME = MAX_FACES * NUM_FEATURES_FACE
NUM_FEATURES_VIDEO = NUM_FEATURES_FRAME * len(config.FRAME_NUMBERS)
video_features = []
for item in images:
    image_name = item['video']
    filename = os.path.join('images', image_name, 'emotions.json')
    # one image only
    with open(filename) as f:
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
        ])
    # features per frame: pad until NUM_FEATURES_FRAME
    frame_features = np.array(frame_features).flatten()
    frame_features_padded = np.zeros(NUM_FEATURES_FRAME)
    frame_features_padded[:len(frame_features)] = frame_features
    image_features = np.stack((frame_features_padded,) * len(config.FRAME_NUMBERS), axis=1).flatten()
    # repeat array
    video_features.append(image_features)
# features per video
X_test_features = np.array(video_features)
# Fit and predict
predictions_features = fit_predict(X_features, Y, X_test_features, 'Emotions')
# Save predictions
preds_filename = 'images/{}_emotions'.format(config.TARGET)
np.save(preds_filename, predictions_features)

## FINAL PREDICTIONS

SUBMISSIONS = {
    'short': { 
        'Captions': 0.40, 
        'Resnet152': 0.55, 
        'Emotions': 0.05, 
        'C3D': 0.00, 
        'LBP': 0.00, 
        'Aesthetics': 0.00 
    },
}

PREDICTIONS = {
    'short': {
        'Captions': 'images/short-term_memorability_captions_embeddings.npy', 
        'Resnet152': 'images/short-term_memorability_pretrained.npy', 
        'Emotions': 'images/short-term_memorability_emotions.npy', 
    }
}

for target in SUBMISSIONS.keys():

    weights = SUBMISSIONS[target]

    print('Sum of weights: {}'.format(sum(weights.values())))

    predictions = []

    for name, weight in weights.items():
        
        if weight > 0:
            print("[INFO] Reading: {} with weight: {}".format(name, weight))
            filename = PREDICTIONS[target][name]
            predictions.append(
                np.load(filename) * weight
            )

    predictions = np.sum(predictions, axis=0)

    # write them!
    results = 'images/predictions.csv'
    with open(results, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['video name', 'memorability score'])
        for video, prediction in zip(X_test['video'], predictions):
            filewriter.writerow( [video, prediction] )

## END

end = default_timer() - start
minutes, seconds = divmod(end, 60)
print('[INFO] Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds))