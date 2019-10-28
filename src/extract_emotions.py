import os
import requests
import json
import numpy as np
from tqdm import tqdm


# Face classifier is running as a separate container
SERVER = 'http://face-classifier:8084'
BATCH_SIZE = 32


def store_emotions(frames_path, emotions_path):

    # Frames from videos
    image_paths = [ os.path.join(frames_path, filename) for filename in os.listdir(frames_path) ]
    
    # loop over the images
    for image_path in tqdm(image_paths):   
        # open image
        image = { 'img': open(image_path, 'rb') }
        # make request
        response = requests.post('{}/classifyImage'.format(SERVER), files=image)
        features = json.loads(response.text)
        # write response: features
        image_name = image_path.split(os.path.sep)[-1]
        filename = os.path.join(emotions_path, '{}.json'.format(image_name))
        with open(filename, 'w') as outfile:
            json.dump(features, outfile)

# DEV
print('[INFO] Extracting emotions from groundtruth frames...')
DEV_FRAMES_DIR = '/datasets/memorability/campus.pub.ro/devset/dev-set/frames'
DEV_EMOTIONS_DIR = '/datasets/emotions/devset'
# store_emotions(config.DEV_FRAMES, config.EMOTIONS_DEV)
store_emotions(DEV_FRAMES_DIR, DEV_EMOTIONS_DIR)

# TEST
print('[INFO] Extracting emotions from test frames...')
TEST_FRAMES_DIR = '/datasets/memorability/campus.pub.ro/testset/me19me-testset/test-set/frames'
TEST_EMOTIONS_DIR = '/datasets/emotions/testset'
# store_emotions(config.TEST_FRAMES, config.EMOTIONS_TEST)
store_emotions(TEST_FRAMES_DIR, TEST_EMOTIONS_DIR)
