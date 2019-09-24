import os
import datetime

# Column to predict
TARGET_SHORT_NAME = 'short' # 'short' or 'long'
TARGET = '{}-term_memorability'.format(TARGET_SHORT_NAME)
# Target columns
TARGET_COLS = [ 'short-term_memorability', 'nb_short-term_annotations', 'long-term_memorability', 'nb_long-term_annotations' ]
# Dictionary that indicates which data sources to use in the model and its weight on the overall ensemble model
FEATURES_WEIGHTS = {
    'C3D': 0,
    'AESTHETICS': 0,
    'HMP': 0,
    'ColorHistogram': 0,
    'LBP': 0,
    'InceptionV3': 0,
    'CAPTIONS': 1,
    'PRE-TRAINED NN': 0,
    'FINE-TUNED NN': 0,
    'Emotions': 0,
    'Our_Aesthetics': 0,
}

# GRID SEARCH
GRID_SEARCH = False
# Dictionary that indicates which model to apply Grid Search on
FEATURES_ALGORITHM = {
    'C3D': 'Bayesian Ridge',
    'AESTHETICS': 'Bayesian Ridge',
    'HMP': 'SVM',
    'ColorHistogram': 'Bayesian Ridge',
    'LBP': 'Bayesian Ridge',
    'InceptionV3': 'Bayesian Ridge',
    'CAPTIONS': 'SVM',
    'PRE-TRAINED NN': 'SVM',
    'FINE-TUNED NN': 'Custom',
    'Emotions': 'Bayesian Ridge',
    'Our_Aesthetics': 'SVM',
}

# EMBEDDINGS
# Encoding Algorithms
ENCODING_ALGORITHM = {
    'CAPTIONS': 'EMBEDDINGS' # TFIDF or EMBEDDINGS
}
# Train embeddings
EMBEDDINGS_TRAINING = False
# Model's path
EMBEDDINGS_MODEL = {
    'short': 'models/captions-embeddings-short-weights-97-0.0051547588.hdf5',
    'long': 'models/captions-embeddings-long-weights-188-0.0198000782.hdf5'
}

# PRE-TRAINED CNN
PRE_TRAINED_NN = 'ResNet152'

# FINE-TUNED CNN
FINE_TUNE_EVALUATE_ONLY = True
FINE_TUNED_MODEL = 'models/fine-tuned-cnn-weights-01-0.0223143869.hdf5'

# EMOTIONS NN
EMOTIONS_NN = True
EMOTIONS_MODEL = 'models/emotions-weights-13-0.0458815261.hdf5'

# CURRENT DIR
current_dir_path = os.path.dirname(os.path.realpath(__file__))

## LOGS and CHECKPOINTS
RUN_NAME = datetime.datetime.now().strftime('run-%Y-%m-%d_%H-%M-%S')
LOG_DIR = os.path.join(current_dir_path, '..', 'logs')
RUN_LOG_DIR = os.path.join(LOG_DIR, RUN_NAME)
RUN_LOG_FOLD_DIR = os.path.join(RUN_LOG_DIR, 'fold_{}')
RUN_CHECKPOINT_DIR = os.path.join(RUN_LOG_DIR, 'checkpoints')

# HD
HD_DIR = '/datasets'

# DATA
DATA_DIR = os.path.join(HD_DIR, 'memorability', 'campus.pub.ro')

# DEV & TEST
DEV_DIR = os.path.join(DATA_DIR, 'devset', 'dev-set')
TEST_DIR = os.path.join(DATA_DIR, 'testset', 'me19me-testset', 'test-set')

# GROUNDTRUTH
DEV_GROUNDTRUTH = os.path.join(DEV_DIR, 'ground-truth', 'ground-truth_dev-set.csv')
TEST_GROUNDTRUTH = os.path.join(DATA_DIR, 'testset', 'test-set_list.txt')

# CAPTIONS
DEV_CAPTIONS = os.path.join(DEV_DIR, 'dev-set_video-captions.txt')
# DEV_CAPTIONS_2 = os.path.join(DEV_DIR, 'dev-set_video-captions2.txt')
TEST_CAPTIONS = os.path.join(TEST_DIR, 'test-set_videos-captions.txt')

# PRE-TRAINED EMBEDDINGS
GLOVE_FILE = os.path.join(HD_DIR, 'glove', 'glove.6B.300d.txt')
EMBEDDING_DIM = 300

# FEATURES
DEV_FEATURES = os.path.join(DEV_DIR, 'features')
DEV_FEATURES_PATH = {
    'C3D': os.path.join(DEV_FEATURES, 'C3D'),
    'AESTHETICS': os.path.join(DEV_FEATURES, 'aesthetic_visual_features', 'aesthetic_feat_dev-set_mean'),
    'HMP': os.path.join(DEV_FEATURES, 'HMP'),
    'ColorHistogram': os.path.join(DEV_FEATURES, 'ColorHistogram'),
    'HOG': os.path.join(DEV_FEATURES, 'HOG'),
    'LBP': os.path.join(DEV_FEATURES, 'LBP'),
    'InceptionV3': os.path.join(DEV_FEATURES, 'InceptionV3'),
}
TEST_FEATURES = os.path.join(TEST_DIR, 'features')
TEST_FEATURES_PATH = {
    'C3D': os.path.join(TEST_FEATURES, 'C3D'),
    'AESTHETICS': os.path.join(TEST_FEATURES, 'Aesthetics', 'aesthetics_mean'),
    'HMP': os.path.join(TEST_FEATURES, 'HMP'),
    'ColorHistogram': os.path.join(TEST_FEATURES, 'ColorHistogram'),
    'HOG': os.path.join(TEST_FEATURES, 'HOG'),
    'LBP': os.path.join(TEST_FEATURES, 'LBP'),
    'InceptionV3': os.path.join(TEST_FEATURES, 'InceptionV3'),
}
FEATURES_DIM = {
    'C3D': 101,
    'AESTHETICS': 109,
    'HMP': None,
    'ColorHistogram': None,
    'LBP': 122 * 3,
    'InceptionV3': None,
}

# FRAMES
# First, Middle and Last frames
THREE_FRAMES = [0, 56, 112]
# 8 frames: one per second
FRAME_NUMBERS = [1, 24, 48, 72, 96, 120, 144, 168]
# SOME FRAMES
FRAME_SAMPLES = 'figures/frame_samples/'

# SOURCES
DEV_SOURCES = os.path.join(DEV_DIR, 'sources') # 8,000 videos
# DEV_SOURCES_2 = os.path.join(DEV_DIR, 'sources2') # 6,000 videos
TEST_SOURCES = os.path.join(TEST_DIR, 'sources') # 2,000 videos

# FRAMES
DEV_FRAMES = os.path.join(DEV_DIR, 'frames')
DEV_NUMPY_FRAMES = os.path.join(DEV_DIR, 'npy_frames')
TEST_FRAMES = os.path.join(TEST_DIR, 'frames')

# PROCESSED FEATURES
MY_FEATURES_DIR = os.path.join(HD_DIR, 'processed')

# MY FEATURES
# EMOTIONS
EMOTIONS_DIR = os.path.join(HD_DIR, 'emotions')
EMOTIONS_DEV = os.path.join(EMOTIONS_DIR, 'devset')
EMOTIONS_TEST = os.path.join(EMOTIONS_DIR, 'testset')
# AESTHETICS
OUR_AESTHETICS_DIR = os.path.join(HD_DIR, 'feiyan_aesthetics')
OUR_AESTHETICS_DEV = os.path.join(OUR_AESTHETICS_DIR, 'aesthetic_feature_dev')
OUR_AESTHETICS_TEST = os.path.join(OUR_AESTHETICS_DIR, 'aesthetic_feature_test')

# PREDICTIONS
PREDICTIONS_DIR = 'predictions'
PREDICTIONS_TRAIN = os.path.join(PREDICTIONS_DIR, 'training')
PREDICTIONS_TEST = os.path.join(PREDICTIONS_DIR, 'test')

# TOKENIZER
CAPTIONS_EMBEDDINGS_TOKENIZER = os.path.join(MY_FEATURES_DIR, 'embeddings_tokenizer.pickle')
CAPTIONS_MAX_SEQUENCE_LENGTH = os.path.join(MY_FEATURES_DIR, 'max_sequence_length.txt')
