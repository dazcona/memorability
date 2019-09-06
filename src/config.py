import os
import datetime

# Column to predict
TARGET = 'short-term_memorability'
# Target columns
TARGET_COLS = [ 'short-term_memorability', 'nb_short-term_annotations', 'long-term_memorability', 'nb_long-term_annotations' ]
# Dictionary that indicates which data sources to use in the model and its weight on the overall ensemble model
FEATURES_WEIGHTS = {
    'C3D': 0,
    'AESTHETICS': 0,
    'HMP': 0.1,
    'ColorHistogram': 0,
    'LBP': 0,
    'InceptionV3': 0,
    'CAPTIONS': 0.1,
    'PRE-TRAINED NN': 0.8,
}
# Number of folds for Cross-Validation
NFOLDS = 10
# Encoding Algorithms
ENCODING_ALGORITHM = {
    'CAPTIONS': 'EMBEDDINGS' # TFIDF or EMBEDDINGS
}
# Train embeddings
EMBEDDINGS_TRAINING = False
# Model's path
EMBEDDINGS_MODEL = 'checkpoints/run-2019-09-03_16-40-09/weights-fold_0-97-0.0051547588.hdf5'
# Pre-trained CNN
PRE_TRAINED_NN = 'ResNet152'
# GRID SEARCH
GRID_SEARCH = False
# Features Algorithms
FEATURES_ALGORITHM = {
    'C3D': 'Bayesian Ridge',
    'AESTHETICS': 'Bayesian Ridge',
    'HMP': 'SVM',
    'ColorHistogram': 'Bayesian Ridge',
    'LBP': 'Bayesian Ridge',
    'InceptionV3': 'Bayesian Ridge',
    'CAPTIONS': 'SVM',
    'PRE-TRAINED NN': 'SVM',
}
# EPOCHS
NUM_EPOCHS = 50

# CURRENT DIR
current_dir_path = os.path.dirname(os.path.realpath(__file__))

## LOGS and CHECKPOINTS
RUN_NAME = datetime.datetime.now().strftime('run-%Y-%m-%d_%H-%M-%S')
LOG_DIR = os.path.join(current_dir_path, '..', 'logs')
RUN_LOG_DIR = os.path.join(LOG_DIR, RUN_NAME)
RUN_LOG_FOLD_DIR = os.path.join(RUN_LOG_DIR, 'fold_{}')
CHECKPOINT_DIR = os.path.join(current_dir_path, '..', 'checkpoints')
RUN_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, RUN_NAME)

# DATA
DATA_DIR = '/datasets'

# DEV
DEV_DIR = os.path.join(DATA_DIR, 'devset', 'dev-set')
DEV_GROUNDTRUTH = os.path.join(DEV_DIR, 'ground-truth', 'ground-truth_dev-set.csv')

# CAPTIONS
DEV_CAPTIONS = os.path.join(DEV_DIR, 'dev-set_video-captions.txt')
# DEV_CAPTIONS_2 = os.path.join(DEV_DIR, 'dev-set_video-captions2.txt')

# EMBEDDINGS
GLOVE_FILE = os.path.join('data', 'glove.6B.300d.txt')
EMBEDDING_DIM = 300

# FEATURES
DEV_FEATURES = os.path.join(DEV_DIR, 'features')
DEV_C3D_FEATURE = os.path.join(DEV_FEATURES, 'C3D')
DEV_AESTHETIC_FEATURE = os.path.join(DEV_FEATURES, 'aesthetic_visual_features', 'aesthetic_feat_dev-set_mean')
DEV_HMP_FEATURE = os.path.join(DEV_FEATURES, 'HMP')
DEV_ColorHistogram_FEATURE = os.path.join(DEV_FEATURES, 'ColorHistogram')
DEV_HOG_FEATURE = os.path.join(DEV_FEATURES, 'HOG')
DEV_LBP_FEATURE = os.path.join(DEV_FEATURES, 'LBP')
DEV_InceptionV3_FEATURE = os.path.join(DEV_FEATURES, 'InceptionV3')
FEATURES_PATH = {
    'C3D': DEV_C3D_FEATURE,
    'AESTHETICS': DEV_AESTHETIC_FEATURE,
    'HMP': DEV_HMP_FEATURE,
    'ColorHistogram': DEV_ColorHistogram_FEATURE,
    'LBP': DEV_LBP_FEATURE,
    'InceptionV3': DEV_InceptionV3_FEATURE,
}
FEATURES_DIM = {
    'C3D': 101,
    'AESTHETICS': 109,
    'HMP': None,
    'ColorHistogram': None,
    'LBP': 122 * 3,
    'InceptionV3': None,
}
# First, Middle and Last frames
THREE_FRAMES = [0, 56, 112]

# SOURCES
DEV_SOURCES = os.path.join(DEV_DIR, 'sources') # 8,000 videos
# DEV_SOURCES_2 = os.path.join(DEV_DIR, 'sources2') # 6,000 videos

# FRAMES
DEV_FRAMES = os.path.join(DEV_DIR, 'frames')

# TEST
TEST_GROUNDTRUTH = os.path.join(DATA_DIR, 'testset', 'test-set_list.txt')
TEST_DIR = os.path.join(DATA_DIR, 'testset', 'me19me-testset', 'test-set')
TEST_CAPTIONS = os.path.join(TEST_DIR, 'test-set_video-captions.txt')
TEST_FEATURES = os.path.join(TEST_DIR, 'features')
TEST_FEATURES_LIST = ['C3D', 'HMP', 'InceptionV3', 'LBP', 'Aesthetics', 'ColorHistogram', 'HOG', 'ORB']
TEST_SOURCES = os.path.join(TEST_DIR, 'sources') # 2,000 videos
TEST_FRAMES = os.path.join(TEST_DIR, 'frames')

# ONLY RUN ONCE
ONLY_RUN_ONE_FOLD_CV = True
