import os
import datetime

# Column to predict
TARGET_SHORT_NAME = 'long' # 'short' or 'long'
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
EMBEDDINGS_TRAINING = True
# Model's path
EMBEDDINGS_MODEL = ''
# 'logs/run-2019-09-19_16-42-25/checkpoints/weights-fold_0-47-0.0197818204.hdf5' 
# 'logs/run-2019-09-16_16-10-58/checkpoints/weights-fold_0-23-0.0203232624.hdf5'
# long-term: 'logs/run-2019-09-16_16-10-58/checkpoints/weights-fold_0-23-0.0203232624.hdf5'
# short-term: 'checkpoints/run-2019-09-03_16-40-09/weights-fold_0-97-0.0051547588.hdf5'

# PRE-TRAINED CNN
PRE_TRAINED_NN = 'DenseNet121'

# FINE-TUNED CNN
FINE_TUNE_EVALUATE_ONLY = True
FINE_TUNED_MODEL = 'logs/run-2019-09-18_12-42-36/checkpoints/weights-part_1-01-0.0223143869.hdf5'
# 'logs/run-2019-09-18_09-51-01/checkpoints/weights-part_1-02-0.0229876925.hdf5'
# 'logs/run-2019-09-17_23-20-46/checkpoints/weights-part_0-01-0.0220884897.hdf5'
# 'logs/run-2019-09-16_23-38-20/checkpoints/weights-part_1-29-0.0068147990.hdf5'
# 'logs/run-2019-09-14_18-59-52/checkpoints/weights-part_1-43-0.0065387096.hdf5'
# 'logs/run-2019-09-14_15-40-55/checkpoints/weights-part_0-01-0.0100537471.hdf5'
# 'logs/run-2019-09-14_13-59-54/checkpoints/weights-part_0-01-0.0264736740.hdf5'
# 'models/fine-tuning-warmup-one-frame-weights-fold_0-01-0.0271032082.hdf5'
# 'models/fine-tuning-warmup-weights-fold_0-01-0.0264373418.hdf5' # 8 frames, 1 epoch, 0.25854 (p-value 0.00000)

# 'checkpoints/run-2019-09-13_14-44-43/weights-fold_0-06-17.9225301506.hdf5'
# 'checkpoints/run-2019-09-13_12-12-56/weights-fold_0-01-0.0247115156.hdf5'
# 'checkpoints/run-2019-09-13_11-33-32/weights-fold_0-01-0.0275342398.hdf5'
# 'checkpoints/run-2019-09-13_11-14-40/weights-fold_0-01-0.0275342398.hdf5'
# 'checkpoints/run-2019-09-12_17-57-21/weights-fold_0-01-0.0275577824.hdf5'
# 'checkpoints/run-2019-09-12_17-11-19/weights-fold_0-01-0.0275342398.hdf5'
# 'checkpoints/run-2019-09-12_16-05-08/weights-fold_0-01-0.0241449984.hdf5'
# 'checkpoints/run-2019-09-11_17-55-01/weights-fold_0-08-0.0069531268.hdf5'
# 'logs/run-2019-09-11_17-55-01/fine_tuning_model_after_warmup.h5'

EMOTIONS_NN = True
EMOTIONS_MODEL = 'logs/run-2019-09-19_14-26-02/checkpoints/emotions-weights-fold_0-13-0.0458815261.hdf5'
# 'logs/run-2019-09-18_12-29-21/checkpoints/emotions-weights-fold_0-270-0.0163350874.hdf5'
# 'logs/run-2019-09-18_12-08-36/checkpoints/emotions-weights-fold_0-111-0.0163384946.hdf5'

# EPOCHS
# NUM_EPOCHS = 100

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
TEST_CAPTIONS = os.path.join(TEST_DIR, 'test-set_video-captions.txt')

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
    'AESTHETICS': os.path.join(TEST_FEATURES, 'aesthetic_visual_features', 'aesthetic_feat_test-set_mean'),
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