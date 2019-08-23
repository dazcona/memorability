
import os 
import datetime

# Column to predict
TARGET = 'short-term_memorability'
# Target columns
TARGET_COLS = [ 'short-term_memorability', 'nb_short-term_annotations', 'long-term_memorability', 'nb_long-term_annotations' ]
# Dictionary that indicates which data sources to use in the model
FEATURES_DICT = {
    'CAPTIONS': True,
    'C3D': False,
    'AESTHETICS': False,
    'HMP': False,
}
FEATURES_WEIGHTS = {
    'CAPTIONS': 1,
    'C3D': 0,
    'AESTHETICS': 0,
    'HMP': 0,
}
# Algorithm
ALGORITHM = 'Bayesian Ridge'
# Number of folds for Cross-Validation
NFOLDS = 10
# Captions approach
CAPTIONS_ALGORITHM = 'EMBEDDINGS' # TFIDF or EMBEDDINGS
# EPOCHS
NUM_EPOCHS = 30

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
DATA_DIR = '/datasets' # '/Volumes/Samsung_T5/Memorability/data/raw/Memorability 2018/' mounted as datasets in docker-compose.yml

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
DEV_FEATURES_LIST = ['C3D', 'HMP', 'InceptionV3', 'LBP', 'aesthetic_feat_dev-set_mean', 'ColorHistogram', 'HOG', 'ORB']
DEV_C3D_FEATURE = os.path.join(DEV_FEATURES, 'C3D')
DEV_AESTHETIC_FEATURE = os.path.join(DEV_FEATURES, 'aesthetic_feat_dev-set_mean')
DEV_HMP_FEATURE = os.path.join(DEV_FEATURES, 'HMP')
DEV_HOG_FEATURE = os.path.join(DEV_FEATURES, 'HOG')
C3D_DIM = 101
AESTHETIC_DIM = 109
HMP_DIM = 6075
HOG_DIM = 'variable'

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