
import os 

# Current dir
# current_dir_path = os.path.dirname(os.path.realpath(__file__))

# Data dir
DATA_DIR = '/datasets' # '/Volumes/Samsung_T5/Memorability/data/raw/Memorability 2018/' mounted as datasets in docker-compose.yml

DEV_DIR = os.path.join(DATA_DIR, 'dev-set')
DEV_GROUNDTRUTH = os.path.join(DEV_DIR, 'ground-truth', 'ground-truth_dev-set.csv')

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

DEV_CAPTIONS = os.path.join(DEV_DIR, 'dev-set_video-captions.txt')
DEV_CAPTIONS_2 = os.path.join(DEV_DIR, 'dev-set_video-captions2.txt')

DEV_SOURCES = os.path.join(DEV_DIR, 'sources') # 8,000 videos
DEV_SOURCES_2 = os.path.join(DEV_DIR, 'sources2') # 6,000 videos

TEST_DIR = os.path.join(DATA_DIR, 'test-set')
TEST_FEATURES = os.path.join(TEST_DIR, 'features')
TEST_FEATURES_LIST = ['C3D', 'HMP', 'InceptionV3', 'LBP', 'Aesthetics', 'ColorHistogram', 'HOG', 'ORB']
TEST_CAPTIONS = os.path.join(TEST_DIR, 'test-set_video-captions.txt')
TEST_SOURCES = os.path.join(TEST_DIR, 'sources') # 2,000 videos