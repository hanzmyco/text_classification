PROJECT_NAME='weixin_12_25'
# parameters for preprocessing the dataset

ORIGIN_DATA = '../../data/'+PROJECT_NAME+'/all.txt'
ORIGIN_LABEL='../../data/'+PROJECT_NAME+'/label.txt'
TOKENIZED_DATA = '../../data/'+PROJECT_NAME+'/all.txt.tok'
PROCESSED_LABEL = '../../data/'+PROJECT_NAME+'/processed_label.txt'
ID_DATA ='../../data/'+PROJECT_NAME+'/all.ids'
TRAIN_FILES_OUT = '../../data/'+PROJECT_NAME+'/train/data'
TEST_FILES_OUT = '../../data/'+PROJECT_NAME+'/test/data'
TRAIN_LABELS_OUT = '../../data/'+PROJECT_NAME+'/train/label'
TEST_LABELS_OUT = '../../data/'+PROJECT_NAME+'/test/label'
TRAIN_DIRECTORY = TRAIN_FILES_OUT[:TRAIN_FILES_OUT.rfind('/')+1]
TEST_DIRECTORY = TEST_FILES_OUT[:TEST_FILES_OUT.rfind('/')+1]
