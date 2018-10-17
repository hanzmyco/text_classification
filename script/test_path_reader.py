import sys
sys.path.append('../src')
sys.path.append('../src/utils')
import config
import utils

print(config.PROJECT_NAME)
utils.safe_mkdir_depths('test/')
