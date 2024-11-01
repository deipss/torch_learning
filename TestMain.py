import sys
import os
project_root =os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'basics'))
from basics import *

if __name__ == '__main__':
    print(data_root_path)
    print(separator)