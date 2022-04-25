import sys, os.path as path
from inspect import getsourcefile

parent_dir = path.dirname(path.dirname(path.abspath(getsourcefile(lambda: 0))))
sys.path.append(path.join(parent_dir, 'BITES'))
