
from measure_inteference import *
import numpy as np


tmp_dict = {'a':1, 'b':2}

def update_dict(di):
    di['c'] = 3
    di['a'] += 1

update_dict(tmp_dict)
print(tmp_dict)
