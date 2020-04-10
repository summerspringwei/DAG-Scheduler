from utils import *
# import read_profile_data
import read_profile_data
import os

result = os.popen("cat test.txt").read()
print(result)
com = result.split("\n")
print(com)
