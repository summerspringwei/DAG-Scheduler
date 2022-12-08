# from sklearn.cluster import KMeans
# import numpy as np
# X = np.array([[32], [32], [32], [16], [16], [16], [16], \
#     [16], [32], [8], [8], [8], [8], [4], [4], \
#         [8], [8], [8], [4], [4]])
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# print(kmeans.labels_)
from profile import read_profile_data
from utils import utils
import logging
import xml.etree.ElementTree as ET


class Student:
    def __init__(self, name):
        self.name = name
        self.age = 0

class BottomLevelFuncType:
  COMPUTE = 0
  COMPUTE_COMM = 1
  RANK = 2


def func(a):
  if a> 1:
    c = 2
  else:
    c = 3
  return c

def parse_cplex(cplex_result_file_path):
  tree = ET.parse(cplex_result_file_path)
  root = tree.getroot()
  for child in root:
    if child.tag == "variables":
      for variable in child:
        print(variable.attrib['name'], variable.attrib['value'])


if __name__ == "__main__":
    # print(utils.get_project_path())
    # print(isinstance(BottomLevelFuncType.COMPUTE_COMM, BottomLevelFuncType))
    # print(func(3))
    parse_cplex('/Users/xiachunwei/Projects/DAG-Scheduler/mnn/src/tmp.xml')