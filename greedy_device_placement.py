#! /usr/bin/python

import mace_pb2
import read_inception

class Operator:
  def __init__(self, name):
    self.name = name
    self.parents = set()
    self.children = set()
    self.op_def = 0
  
  def __str__(self):
    return self.name + " " + self.op_def.type + " " + str(self.parents) + " " + str(self.children)


def build_relationship_for_op():
  netdef = read_inception.read_netdef("inception_v3_latency.pb")
  ops_relation_dict = dict()
  # For each op, find its parents and childs
  for i in range(len(netdef.op)):
    opdef1 = netdef.op[i]
    op = Operator(opdef1.name)
    op.op_def = opdef1
    for j in range(len(netdef.op)):
      if i == j:
        continue
      opdef2 = netdef.op[j]
      # find parents
      for input in opdef1.input:
        for output in opdef2.output:
          if input == output:
            op.parents.add(opdef2.name)
      # find childs
      for output in opdef1.output:
        for input in opdef2.input:
          if output == input:
            op.children.add(opdef2.name)
      ops_relation_dict[opdef1.name] = op
  for key in ops_relation_dict.keys():
    print(ops_relation_dict[key])
  print(len(ops_relation_dict))
  return netdef, ops_relation_dict
  


if __name__ == "__main__":
  netdef = read_inception.read_netdef("inception_v3_latency.pb")
  ops_relation_dict = dict()
  # For each op, find its parents and childs
  for i in range(len(netdef.op)):
    opdef1 = netdef.op[i]
    op = Operator(opdef1.name)
    op.op_def = opdef1
    for j in range(len(netdef.op)):
      if i == j:
        continue
      opdef2 = netdef.op[j]
      # find parents
      for input in opdef1.input:
        for output in opdef2.output:
          if input == output:
            op.parents.add(opdef2.name)
      # find childs
      for output in opdef1.output:
        for input in opdef2.input:
          if output == input:
            op.children.add(opdef2.name)
      ops_relation_dict[opdef1.name] = op
  for key in ops_relation_dict.keys():
    print(ops_relation_dict[key])
  print(len(ops_relation_dict))