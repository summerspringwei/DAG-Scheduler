#! /usr/bin/python

import mace_pb2
import read_inception


# enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3, HTA = 4, APU = 5 };
# Follow the mace
class DeviceType:
  CPU = 0
  GPU = 2
  HEXAGON = 3
  HTA = 4
  APU = 5


class Operator:
  def __init__(self, name):
    self.name = name
    self.parents = set()
    self.children = set()
    self.op_def = 0
    self.executed = False
    self.data_format = DeviceType.CPU # 0 for CPU, 1 for GPU
    self.earlist_start_point = 0.0
  
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


def update_children_start_point(op, ops_relation_dict, device_start_point, latency):
  for child_name in op.children:
    op_child = ops_relation_dict[child_name]
    op_child.earlist_start_point = max(op_child.earlist_start_point, device_start_point + latency)


def sort_operator(operator):
  return operator.earlist_start_point


def assign_op_to_device(op, opsops_relation_dict, device_type, device_end_point, latency):
  op.executed = True
  op.netdef.device_type = device_type
  update_children_start_point(op, ops_relation_dict, device_end_point, latency)
  return device_end_point + latency


def is_parents_executed(op, ops_relation_dict):
  ready = True # When all his father has been executed, then the op can start executing
  for op_parent_name in op.parents:
    op_parent = ops_relation_dict[op_parent_name]
    if op_parent.executed == False:
      ready = False
  return ready


# enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3, HTA = 4, APU = 5 };
# Follow the mace
def greedy_device_placement(netdef, ops_relation_dict):
  input_node_name = "fc9d2ee0"
  CPU_end_point = 0.0
  GPU_end_point = 0.0
  # Execute the first op
  op = ops_relation_dict[input_node_name]
  if op.netdef.operatorLatency.CPU_latency < \
    op.netdef.operatorLatency.GPU_latency + op.netdef.operatorLatency.Transpose_latency_NHWC_to_NCHW:
    CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, op.netdef.operatorLatency.CPU_latency)
  else:
    GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, op.netdef.operatorLatency.GPU_latency)

  ops_queue = list()
  

  while(1):
    # Add child to ops_queue
    for child_name in op.children:
      if is_parents_executed(ops_relation_dict[child_name], ops_relation_dict):
        ops_queue.append(ops_relation_dict[child_name])
    # All ops are assigned to devices
    if(len(ops_queue) <= 0):
      break
    # Sort queue according to start point
    ops_queue.sort(key=sort_operator)
    # Fetch an op from queue
    for op_in_queue in ops_queue:
      # When all his father has been executed, then the op can start executing
      if is_parents_executed(op_in_queue, ops_relation_dict):
        op = op_in_queue
        ops_queue.remove(op_in_queue)
        break
    # For ops that are not supported by GPU, set their device type as CPU(Fall back to CPU)
    if op.netdef.type == "Concat":
      op.netdef.device_type = DeviceType.CPU
      continue
    # Assign the op to CPU or GPU
    # Find its father, get transpose latency
    to_CPU_transpose_latency = 0.0
    to_GPU_transpose_latency = 0.0
    for op_parent_name in op.parents:
      op_parent = ops_relation_dict[op_parent_name]
      if op_parent.op_def.device_type == DeviceType.CPU:
        to_GPU_transpose_latency += op_parent.op_def.operatorLatency.Transpose_latency_NCHW_to_NHWC
      elif op_parent.op_def.device_type == DeviceType.GPU:
        to_CPU_transpose_latency += op_parent.op_def.operatorLatency.Transpose_latency_NHWC_to_NCHW
    # Get computation latency on devices
    CPU_latency = op.netdef.operatorLatency.CPU_latency + to_CPU_transpose_latency
    GPU_latency = op.netdef.operatorLatency.GPU_latency + to_GPU_transpose_latency
    # TODO(xcw)add to_GPU_transpose_latency to CPU_end_point
    # op can be executed at the very first time, but CPU and GPU are busy
    if CPU_end_point >= op.earlist_start_point and GPU_end_point >= op.earlist_start_point:
      if CPU_end_point + CPU_latency < GPU_end_point + GPU_latency: # CPU is better
        CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency)
      else: # GPU is better
        GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency)
    
    # One device is ready but the other one is busy(or just finish work)
    elif (op.earlist_start_point >= CPU_end_point and op.earlist_start_point <= GPU_end_point):
      if op.earlist_start_point + CPU_latency < GPU_end_point + GPU_latency:
        CPU_end_point = op.earlist_start_point
        CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency)
      else:
        GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency)
    elif(op.earlist_start_point <= CPU_end_point and op.earlist_start_point >= GPU_end_point):
      if op.earlist_start_point + GPU_latency < CPU_end_point + CPU_latency:
        GPU_end_point = op.earlist_start_point
        GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency)
      else:
        CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency)
    else:
      if CPU_latency < GPU_latency:
        CPU_end_point = op.earlist_start_point
        CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency)
      else:
        GPU_end_point = op.earlist_start_point
        GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency)
    
        
    
      

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