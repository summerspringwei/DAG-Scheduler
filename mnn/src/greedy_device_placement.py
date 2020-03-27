#! /usr/bin/python
import logging
import os
from read_profile_data import *
from utils import *

def update_children_start_point(op, ops_relation_dict, \
  device_start_point, latency):
  for child_name in op.children:
    if child_name in ops_relation_dict:
      op_child = ops_relation_dict[child_name]
      op_child.earlist_start_point = \
        max(op_child.earlist_start_point, device_start_point + latency)
    else:
      print("Can not find op %s in dict" % child_name)


def key_sort_operator(operator):
  return operator.earlist_start_point


# Place the op on CPU or GPU, return the updated device end point
def assign_op_to_device(op, ops_relation_dict, device_type, device_end_point, latency):
  op.executed = True
  op.op_def.device_type = device_type
  update_children_start_point(op, ops_relation_dict, device_end_point, latency)
  return device_end_point + latency


def is_parents_executed(op, ops_relation_dict):
  ready = True # When all his father has been executed, then the op can start executing
  for op_parent_name in op.parents:
    if not op_parent_name in ops_relation_dict.keys():
      raise KeyError()
    op_parent = ops_relation_dict[op_parent_name]
    if op_parent.executed == False:
      ready = False

  return ready


def write_execute_order(filename, op_execute_order):
  f = open(filename, 'w')
  for idx in op_execute_order:
    f.write(str(idx) + " ")
  f.flush()
  f.close()


def write_device_placement(filename, net_def):
  f = open(filename, 'w')
  for op in net_def.op:
    f.write("%s %d\n" % (op.name, op.op_def.device_type))
  f.flush()
  f.close()
  print("Write device placement done.")

# Follow the mace
def greedy_device_placement(netdef, ops_relation_dict, folder_path, model_name, mobile):
  ops_not_support_by_GPU = set(['concat', 'SpatialSqueeze', 'Shape', 'Reshape', 'Softmax', 'Reshape_1'])
  # Record the CPU queue and GPU queue finish timestamp
  CPU_end_point = 0.0
  GPU_end_point = 0.0
  # Need to record the execute order
  op_execute_order = list()
  idx = 0
  op_to_idx_dict = dict()
  for op in netdef.op:
    op_to_idx_dict[op.name] = idx
    idx += 1
  
  # Execute the first op
  op = netdef.op[0]
  GPU_latency = op.op_def.operatorLatency.GPU_latency + op.op_def.operatorLatency.Transpose_latency_NHWC_to_NCHW
  if op.op_def.operatorLatency.CPU_latency < GPU_latency:
    CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, op.op_def.operatorLatency.CPU_latency)
  else:
    GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency)
  op_execute_order.append(op_to_idx_dict[op.name])
  ops_queue = list()
  for op_name, op_const in ops_relation_dict.items():
    if len(op_const.parents) == 0:
      ops_queue.append(op_const)
  # Start greedy assign
  while(True):
    # Add child to ops_queue if all his parents has been executed
    for child_name in op.children:
      if is_parents_executed(ops_relation_dict[child_name], ops_relation_dict) and\
        ops_relation_dict[child_name] not in ops_queue:
        ops_queue.append(ops_relation_dict[child_name])
        print("Add %s" % (child_name))
    # All ops are assigned to devices, stop
    if(len(ops_queue) <= 0):
      break
    # Sort queue according to start point
    ops_queue.sort(key=key_sort_operator)
    # Fetch an op from queue
    for op_in_queue in ops_queue:
      # When all his father has been executed, then the op can start executing
      if is_parents_executed(op_in_queue, ops_relation_dict):
        op = op_in_queue
        ops_queue.remove(op_in_queue)
        logging.debug("Fetch op %s " % op.name)
        break
    # Record the execute index
    op_execute_order.append(op_to_idx_dict[op.name])
    # For ops that are not supported by GPU, set their device type as CPU(Fall back to CPU)
    if op.op_def.type in ops_not_support_by_GPU:
      CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, op.op_def.operatorLatency.CPU_latency)
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
    CPU_latency = op.op_def.operatorLatency.CPU_latency + to_CPU_transpose_latency
    GPU_latency = op.op_def.operatorLatency.GPU_latency + to_GPU_transpose_latency
    logging.debug("op %s CPU and GPU endpoint: %f %f " % ( op.name, CPU_end_point, GPU_end_point))
    logging.debug("op %s CPU and GPU latency: %f %f to CPU and to GPU latency: %f %f" \
      % ( op.name, CPU_latency, GPU_latency, to_CPU_transpose_latency, to_GPU_transpose_latency))
    # TODO(xcw)add to_GPU_transpose_latency to CPU_end_point
    # Op can be executed at the very first time, but CPU and GPU are busy
    if CPU_end_point >= op.earlist_start_point and GPU_end_point >= op.earlist_start_point:
      if CPU_end_point + CPU_latency < GPU_end_point + GPU_latency: # CPU can finish this op earlier(Greedy here)
        CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency)
      else: # GPU is better
        GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency)
    
    # One device is ready but the other one is busy(or just finish work)
    elif (op.earlist_start_point >= CPU_end_point and op.earlist_start_point <= GPU_end_point):
      if op.earlist_start_point + CPU_latency < GPU_end_point + GPU_latency:# Note, CPU_end_point changed
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
  # End of while
  # for op in netdef.op:
  #   print(op.name + " " + str(op.op_def.device_type))
  
  write_execute_order(os.path.join(folder_path, "op_execute_order" + mobile + "-" + model_name +".txt") , op_execute_order)
  write_device_placement(os.path.join(folder_path, 'greedy-' + mobile + "-" + model_name + '-device-placement.txt') , netdef)
  print("CPU end point: %s ms." % CPU_end_point)
  print("GPU end point: %s ms." % GPU_end_point)
  print(op_execute_order)
  


if __name__ == "__main__":
  logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
  model, mobile, thread = parse_model_mobile()
  model_dir = os.path.join("../models/", model)
  folder_path = os.path.join(model_dir, mobile)
  op_name_list, name_op_dict, net_def = gather_model_profile(
        os.path.join(model_dir, model + "-info.txt"),
        "../models/inception-v3/redmi_data_trans.txt",
        os.path.join(model_dir, mobile, mobile+"-"+model+"-layerwise-latency.csv"),
        thread)
  greedy_device_placement(net_def, name_op_dict, folder_path, model, mobile)
