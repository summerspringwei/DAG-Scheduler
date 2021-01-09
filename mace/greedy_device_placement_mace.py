#! /usr/bin/python
import logging
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


def build_relationship_for_op(file_name):
  netdef = read_inception.read_netdef(file_name)
  ops_relation_dict = dict()
  # For each op, find its parents and childs
  for i in range(len(netdef.op)):
    opdef1 = netdef.op[i]
    op = Operator(opdef1.name)
    op.op_def = opdef1
    # Skip itself
    for j in range(len(netdef.op)):
      if i == j:
        continue
      opdef2 = netdef.op[j]
      # Find parents
      for input in opdef1.input:
        for output in opdef2.output:
          if input == output:
            op.parents.add(opdef2.name)
      # Find children
      for output in opdef1.output:
        for input in opdef2.input:
          if output == input:
            op.children.add(opdef2.name)
    ops_relation_dict[opdef1.name] = op
  for key in ops_relation_dict.keys():
    print(ops_relation_dict[key])
  print(len(ops_relation_dict))
  return netdef, ops_relation_dict


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
def assign_op_to_device(op, opsops_relation_dict, device_type, device_end_point, latency):
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


# enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3, HTA = 4, APU = 5 };
# Follow the mace
def greedy_device_placement(netdef, ops_relation_dict):
  input_node_name = "fc9d2ee0"
  # input_node_name = "op1"
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
  op = ops_relation_dict[input_node_name]
  GPU_latency = op.op_def.operator_latency.GPU_latency + op.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW
  if op.op_def.operator_latency.CPU_latency < GPU_latency:
    CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, op.op_def.operator_latency.CPU_latency)
  else:
    GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency)
  op_execute_order.append(op_to_idx_dict[input_node_name])
  ops_queue = list()
  # Start greedy assign
  while(True):
    # Add child to ops_queue if all his parents has been executed
    for child_name in op.children:
      if is_parents_executed(ops_relation_dict[child_name], ops_relation_dict) and\
        ops_relation_dict[child_name] not in ops_queue:
        ops_queue.append(ops_relation_dict[child_name])
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
    if op.op_def.type == "Concat":
      CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, op.op_def.operator_latency.CPU_latency)
      continue
    # Assign the op to CPU or GPU
    # Find its father, get transpose latency
    to_CPU_transpose_latency = 0.0
    to_GPU_transpose_latency = 0.0
    for op_parent_name in op.parents:
      op_parent = ops_relation_dict[op_parent_name]
      if op_parent.op_def.device_type == DeviceType.CPU:
        to_GPU_transpose_latency += op_parent.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC
      elif op_parent.op_def.device_type == DeviceType.GPU:
        to_CPU_transpose_latency += op_parent.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW
    # Get computation latency on devices
    CPU_latency = op.op_def.operator_latency.CPU_latency + to_CPU_transpose_latency
    GPU_latency = op.op_def.operator_latency.GPU_latency + to_GPU_transpose_latency
    logging.debug("op %s CPU and GPU endpoint: %f %f " % ( op.name, CPU_end_point, GPU_end_point))
    logging.debug("op %s CPU and GPU latency: %f %f " % ( op.name, CPU_latency, GPU_latency))
    # TODO(xcw)add to_GPU_transpose_latency to CPU_end_point
    # op can be executed at the very first time, but CPU and GPU are busy
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
  for op in netdef.op:
    # print("%s %s %s" % op.name, op.type, str(op.device_type))
    #print(op.name + " " + op.type + " " + str(op.device_type))
    print(op.name + " " + str(op.device_type))
  write_execute_order("op_execute_order.txt", op_execute_order)
  print("CPU end point: %s ms." % CPU_end_point)
  print("GPU end point: %s ms." % GPU_end_point)
  print(op_execute_order)
  scheduled_net_def = mace_pb2.NetDef()
  for arg in netdef.arg:
    new_arg = scheduled_net_def.arg.add()
    new_arg.CopyFrom(arg)
  for tensor in netdef.tensors:
    new_tensor = scheduled_net_def.tensors.add()
    new_tensor.CopyFrom(tensor)
  for input_info in netdef.input_info:
    new_input_info = scheduled_net_def.input_info.add()
    new_input_info.CopyFrom(input_info)
  for output_info in netdef.output_info:
    new_output_info = scheduled_net_def.output_info.add()
    new_output_info.CopyFrom(output_info)
  for i in range(len(netdef.op)):
    idx = op_execute_order[i]
    new_op = scheduled_net_def.op.add()
    new_op.CopyFrom(netdef.op[idx])
  
  read_inception.write_bench_netdef("s_my_inception_v3.pb", scheduled_net_def)
  return scheduled_net_def


if __name__ == "__main__":
  logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
  #netdef, ops_relation_dict = build_relationship_for_op("my_dag.pb")
  netdef, ops_relation_dict = build_relationship_for_op("inception_v3_latency.pb")
  greedy_device_placement(netdef, ops_relation_dict)
