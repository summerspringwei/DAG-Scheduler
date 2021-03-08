import pysnooper
import queue

from profile import subgraph
from profile import net_struct
from utils import utils

logger = utils.get_logger()

def update_children_start_point(op, name_op_dict, \
  device_start_point, latency):
  for child_name in op.children:
    if child_name in name_op_dict:
      op_child = name_op_dict[child_name]
      op_child.earlist_start_point = \
        max(op_child.earlist_start_point, device_start_point + latency)
    else:
      print("Can not find op %s in dict" % child_name)


# Place the op on CPU or GPU, return the updated device end point
def assign_op_to_device(op, ops_relation_dict, device_type, device_end_point, latency, op_execute_order_list):
  op.executed = True
  op.op_def.device_type = device_type
  op_execute_order_list.append((op.name, device_type, device_end_point + latency))
  update_children_start_point(op, ops_relation_dict, device_end_point, latency)
  return device_end_point + latency


def is_parents_executed(op, op_name_list, ops_relation_dict, is_subgraph=False):
  ready = True # When all his father has been executed, then the op can start executing
  for op_parent_name in op.parents:
    if not op_parent_name in ops_relation_dict.keys():
      raise KeyError()
    if op_parent_name not in op_name_list:
      continue
    op_parent = ops_relation_dict[op_parent_name]
    if is_subgraph:
      if not isinstance(op_parent, subgraph.Subgraph):
        continue
    if op_parent.executed == False:
      ready = False
      break
  return ready


def write_device_placement(filename, net_def):
  f = open(filename, 'w')
  for op in net_def.op:
    f.write("%s %d\n" % (op.name, op.op_def.device_type))
  f.flush()
  f.close()
  print("Write device placement done.")


# def get_ops_total_latency(op, ops_relation_dict):
#   to_CPU_transpose_latency = 0.0
#   to_GPU_transpose_latency = 0.0
  
#   for op_parent_name in op.parents:
#     op_parent = ops_relation_dict[op_parent_name]
#     if op_parent.op_def.device_type == net_struct.DeviceType.CPU:
#       to_GPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC)
#     elif op_parent.op_def.device_type == net_struct.DeviceType.GPU:
#       to_CPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW)
  
#   # print(op.op_def.operator_latency)
#   CPU_latency = op.op_def.operator_latency.CPU_latency + to_CPU_transpose_latency
#   GPU_latency = op.op_def.operator_latency.GPU_latency + to_GPU_transpose_latency
#   return CPU_latency, GPU_latency

# @pysnooper.snoop()
def get_ops_total_latency(op, name_op_dict):
  """Get the op's execution latency on CPU and GPU
  based on the device placement result of op's parents
  considering the communication latency
  """
  to_CPU_transpose_latency = 0.0
  to_GPU_transpose_latency = 0.0
  utils.get_logger().info("op name: {}, input tensors:{}, data trans dict {}".format(op.name, op.input_tensors, op.op_def.operator_latency.input_data_trans_latency))
  for op_parent_name in op.parents:
    op_parent = name_op_dict[op_parent_name]
    for child_tensor_addr, child_tensor_shape in op.input_tensors:
      for parent_tensor_addr, parent_tensor_shape in op_parent.output_tensors:
        if child_tensor_addr == parent_tensor_addr:
          utils.get_logger().info("{} {} {}".format(op.name, op_parent.name, parent_tensor_shape))
          if child_tensor_addr not in op.op_def.operator_latency.input_data_trans_latency.keys():
            continue
          if op_parent.op_def.device_type == net_struct.DeviceType.CPU:
            to_GPU_transpose_latency += op.op_def.operator_latency.input_data_trans_latency[child_tensor_addr][1]
          elif op_parent.op_def.device_type == net_struct.DeviceType.GPU:
            to_CPU_transpose_latency += op.op_def.operator_latency.input_data_trans_latency[child_tensor_addr][0]
  utils.get_logger().info("{} {} {} {} {}".format(op.name, \
    op.op_def.operator_latency.CPU_latency, op.op_def.operator_latency.GPU_latency,\
     to_CPU_transpose_latency, to_GPU_transpose_latency))
  CPU_latency = op.op_def.operator_latency.CPU_latency + to_CPU_transpose_latency
  GPU_latency = op.op_def.operator_latency.GPU_latency + to_GPU_transpose_latency
  return CPU_latency, GPU_latency

def get_op_avg_computing_latency(op):
    return (op.op_def.operator_latency.CPU_latency \
      + op.op_def.operator_latency.GPU_latency) / 2.0

def get_op_avg_computing_comm_latency(op):
  return (op.op_def.operator_latency.CPU_latency \
    + op.op_def.operator_latency.GPU_latency) / 2.0 \
      + op.op_def.operator_latency.get_avg_data_trans_latency()

def get_op_rank(op):
  return 1

class BottomLevelFuncType:
  COMPUTE = 0
  COMPUTE_COMM = 1
  RANK = 2

bottom_level_func_map = {
  BottomLevelFuncType.COMPUTE: get_op_avg_computing_latency,
  BottomLevelFuncType.COMPUTE_COMM: get_op_avg_computing_comm_latency,
  BottomLevelFuncType.RANK: get_op_rank
}


def compute_bottom_level(op_name_list, name_op_dict, bottom_level_func_name=BottomLevelFuncType.COMPUTE):
    """Compute the bottom level of every operator in the DAG
    The bottom level bl(i) of a task vi ∈ V is defined 
    as the largest weight of a path from vi to a target node (vertex without successors), 
    including the weight wi of vi, and all communication costs.
    """
    output_node_list = [op_name for op_name in op_name_list if len(name_op_dict[op_name].children) == 0]
    logger.info("output nodes:")
    logger.info(output_node_list)
    ready = queue.Queue()
    visited = set()
    if bottom_level_func_name not in bottom_level_func_map.keys():
      logger.error("bottom_level_func_name is invalid")
      exit(0)
    get_op_bottom_level_func = bottom_level_func_map[bottom_level_func_name]
    # Init output operators' bottom_level
    for output_node_name in output_node_list:
        output_op = name_op_dict[output_node_name]
        output_op.bottom_level = get_op_bottom_level_func(output_op)
        ready.put(output_node_name)
        visited.add(output_node_name)
    while not ready.empty():
        op_name = ready.get()
        op = name_op_dict[op_name]
        visited.add(op_name)
        for op_parent_name in op.parents:
            op_parent = name_op_dict[op_parent_name]
            op_parent.bottom_level = max(op_parent.bottom_level, \
                op.bottom_level + get_op_bottom_level_func(op_parent))
            # Check if all op_parent ready
            isready = True
            for op_parent_child_name in op_parent.children:
              if op_parent_child_name not in visited:
                isready = False
                break
            if isready:
                ready.put(op_parent_name)
    logger.info("Ops bottom level:")
    for op_name in op_name_list:
        logger.info("{} {}".format(op_name, name_op_dict[op_name].bottom_level))
