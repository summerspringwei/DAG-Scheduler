#! /usr/bin/python

import os
import queue
from operator import itemgetter, attrgetter
import timeit
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger()

from profile import read_profile_data
from profile import graph_partition
from profile import find_critical_node
from profile import subgraph
from profile import net_struct
from solver import optimizer
from solver import search_tree
from solver import scheduler_utils
from utils import utils

search_window_size = 5

def key_sort_operator(operator):
  return operator.earlist_start_point


def greedy_device_placement_v3(op_name_list, name_op_dict, folder_path, model_name, mobile, thread):
  # ops_not_support_by_GPU = set(['concat', 'SpatialSqueeze', 'Shape', 'Reshape', 'Softmax', 'Reshape_1'])
  ops_not_support_by_GPU = []
  # Record the CPU queue and GPU queue finish timestamp
  CPU_end_point = 0.0
  GPU_end_point = 0.0
  
  # Need to record the execute order
  op_execute_order_list = []
  
  input_op_names = subgraph.find_input_nodes(op_name_list, name_op_dict)
  ops_queue = [name_op_dict[op_name] for op_name in input_op_names]
  print(input_op_names)
  assert(len(ops_queue) > 0)
  print("Start Greedy")
  
  # Start greedy assign
  to_be_sche_ops = []
  while(True):
    # Add child to ops_queue if all his parents has been executed
    child_list = []
    for op in to_be_sche_ops:
      child_list.extend(op.children)
    to_be_sche_ops.clear()
    for child_name in child_list:
      # If a child's parents are executed and this child is not executed and not in queue
      # Add this child to queue
      if child_name in op_name_list and \
        scheduler_utils.is_parents_executed(name_op_dict[child_name], op_name_list, name_op_dict) and \
        name_op_dict[child_name] not in ops_queue and \
        not name_op_dict[child_name].executed:
        ops_queue.append(name_op_dict[child_name])
    # All ops are assigned to devices, stop
    if(len(ops_queue) <= 0):
      break
    
    # Sort queue according to start point
    # ops_queue= sorted(ops_queue, key=key_sort_operator)
    ops_queue = sorted(ops_queue, key=attrgetter("earlist_start_point", "name"))
    # Using search tree to find optimal device placement strategy
    to_be_scheduled_op_names = [op.name for op in ops_queue][0:search_window_size]
    logger.info("Search tree schedule {} {}".format(len(to_be_scheduled_op_names), to_be_scheduled_op_names))
    root, leaf_node_list = search_tree.build_search_tree(to_be_scheduled_op_names, name_op_dict, [CPU_end_point, GPU_end_point])
    [CPU_end_point, GPU_end_point], device_placement = search_tree.get_optimal_device_placement(root, leaf_node_list, name_op_dict)
    utils.get_logger().info(device_placement)
    op_execute_order_list.extend(device_placement)
    
    for scheduled_op_name in to_be_scheduled_op_names:
      scheduled_op = name_op_dict[scheduled_op_name]
      to_be_sche_ops.append(scheduled_op)
      ops_queue.remove(scheduled_op)

  # End of while
  # for op in netdef.op:
  # print(op.name + " " + str(op.op_def.device_type))
  
  # write_device_placement(os.path.join(folder_path, 'greedy-' + mobile + "-" + model_name + '-cpu-' + str(thread) +  '.txt') , netdef)
  print("CPU end point: %s ms." % CPU_end_point)
  print("GPU end point: %s ms." % GPU_end_point)
  print("Greedy Result %f" % (max(CPU_end_point, GPU_end_point)))
  # print(op_execute_order_list)
  lines = subgraph.write_subgraph_device_placement_result(name_op_dict=name_op_dict, op_execution_order_list=op_execute_order_list)
  return lines



# Follow the mace
def greedy_device_placement_v2(op_name_list, ops_relation_dict, folder_path, model_name, mobile, thread):
  # ops_not_support_by_GPU = set(['concat', 'SpatialSqueeze', 'Shape', 'Reshape', 'Softmax', 'Reshape_1'])
  ops_not_support_by_GPU = []
  # Record the CPU queue and GPU queue finish timestamp
  CPU_end_point = 0.0
  GPU_end_point = 0.0
  
  # Need to record the execute order
  op_execute_order_list = []
  
  input_op_names = find_input_nodes(op_name_list, ops_relation_dict)
  ops_queue = [ops_relation_dict[op_name] for op_name in input_op_names]
  print(input_op_names)
  assert(len(ops_queue) > 0)
  print("Start Greedy")
  op = ops_queue[0]
  # Execute the first op
  GPU_latency = op.op_def.operator_latency.GPU_latency + op.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW
  if op.op_def.operator_latency.CPU_latency < GPU_latency:
    CPU_end_point = scheduler_utils.assign_op_to_device(op, ops_relation_dict, net_struct.DeviceType.CPU, CPU_end_point, \
      op.op_def.operator_latency.CPU_latency, op_execute_order_list)
  else:
    GPU_end_point = scheduler_utils.assign_op_to_device(op, ops_relation_dict, net_struct.DeviceType.GPU, GPU_end_point, \
      GPU_latency, op_execute_order_list)
  ops_queue.remove(op)
  to_be_sche_ops = []
  to_be_sche_ops.append(op)
  # Start greedy assign
  while(True):
    # Add child to ops_queue if all his parents has been executed
    child_list = []
    for op in to_be_sche_ops:
      child_list.extend(op.children)
    for child_name in child_list:
      if child_name in op_name_list and \
        scheduler_utils.is_parents_executed(ops_relation_dict[child_name], op_name_list, ops_relation_dict) and \
        ops_relation_dict[child_name] not in ops_queue and \
        not ops_relation_dict[child_name].executed:
        ops_queue.append(ops_relation_dict[child_name])
        logging.debug("Add %s" % (child_name))
    # All ops are assigned to devices, stop
    if(len(ops_queue) <= 0):
      break
    
    # Sort queue according to start point
    # ops_queue= sorted(ops_queue, key=key_sort_operator)
    to_be_sche_ops.clear()
    ops_queue = sorted(ops_queue, key=attrgetter("earlist_start_point", "name"))
    # ops_queue.sort(key=key_sort_operator)
    # Try to fetch two ops from queue
    for op_in_queue in ops_queue:
      # When all his father has been executed, then the op can start executing
      if scheduler_utils.is_parents_executed(op_in_queue, op_name_list, ops_relation_dict):
        to_be_sche_ops.append(op_in_queue)
        ops_queue.remove(op_in_queue)
        logging.debug("Fetch op %s " % op.name)
        if len(to_be_sche_ops) == 2:
          break
    print("ops_queue size %d" % (len(ops_queue)))
    CPU_latency_scale = 1.0
    # Only one op can be scheduled
    if len(to_be_sche_ops) == 1:
      op = to_be_sche_ops[0]
      # For ops that are not supported by GPU, set their device type as CPU(Fall back to CPU)
      if op.op_def.type in ops_not_support_by_GPU:
        CPU_end_point = scheduler_utils.assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, \
          CPU_end_point, op.op_def.operator_latency.CPU_latency, op_execute_order_list)
        continue
      # Assign the op to CPU or GPU
      # Find its father, get transpose latency
      CPU_latency, GPU_latency = scheduler_utils.get_ops_total_latency(op, ops_relation_dict)
      CPU_latency = CPU_latency * CPU_latency_scale
      # Get computation latency on devices
      logging.debug("op %s CPU and GPU endpoint: %f %f ,op endpoint %f" % ( op.name, CPU_end_point, GPU_end_point, op.earlist_start_point))
      # logging.debug("op %s CPU and GPU latency: %f %f to CPU and to GPU latency: %f %f" \
      #   % ( op.name, op.op_def.operator_latency.CPU_latency, op.op_def.operator_latency.GPU_latency, to_CPU_transpose_latency, to_GPU_transpose_latency))
      
      # Execute on CPU will be faster
      if max(CPU_end_point, op.earlist_start_point) + CPU_latency \
        <= max(GPU_end_point, op.earlist_start_point) + GPU_latency:
        CPU_end_point = max(CPU_end_point, op.earlist_start_point)
        CPU_end_point = scheduler_utils.assign_op_to_device(op, ops_relation_dict, net_struct.DeviceType.CPU, CPU_end_point, CPU_latency, op_execute_order_list)
      else:
        GPU_end_point = max(GPU_end_point, op.earlist_start_point)
        GPU_end_point = scheduler_utils.assign_op_to_device(op, ops_relation_dict, net_struct.DeviceType.GPU, GPU_end_point, GPU_latency, op_execute_order_list)
      # End of one Op
      
    else:
      CPU_latency0, GPU_latency0 = scheduler_utils.get_ops_total_latency(to_be_sche_ops[0], ops_relation_dict)
      CPU_latency0 = CPU_latency0 * CPU_latency_scale
      CPU_latency1, GPU_latency1 = scheduler_utils.get_ops_total_latency(to_be_sche_ops[1], ops_relation_dict)
      CPU_latency1 = CPU_latency1 * CPU_latency_scale
      to_be_CPU_latencies = [CPU_latency0, CPU_latency1]
      to_be_GPU_latencies = [GPU_latency0, GPU_latency1]
      # Choose one policy 
      # op0->CPU and op1->GPU
      policy_cpu_gpu = max(max(CPU_end_point, to_be_sche_ops[0].earlist_start_point)+to_be_CPU_latencies[0], \
        max(GPU_end_point, to_be_sche_ops[1].earlist_start_point)+to_be_GPU_latencies[1])
      policy_gpu_cpu = max(max(GPU_end_point, to_be_sche_ops[0].earlist_start_point)+to_be_GPU_latencies[0], \
        max(CPU_end_point, to_be_sche_ops[1].earlist_start_point)+to_be_CPU_latencies[1])
      # xpu_xpu policy is not accurate
      policy_cpu_cpu = max(max(to_be_sche_ops[0].earlist_start_point, to_be_sche_ops[1].earlist_start_point), \
        CPU_end_point) + sum(to_be_CPU_latencies)
      policy_gpu_gpu = max(max(to_be_sche_ops[0].earlist_start_point, to_be_sche_ops[1].earlist_start_point), \
        GPU_end_point) + sum(to_be_GPU_latencies)

      policy_latencies = [policy_cpu_gpu, policy_gpu_cpu, policy_cpu_cpu, policy_gpu_gpu]
      min_policy_latency = min(policy_latencies)
      if policy_cpu_gpu == min_policy_latency:
        CPU_end_point = max(CPU_end_point, to_be_sche_ops[0].earlist_start_point)
        CPU_end_point = assign_op_to_device(to_be_sche_ops[0], ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency0, op_execute_order_list)
        GPU_end_point = max(GPU_end_point, to_be_sche_ops[1].earlist_start_point)
        GPU_end_point = assign_op_to_device(to_be_sche_ops[1], ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency1, op_execute_order_list)
      elif policy_gpu_cpu == min_policy_latency:
        CPU_end_point = max(CPU_end_point, to_be_sche_ops[1].earlist_start_point)
        CPU_end_point = assign_op_to_device(to_be_sche_ops[1], ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency1, op_execute_order_list)
        GPU_end_point = max(GPU_end_point, to_be_sche_ops[0].earlist_start_point)
        GPU_end_point = assign_op_to_device(to_be_sche_ops[0], ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency0, op_execute_order_list)
      elif policy_cpu_cpu == min_policy_latency:
        CPU_end_point = max(CPU_end_point, to_be_sche_ops[0].earlist_start_point)
        CPU_end_point = assign_op_to_device(to_be_sche_ops[0], ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency0, op_execute_order_list)
        CPU_end_point = max(CPU_end_point, to_be_sche_ops[1].earlist_start_point)
        CPU_end_point = assign_op_to_device(to_be_sche_ops[1], ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency1, op_execute_order_list)
      else:
        GPU_end_point = max(GPU_end_point, to_be_sche_ops[0].earlist_start_point)
        GPU_end_point = assign_op_to_device(to_be_sche_ops[0], ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency0, op_execute_order_list)
        GPU_end_point = max(GPU_end_point, to_be_sche_ops[1].earlist_start_point)
        GPU_end_point = assign_op_to_device(to_be_sche_ops[1], ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency0, op_execute_order_list)

  # End of while
  # for op in netdef.op:
  # print(op.name + " " + str(op.op_def.device_type))
  
  # write_device_placement(os.path.join(folder_path, 'greedy-' + mobile + "-" + model_name + '-cpu-' + str(thread) +  '.txt') , netdef)
  print("CPU end point: %s ms." % CPU_end_point)
  print("GPU end point: %s ms." % GPU_end_point)
  print("Greedy Result %f" % (max(CPU_end_point, GPU_end_point)))
  # print(op_execute_order_list)
  lines = write_subgraph_device_placement_result(name_op_dict=name_op_dict, op_execution_order_list=op_execute_order_list)
  return lines



# Follow the mace
def greedy_device_placement(op_name_list, ops_relation_dict, folder_path, model_name, mobile, thread):
  # ops_not_support_by_GPU = set(['concat', 'SpatialSqueeze', 'Shape', 'Reshape', 'Softmax', 'Reshape_1'])
  ops_not_support_by_GPU = []
  # Record the CPU queue and GPU queue finish timestamp
  CPU_end_point = 0.0
  GPU_end_point = 0.0
  
  # Need to record the execute order
  op_execute_order_list = []
  
  input_op_names = find_input_nodes(op_name_list, ops_relation_dict)
  ops_queue = [ops_relation_dict[op_name] for op_name in input_op_names]
  print(input_op_names)
  assert(len(ops_queue) > 0)
  print("Start Greedy")
  op = ops_queue[0]
  # Execute the first op
  GPU_latency = op.op_def.operator_latency.GPU_latency + op.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW
  if op.op_def.operator_latency.CPU_latency < GPU_latency:
    CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, \
      op.op_def.operator_latency.CPU_latency, op_execute_order_list)
  else:
    GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, \
      GPU_latency, op_execute_order_list)
  ops_queue.remove(op)
  
  # Start greedy assign
  while(True):
    # Add child to ops_queue if all his parents has been executed
    for child_name in op.children:
      if child_name in op_name_list and \
        is_parents_executed(ops_relation_dict[child_name], op_name_list, ops_relation_dict) and \
        ops_relation_dict[child_name] not in ops_queue \
        and not ops_relation_dict[child_name].executed :
        ops_queue.append(ops_relation_dict[child_name])
        logging.debug("Add %s" % (child_name))
    # All ops are assigned to devices, stop
    if(len(ops_queue) <= 0):
      break
    # Sort queue according to start point
    # ops_queue= sorted(ops_queue, key=key_sort_operator)
    ops_queue= sorted(ops_queue, key=attrgetter("earlist_start_point", "name"))
    print("ops_queue size %d" % (len(ops_queue)))
    # ops_queue.sort(key=key_sort_operator)
    # Fetch an op from queue
    for op_in_queue in ops_queue:
      # When all his father has been executed, then the op can start executing
      if is_parents_executed(op_in_queue, op_name_list, ops_relation_dict):
        op = op_in_queue
        ops_queue.remove(op_in_queue)
        logging.debug("Fetch op %s " % op.name)
        break
    
    # For ops that are not supported by GPU, set their device type as CPU(Fall back to CPU)
    if op.op_def.type in ops_not_support_by_GPU:
      CPU_end_point = assign_op_to_device(op, ops_relation_dict, net_struct.DeviceType.CPU, \
        CPU_end_point, op.op_def.operator_latency.CPU_latency, op_execute_order_list)
      continue
    # Assign the op to CPU or GPU
    # Find its father, get transpose latency
    to_CPU_transpose_latency = 0.0
    to_GPU_transpose_latency = 0.0
    
    for op_parent_name in op.parents:
      op_parent = ops_relation_dict[op_parent_name]
      if op_parent.op_def.device_type == net_struct.DeviceType.CPU:
        to_GPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC)
      elif op_parent.op_def.device_type == net_struct.DeviceType.GPU:
        to_CPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW)
    
    # Get computation latency on devices
    logging.debug("op %s CPU and GPU endpoint: %f %f ,op endpoint %f" % ( op.name, CPU_end_point, GPU_end_point, op.earlist_start_point))
    logging.debug("op %s CPU and GPU latency: %f %f to CPU and to GPU latency: %f %f" \
      % ( op.name, op.op_def.operator_latency.CPU_latency, op.op_def.operator_latency.GPU_latency, to_CPU_transpose_latency, to_GPU_transpose_latency))
    # print(op.op_def.operator_latency)
    CPU_latency = op.op_def.operator_latency.CPU_latency + to_CPU_transpose_latency
    GPU_latency = op.op_def.operator_latency.GPU_latency + to_GPU_transpose_latency
    
    
    # Execute on CPU will be faster
    if max(CPU_end_point, op.earlist_start_point) + CPU_latency \
       <= max(GPU_end_point, op.earlist_start_point) + GPU_latency:
      CPU_end_point = max(CPU_end_point, op.earlist_start_point)
      CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency, op_execute_order_list)
    else:
      GPU_end_point = max(GPU_end_point, op.earlist_start_point)
      GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency, op_execute_order_list)
      
  # End of while
  # for op in netdef.op:
  # print(op.name + " " + str(op.op_def.device_type))
  
  # write_device_placement(os.path.join(folder_path, 'greedy-' + mobile + "-" + model_name + '-cpu-' + str(thread) +  '.txt') , netdef)
  print("CPU end point: %s ms." % CPU_end_point)
  print("GPU end point: %s ms." % GPU_end_point)
  print("Greedy Result %f" % (max(CPU_end_point, GPU_end_point)))
  # print(op_execute_order_list)
  lines = write_subgraph_device_placement_result(name_op_dict=name_op_dict, op_execution_order_list=op_execute_order_list)
  return lines
  

if __name__ == "__main__":
  logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
  model, mobile, thread, CPU_little_thread_index = utils.parse_model_mobile()
  # model, mobile, thread = "inception-v3", "redmi", 2
  model_dir = os.path.join(utils.get_project_path(), "mnn/models/", model)
  folder_path = os.path.join(model_dir, mobile)
  op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread, CPU_little_thread_index=CPU_little_thread_index)

  # greedy_device_placement(op_name_list, name_op_dict, folder_path, model, mobile, thread)
  unsupported_op_names = []
  if model.find("nasnet") >= 0:
    unsupported_op_names = ["final_layer/Relu", "final_layer/Mean/reduction_indices", \
          "final_layer/Relu___tr4final_layer/Mean", "final_layer/Mean", \
          "final_layer/FC/weights", "final_layer/FC/MatMul", \
          "final_layer/FC/biases", "final_layer/FC/BiasAdd", "final_layer/predictions"]
  lines = []
  graph_mode = 1
  # 
  if graph_mode == 0:
    subgraph_name_list, name_op_dict = subgraph.build_multi_subgraphs(model, mobile, thread)
    lines = greedy_device_placement_v3(subgraph_name_list, name_op_dict, folder_path, model, mobile, thread)
    # edges = []
    # for graph in subgraph_name_list:
    #   if graph.find("cell_7") == 0:
    #     sub_graph = name_op_dict[graph].op_name_list
    #     for op_name in sub_graph:
    #       op = name_op_dict[op_name]
    #       if len(op.children) > 0:
    #         for child in op.children:
    #           edges.append((op_name, child))
    # for (op_name, child) in edges:
    #   print('\"{}\"->\"{}";'.format(op_name, child))
    # draw_dag(edges)
  elif graph_mode == 1:
    lines = greedy_device_placement_v3(op_name_list, name_op_dict, folder_path, model, mobile, thread)
  elif graph_mode == 2:
    input_node_name = None
    if model == "inception-v3":
        input_node_name = "InceptionV3/InceptionV3/Conv2d_1a_3x3/Conv2D"
    elif model == "inception-v4":
        input_node_name = "InceptionV4/InceptionV4/Conv2d_1a_3x3/Conv2D"
    elif model == "pnasnet-mobile" or model == "pnasnet-large" \
      or model == "nasnet-mobile" or model == "nasnet-large":
        input_node_name = 'conv0/Conv2D'
    else:
      print("Model {} not support, exit now".format(model))
      exit(0)
    name_weight_dict, result = find_critical_node.get_node_weight(model, input_node_name)
    subgraph_list = graph_partition.auto_build_multi_subgrap_with_weight(op_name_list, name_op_dict, name_weight_dict, 648000)
    subgraph.build_subgraph_relationship([subgraph.name for subgraph in subgraph_list], name_op_dict)
    lines = greedy_device_placement_v3([subgraph.name for subgraph in subgraph_list], name_op_dict, folder_path, model, mobile, thread)
    # Deal with ops that are not in the module prefix
  lines, untreated_op_latency = subgraph.insert_untreated_ops(lines, op_name_list, name_op_dict, \
    unsupported_op_names=unsupported_op_names)
  # Write results
  device_map_file_path = os.path.join(model_dir, mobile, "greedy-placement-{}-cpu-{}.txt".format(model, thread))
  # replace GPU device to little device
  new_lines = []
  if CPU_little_thread_index != None:
      device_map_file_path = os.path.join(model_dir, mobile, \
          "mDeviceMap-{}-cpu-big-{}-little-{}.txt".format(model, thread, CPU_little_thread_index))
      # Little core device type is 2
      for line in lines:
          line = line.replace(' 3', ' 2')
          new_lines.append(line)
      lines = new_lines
  utils.write_lines(device_map_file_path, lines)

  optimized_lines = optimizer.depth_first_reorder(lines, op_name_list, name_op_dict)
  utils.write_lines(device_map_file_path+".opt", optimized_lines)
  
  rc = os.system("adb push {} /data/local/tmp/".format(device_map_file_path))
  if rc == 0:
    print("Push greedy device file to device")
  else:
    print("There is no device")
  # Special for ACL
  # if model.find("acl") >= 0:
  #   model_name = model.split('-')[-1]
  #   run_cmd = 'adb shell "cd /data/local/tmp/ && ./acl-run.sh {} CL parallel {} {}"'.format(model_name, thread, "greedy-placement-{}-cpu-{}.txt".format(model, thread))
  #   os.system(run_cmd)
  exit(0)
  # sh_cmd = 'python analyze/compare_latency.py {} {} {}'.format(model, mobile, thread)
  # print(sh_cmd)
  # os.system(sh_cmd)
  