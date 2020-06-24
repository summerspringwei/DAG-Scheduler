#! /usr/bin/python
import logging
import os
import queue
import optimizer
from operator import itemgetter, attrgetter

from read_profile_data import *
from utils import *
from read_net_structure import *
from draw_dag import *


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
def assign_op_to_device(op, ops_relation_dict, device_type, device_end_point, latency, op_execute_order_list):
  op.executed = True
  op.op_def.device_type = device_type
  op_execute_order_list.append((op.name, device_type, device_end_point + latency))
  update_children_start_point(op, ops_relation_dict, device_end_point, latency)
  return device_end_point + latency


def is_parents_executed(op, op_name_list, ops_relation_dict):
  ready = True # When all his father has been executed, then the op can start executing
  for op_parent_name in op.parents:
    if not op_parent_name in ops_relation_dict.keys():
      raise KeyError()
    if op_parent_name not in op_name_list:
      continue
    op_parent = ops_relation_dict[op_parent_name]
    if op_parent.executed == False:
      ready = False
  return ready


def write_device_placement(filename, net_def):
  f = open(filename, 'w')
  for op in net_def.op:
    f.write("%s %d\n" % (op.name, op.op_def.device_type))
  f.flush()
  f.close()
  print("Write device placement done.")

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
  GPU_latency = op.op_def.operatorLatency.GPU_latency + op.op_def.operatorLatency.Transpose_latency_NHWC_to_NCHW
  if op.op_def.operatorLatency.CPU_latency < GPU_latency:
    CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, \
      op.op_def.operatorLatency.CPU_latency, op_execute_order_list)
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
      CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, \
        CPU_end_point, op.op_def.operatorLatency.CPU_latency, op_execute_order_list)
      continue
    # Assign the op to CPU or GPU
    # Find its father, get transpose latency
    to_CPU_transpose_latency = 0.0
    to_GPU_transpose_latency = 0.0
    
    for op_parent_name in op.parents:
      op_parent = ops_relation_dict[op_parent_name]
      if op_parent.op_def.device_type == DeviceType.CPU:
        to_GPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operatorLatency.Transpose_latency_NCHW_to_NHWC)
      elif op_parent.op_def.device_type == DeviceType.GPU:
        to_CPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operatorLatency.Transpose_latency_NHWC_to_NCHW)
    
    # Get computation latency on devices
    logging.debug("op %s CPU and GPU endpoint: %f %f ,op endpoint %f" % ( op.name, CPU_end_point, GPU_end_point, op.earlist_start_point))
    logging.debug("op %s CPU and GPU latency: %f %f to CPU and to GPU latency: %f %f" \
      % ( op.name, op.op_def.operatorLatency.CPU_latency, op.op_def.operatorLatency.GPU_latency, to_CPU_transpose_latency, to_GPU_transpose_latency))
    # print(op.op_def.operatorLatency)
    CPU_latency = op.op_def.operatorLatency.CPU_latency + to_CPU_transpose_latency
    GPU_latency = op.op_def.operatorLatency.GPU_latency + to_GPU_transpose_latency
    
    # Deal with concat
    # if op.name.find('concat') >= 0:
    #   GPU_end_point = max(GPU_end_point, op.earlist_start_point)
    #   GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency, op_execute_order_list)
    #   continue
    # TODO(xcw)add to_GPU_transpose_latency to CPU_end_point
    # Op can be executed at the very first time, but CPU and GPU are busy
    # if CPU_end_point >= op.earlist_start_point and GPU_end_point >= op.earlist_start_point:
    #   if CPU_end_point + CPU_latency < GPU_end_point + GPU_latency: # CPU can finish this op earlier(Greedy here)
    #     CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency, op_execute_order_list)
    #     # We need to add the to_CPU_transpose_latency to GPU_queue as the data transformation is done by GPU
    #     GPU_end_point += to_CPU_transpose_latency
    #   else: # GPU is better
    #     GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency, op_execute_order_list)
    
    # # One device is ready but the other one is busy(or just finish work)
    # elif (op.earlist_start_point >= CPU_end_point and op.earlist_start_point <= GPU_end_point):
    #   if op.earlist_start_point + CPU_latency < GPU_end_point + GPU_latency:# Note, CPU_end_point changed
    #     CPU_end_point = op.earlist_start_point
    #     CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency, op_execute_order_list)
    #     GPU_end_point += to_CPU_transpose_latency
    #   else:
    #     GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency, op_execute_order_list)
    # elif(op.earlist_start_point <= CPU_end_point and op.earlist_start_point >= GPU_end_point):
    #   if op.earlist_start_point + GPU_latency < CPU_end_point + CPU_latency:
    #     GPU_end_point = op.earlist_start_point
    #     GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency, op_execute_order_list)
    #   else:
    #     CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency, op_execute_order_list)
    #     GPU_end_point += to_CPU_transpose_latency
    # else:
    #   if CPU_latency < GPU_latency:
    #     CPU_end_point = op.earlist_start_point
    #     CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency, op_execute_order_list)
    #     GPU_end_point += to_CPU_transpose_latency
    #   else:
    #     GPU_end_point = op.earlist_start_point
    #     GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency, op_execute_order_list)
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
  # model, mobile, thread = parse_model_mobile()
  model, mobile, thread = "pnasnet-large", "redmi", 2
  model_dir = os.path.join("/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/", model)
  folder_path = os.path.join(model_dir, mobile)
  op_name_list, name_op_dict, net_def = gather_model_profile(
        os.path.join(model_dir, model + "-info.txt"),
        os.path.join(model_dir, mobile, model+'-'+mobile+'-data-trans.csv'),
        os.path.join(model_dir, mobile, mobile+"-"+model+"-layerwise-latency.csv"),
        thread)
  # greedy_device_placement(op_name_list, name_op_dict, folder_path, model, mobile, thread)
  lines = []
  if True:
    subgraph_name_list, name_op_dict = build_multi_subgraphs(model, mobile, thread)
    lines = greedy_device_placement(subgraph_name_list, name_op_dict, folder_path, model, mobile, thread)
    # edges = []
    # for graph in subgraph_name_list:
    #   if graph.find("cell_7") == 0:
    #     sub_graph = name_op_dict[graph].op_name_list
    #     for op_name in sub_graph:
    #       op = name_op_dict[op_name]
    #       if len(op.children) > 0:
    #         for child in op.children:
    #           edges.append((op_name, child))
    # draw_dag(edges)
  else:
    lines = greedy_device_placement(op_name_list, name_op_dict, folder_path, model, mobile, thread)
  unsupported_op_names = []
  if model.find("nasnet") >= 0:
    unsupported_op_names = ["final_layer/Relu", "final_layer/Mean/reduction_indices", \
          "final_layer/Relu___tr4final_layer/Mean", "final_layer/Mean", \
          "final_layer/FC/weights", "final_layer/FC/MatMul", \
          "final_layer/FC/biases", "final_layer/FC/BiasAdd", "final_layer/predictions"]
    # Deal with ops that are not in the module prefix
  lines, untreated_op_latency = insert_untreated_ops(lines, op_name_list, name_op_dict, \
    unsupported_op_names=unsupported_op_names)
  # Write results
  device_map_file_path = os.path.join(model_dir, mobile, "greedy-placement-{}-cpu-{}.txt".format(model, thread))
  write_lines(device_map_file_path, lines)

  optimized_lines = optimizer.depth_first_reorder(lines, op_name_list, name_op_dict)
  write_lines(device_map_file_path+".opt", optimized_lines)
  
  rc = os.system("adb push {} /data/local/tmp/".format(device_map_file_path))
  if rc == 0:
    print("Push greedy device file to device")
  rc = os.system("adb push {} /data/local/tmp/".format(device_map_file_path+".opt"))
  if rc == 0:
    print("Push greedy device opt file to device")

