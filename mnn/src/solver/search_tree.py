
import numpy as np
import pysnooper
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger()

from profile import net_struct
from solver import scheduler_utils
from utils import utils


class Node:
    """Tree node for building the search tree

    """
    def __init__(self, op, device, parent):
        self.op = op
        self.device = device
        self.end_point = [0, 0] # The timestamp when the self.op execute finished on self.device
        self.children_node = []
        self.parent = parent
        
        
    # @pysnooper.snoop()
    def update(self, name_op_dict):
        """Set devices' end point if put self.op on self.device
        """
        start_point = max(self.op.earlist_start_point, self.parent.end_point[self.device])
        to_CPU_transpose_latency = 0.0
        to_GPU_transpose_latency = 0.0
        CPU_latency, GPU_latency = scheduler_utils.get_ops_total_latency(self.op, name_op_dict)
        
        # for op_parent_name in self.op.parents:
        #     utils.get_logger().info("{} {}; {} {} {}".format(\
        #         self.op.name,self.device, op_parent_name, name_op_dict[op_parent_name].executed, name_op_dict[op_parent_name].op_def.device_type))
        #     op_parent = name_op_dict[op_parent_name]
        #     if op_parent.op_def.device_type == net_struct.DeviceType.CPU:
        #         # to_GPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC)
        #         to_GPU_transpose_latency += op_parent.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC
        #     elif op_parent.op_def.device_type == net_struct.DeviceType.GPU:
        #         # to_CPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW)
        #         to_CPU_transpose_latency += op_parent.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW
        
        # data_trans_latency = [to_CPU_transpose_latency, to_GPU_transpose_latency][self.device]
        
        # execution_latency = scheduler_utils.get_ops_total_latency(self.op, name_op_dict)[self.device]
        # execution_latency = [self.op.op_def.operator_latency.CPU_latency, self.op.op_def.operator_latency.GPU_latency][self.device]
        execution_latency = [CPU_latency, GPU_latency][self.device]
        self.end_point = list(self.parent.end_point)
        self.end_point[self.device] = start_point + execution_latency
    
    def update_not_ready(self, name_op_dict):
        pass
        

def build_search_tree(op_name_list, name_op_dict, end_point):
    root = Node(None, None, None)
    root.end_point = end_point
    leaf_node_list = root.children_node
    for op_name in op_name_list:
        # init
        if len(leaf_node_list) == 0:
            for device in [0, 1]:
                node = Node(name_op_dict[op_name], device, root)
                node.update(name_op_dict)
                leaf_node_list.append(node)
                # print("Add node {}".format(node.op.name))
        else:
            tmp_leaf_node_list = []
            for leaf_node in leaf_node_list:
                for device in [0, 1]:
                    node = Node(name_op_dict[op_name], device, leaf_node)
                    node.update(name_op_dict)
                    tmp_leaf_node_list.append(node)
                    # print("Add node {}".format(node.op.name))
            leaf_node_list.clear()
            leaf_node_list = tmp_leaf_node_list
    return root, leaf_node_list


def get_optimal_device_placement(root, leaf_node_list, name_op_dict):
    min_end_point = 1e8
    optimal_leaf_node = None
    # traverse all the leaf node find the minimal endpoint
    for leaf_node in leaf_node_list:
        if max(leaf_node.end_point) < min_end_point:
            min_end_point = max(leaf_node.end_point)
            optimal_leaf_node = leaf_node
    
    logger.info("optimal endpoint {}".format(optimal_leaf_node.end_point))
    device_placement = []
    node_ptr = optimal_leaf_node
    while node_ptr.parent != None:
        device_type = [net_struct.DeviceType.CPU, net_struct.DeviceType.GPU][node_ptr.device]
        device_end_point = node_ptr.end_point[node_ptr.device]
        device_placement.append((node_ptr.op.name, device_type, device_end_point))
        # Set op's device type
        node_ptr.op.op_def.device_type = device_type
        # Set op's state as executed
        node_ptr.op.executed = True
        # Update child start point
        for child_name in node_ptr.op.children:
            child = name_op_dict[child_name]
            child.earlist_start_point = max(child.earlist_start_point, device_end_point)
            
        node_ptr = node_ptr.parent
        
    return optimal_leaf_node.end_point, device_placement



# test
def test_search_tree():
    op_name_list = ["a", "b", "c", "d"]
    op_latency_list = [[1,2], [1,2],[1,3], [2, 3]]
    op_data_trans_latency_list = [[0, 0], [0, 0], [0, 0], [0, 0]]
    # op_name_list = ["a"]
    # op_latency_list = [[1,2]]
    # op_data_trans_latency_list = [[1, 0.5]]
    name_op_dict = {}
    for i in range(len(op_name_list)):
        op = Operator(op_name_list[i])
        op_def = OperatorDef()
        op_latency = operator_latency()
        [op_latency.CPU_latency, op_latency.GPU_latency] = op_latency_list[i]
        [op_latency.Transpose_latency_NCHW_to_NHWC, op_latency.Transpose_latency_NCHW_to_NHWC] = op_data_trans_latency_list[i]
        op_def.operator_latency = op_latency
        op.op_def  = op_def
        name_op_dict[op.name] = op
    root, leaf_node_list = build_search_tree(op_name_list, name_op_dict, [0, 0])
    optimal_leaf_node, device_placement = get_optimal_device_placement(root, leaf_node_list, name_op_dict)
    print(optimal_leaf_node)
    print(device_placement)


if __name__ == "__main__":
    test_search_tree()
