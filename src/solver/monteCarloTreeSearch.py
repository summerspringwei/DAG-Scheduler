

from greedy_device_placement import *

op_name_list = []
name_op_dict = {}

DEPTH_THREASHOLD = 5

class TreeNode:

    def __init__(self, parent):
        super().__init__()
        self._parent = parent
        self._children = []
        self._op_name = ""
        self._visited_count = 0
        self._latency = 0.0
        self.CPU_endpoint = 0.0
        self.GPU_endpoint = 0.0
    
    def expand(self, to_be_add_op_names, depth):
        # Search all placement polices if under depth
        if len(to_be_add_op_names) == 0:
            self._latency = max(self.CPU_endpoint, self.GPU_endpoint)
            return
        if depth <= DEPTH_THREASHOLD:
            for op_name in to_be_add_op_names:
                op = name_op_dict[op_name]
                if is_parent_executed(op, op_name_list, name_op_dict):
                    # Assign to CPU
                    node = TreeNode(self)
                    node.CPU_endpoint = max(op.op_def.op.earlist_start_point, self.CPU_endpoint) + op.op_def.operator_latency.CPU_latency
                    node.GPU_endpoint = self.GPU_endpoint
                    self._children.append(node)
                    to_be_add = list(to_be_add_op_names)
                    to_be_add.remove(op_name)
                    node.expand(to_be_add, depth+1)
                    # Assign to GPU
                    node = TreeNode(self)
                    node.GPU_endpoint = max(op.op_def.op.earlist_start_point, self.GPU_endpoint) + op.op_def.operator_latency.GPU_latency
                    node.CPU_endpoint = self.CPU_endpoint
                    self._children.append(node)
                    node.expand(to_be_add, depth+1)
        else:
            # Use greedy algorithm to get a placement latency
            pass

    def is_leaf_node(self):
        return len(self._children) == 0
    
    # Update min latency for parent
    def update_recursive(self):
        pass

