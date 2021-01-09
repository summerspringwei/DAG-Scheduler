# enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3, HTA = 4, APU = 5 };
# Follow the MNN
class DeviceType:
    CPU = 0
    GPU = 3


class OperatorLatency:
    def __init__(self):
        self.CPU_latency = 0
        self.GPU_latency = 0
        self.Transpose_latency_NHWC_to_NCHW = 0  # Transformation latency GPU to CPU
        self.Transpose_latency_NCHW_to_NHWC = 0  # Transformation latency CPU to GPU
        self.input_data_trans_latency = {} # dict of (tensor addr: data trans latency)
    
    def get_avg_data_trans_latency(self):
        """Return the average data transformation latency of all input tensors on CPU and GPU
        """
        sum = 0
        for _, data_trans_latency in self.input_data_trans_latency.items():
            sum += (data_trans_latency[0] + data_trans_latency[1])
        return sum / len(self.input_data_trans_latency)
    
    def __str__(self):
        return "Operator latency: %f %f %s" % (self.CPU_latency, self.GPU_latency, \
            self.input_data_trans_latency)


class OperatorDef:
    def __init__(self):
        self.type = ""
        self.device_type = DeviceType.CPU
        self.operator_latency = OperatorLatency()


class Operator:
    def __init__(self, name):
        self.name = name
        self.parents = set()
        self.children = set()
        self.op_def = OperatorDef()
        self.executed = False
        self.earlist_start_point = 0.0
        self.bottom_level = 0.0  # One of task priority introduced in paper
        #A scalable clustering-based task scheduler for homogeneous processors using DAG partitioning
        self.input_tensors = []  # Tuple list '(tensor_addr, tensor_shape)'
        self.output_tensors = []
        self.input_nodes = []
        self.output_nodes = []
    
    def __str__(self):
        return self.name + " " + self.op_def.type + " " \
                + str(self.op_def.operator_latency.CPU_latency) + " " \
                + str(self.op_def.operator_latency.GPU_latency) + " " \
                + str(self.input_tensors) + " " \
                + str(self.output_tensors) + " " \
                + str(self.parents) + " " + str(self.children)


def build_op_relationship(op_list):
    for op in op_list:
        for op_parent in op_list:
            if op == op_parent:
                continue
            for addr_parent, _ in op_parent.output_tensors:
                if addr_parent in [addr for addr, _ in op.input_tensors]:
                    op.parents.add(op_parent.name)
                    op_parent.children.add(op.name)
                    break

    return op_list


class NetDef:
    def __init__(self):
        self.op = []
