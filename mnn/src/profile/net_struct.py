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
        self.input_data_trans_latency = {}
    
    def __str__(self):
        return "Operator latency: %f %f %s" % (self.CPU_latency, self.GPU_latency, \
            self.input_data_trans_latency)


class OperatorDef:
    def __init__(self):
        self.type = ""
        self.device_type = DeviceType.CPU
        self.operatorLatency = OperatorLatency()


class Operator:
    def __init__(self, name):
        self.name = name
        self.parents = set()
        self.children = set()
        self.op_def = OperatorDef()
        self.executed = False
        self.device_type = DeviceType.CPU  # 0 for CPU, 1 for GPU
        self.earlist_start_point = 0.0
        self.input_tensors = [] # Tuple list '(tensor_addr, tensor_shape)'
        self.output_tensors = []
        self.input_nodes = []
        self.output_nodes = []

    def __str__(self):
        return self.name + " " + self.op_def.type + " " \
                + str(self.op_def.operatorLatency.CPU_latency) + " " \
                + str(self.op_def.operatorLatency.GPU_latency) + " " \
                + str(self.input_tensors) + " " \
                + str(self.output_tensors) + " " \
                + str(self.parents) + " " + str(self.children)


class NetDef:
    def __init__(self):
        self.op = []
