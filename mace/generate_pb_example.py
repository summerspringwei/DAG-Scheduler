
import mace_pb2
import greedy_device_placement

netdef = mace_pb2.NetDef()

def build_op_def(name, type, input, output, device_type, operator_latency):
  op_def = mace_pb2.OperatorDef()
  op_def.name = name
  op_def.input.extend(input)
  op_def.output.extend(output)
  op_def.device_type = device_type
  op_def.operator_latency.CopyFrom(operator_latency)
  return op_def


def build_operator_latency(CPU_latency, GPU_latency, Transpose_latency_NCHW_to_NHWC, Transpose_latency_NHWC_to_NCHW):
  operator_latency = mace_pb2.OpratorLatency()
  operator_latency.CPU_latency = CPU_latency
  operator_latency.GPU_latency = GPU_latency
  operator_latency.Transpose_latency_NCHW_to_NHWC = Transpose_latency_NCHW_to_NHWC
  operator_latency.Transpose_latency_NHWC_to_NCHW = Transpose_latency_NHWC_to_NCHW
  return operator_latency


def build_DAG_example(file_name):
  netdef = mace_pb2.NetDef()
  opL1 = build_operator_latency(10, 5, 1, 1)
  opL2 = build_operator_latency(5, 10, 2, 2)
  op1 = build_op_def("op1", "Conv2d", ["input"], ["output1"], greedy_device_placement.DeviceType.CPU, opL1)
  op2 = build_op_def("op2", "Conv2d", ["output1"], ["output2"], greedy_device_placement.DeviceType.CPU, opL2)

  opL3 = build_operator_latency(10, 5, 1, 1)
  opL4 = build_operator_latency(5, 5, 1.5, 1.5)
  opL5 = build_operator_latency(5, 5, 1, 1)
  op3 = build_op_def("op3", "Conv2d", ["output2"], ["output3"], greedy_device_placement.DeviceType.CPU, opL3)
  op4 = build_op_def("op4", "Conv2d", ["output2"], ["output4"], greedy_device_placement.DeviceType.CPU, opL4)
  op5 = build_op_def("op5", "Conv2d", ["output2"], ["output5"], greedy_device_placement.DeviceType.CPU, opL5)

  opL6 = build_operator_latency(5, 15, 0.5, 0.5)
  opL7 = build_operator_latency(15, 5, 0.5, 0.5)
  opL8 = build_operator_latency(1, 1, 0.5, 0.5)
  op6 = build_op_def("op6", "Conv2d", ["output3"], ["output6"], greedy_device_placement.DeviceType.CPU, opL6)
  op7 = build_op_def("op7", "Conv2d", ["output4"], ["output7"], greedy_device_placement.DeviceType.CPU, opL7)
  op8 = build_op_def("op8", "Conv2d", ["output5", "output6", "output7"], ["output8"], greedy_device_placement.DeviceType.CPU, opL8)

  netdef.op.extend([op1, op2, op3, op4, op5, op6, op7, op8])
  f = open(file_name, 'w')
  f.write(netdef.SerializeToString())
  f.flush()
  f.close()
  print("Serial to String done.")


if __name__ == "__main__":
  build_DAG_example("my_dag.pb")
