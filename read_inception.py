#! /usr/bin/python

import mace_pb2
import sys


def read_netdef(protobuf_file_path):
  netdef = mace_pb2.NetDef()
  try:
    f = open(protobuf_file_path, "rb")
    net_str = f.read()
    netdef.ParseFromString(net_str)
    f.close()
  except IOError:
    print(protobuf_file_path + ": Could not open file.  Creating a new one.")
  print("--------------------------###############------------------")
  #print(netdef.op)
  for op in netdef.op:
    if(op.type == "Conv2D") and op.device_type == 2:
      print(op.name+" "+str(op.device_type))
  op_def = netdef.op[0]
  #print(op_def)
  return netdef



def read_op_latency(file_path):
  try:
    f = open(file_path, 'r')
    lines = f.readlines()
    op_latency_dict = dict()
    for line in lines:
      com = line.split("\t")
      if(len(com) < 5):
        continue
      print(com)
      op_latency_dict[com[0].strip()] = (com[1].strip(), com[2].strip(), com[3].strip(), com[4].strip())

  except IOError:
    print("Open %s failed." % file_path)
  
  return op_latency_dict



def assign_latency_to_netdef(op_latency_dict, netdef):
  for i in range(len(netdef.op)):
    name = netdef.op[i].name
    if op_latency_dict.has_key(name):
      CPU_latency, GPU_latency, nhwc_to_nchw, nchw_to_nhwc = op_latency_dict[name]
      netdef.op[i].opratorLatency.CPU_latency = float(CPU_latency)
      netdef.op[i].opratorLatency.GPU_latency = float(GPU_latency)
      netdef.op[i].opratorLatency.Transpose_latency_NCHW_to_NHWC = float(nchw_to_nhwc) / 1000
      netdef.op[i].opratorLatency.Transpose_latency_NHWC_to_NCHW = float(nhwc_to_nchw) / 1000
    else:
      print("Op name %s can not find latency" % name)
  print(netdef.op)
  return netdef
  


def write_bench_netdef(file_path, netdef):
  f = open(file_path, 'w')
  f.write(netdef.SerializeToString())
  f.flush()
  f.close()



if __name__ == "__main__":
  # op_latency_dict = read_op_latency("ops_latency")
  netdef = read_netdef("s_inception_v3.pb")
  # print(op_latency_dict)
  #netdef = assign_latency_to_netdef(op_latency_dict, netdef)
  #write_bench_netdef("inception_v3_latency.pb", netdef)
