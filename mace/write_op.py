#! /usr/bin/python

import mace_pb2

op_def = mace_pb2.OperatorDef()
op_def.name = "fffff"
op_def.type = "Conv2d"
op_def.device_type = 1
print(op_def)
print("Add field done")

op_def_str = op_def.SerializeToString()
f = open("test_op.pbtxt", 'w')
f.write(op_def_str)
f.flush()
f.close()
print("Serial to String done.")

f2 = open("test_op.pbtxt", 'r')
op_def2 = mace_pb2.OperatorDef()
op_def2.ParseFromString(f2.read())
print(op_def2)
