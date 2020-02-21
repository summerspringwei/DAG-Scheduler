from net_struct import *


# Read Tensor transformation latency
def read_data_trans(file_path):
    f = open(file_path, 'r')
    first_line = True
    data_trans_dict = {}

    for line in f.readlines():
        com = line.strip().split("\t")
        # print("%s*%s*%s" % (com[0],com[1],com[2]))
        if(len(com) != 3):
            print(com)
            continue
        if first_line:
            first_line = False
            continue
        com[0] = str(com[0])[1:len(com[0])-1]
        data_trans_dict[com[0]] = [float(com[1]), float(com[2])]
    # print("dict--")
    # print(data_trans_dict)
    return data_trans_dict


# Read Operator latency, for now we read 4 thread
def read_latency(file_path):
    f = open(file_path, 'r')
    operator_latency_dict = {}
    first_line = True
    op_name_list = []
    for line in f.readlines():
        if first_line:
            first_line = False
            continue
        
        com = line.strip().split("\t")
        if len(com) < 3:
            continue
        op_latency = OperatorLatency()
        op_latency.CPU_latency = float(com[2].strip())/1000
        op_latency.GPU_latency = float(com[4].strip())/1000
        # Set GPU concat to a big value
        if com[0].strip().split('/')[-1] == 'concat':
            # print("concat big value")
            op_latency.GPU_latency = 500
        op_name = com[0].strip()
        op_name_list.append(op_name)
        operator_latency_dict[op_name] = op_latency
    # for k, v in operator_latency_dict.items():
    #     print("%s %s" % (k, v))
    return op_name_list, operator_latency_dict


def read_inception_info(file_path):
    f = open(file_path,'r')
    data_trans_dict = read_data_trans("/mnt/d/home/Projects/DAG-scheduler/mnn/redmi_data_trans.txt")
    op_name_list, latency_dict = read_latency("/mnt/d/home/Projects/DAG-scheduler/mnn/redmi_inception-v3-layerwise-latency.txt")
    # for k, v in latency_dict.items():
    #     print("%s %s" % (k, v))
    # exit(0)
    # print(data_trans_dict)
    op_dict = {}
    net_def = NetDef()
    for line in f.readlines():
        com = line.strip().split(" ")
        if(len(com) < 4):
            continue
        op_name = com[0].strip()
        op_type = op_name.split('/')[-1]
        op = Operator(op_name)
        op_def = OperatorDef()
        op_def.type = op_type
        op_latency = latency_dict[op_name]
        
        inputs = com[2].strip().split(";")
        # Get OperatorLatency transformation latency
        for tensor in inputs:
            if(len(tensor) >= 8):
                op_latency.Transpose_latency_NCHW_to_NHWC += data_trans_dict[tensor][0]
                op_latency.Transpose_latency_NHWC_to_NCHW += data_trans_dict[tensor][1]
        # Get op's children and parents
        parents = com[3].strip().split(";")
        for p in parents:
            if len(p) > 0:
                op.parents.add(op_name_list[int(p)])
        if len(com) == 5:
            children = com[4].strip().split(";")
            for c in children:
                if len(c) > 0:
                    op.children.add(op_name_list[int(c)])
        
        op_def.operatorLatency = op_latency
        op.op_def = op_def
        op_dict[op.name] = op
        net_def.op.append(op)
    return op_name_list, op_dict, net_def



if __name__ == "__main__":
    
    op_name_list, op_dict, net_def = read_inception_info("/mnt/d/home/Projects/DAG-scheduler/mnn/inception-v3-info.txt")
    for op_name in op_name_list:
        print(op_dict[op_name])
    