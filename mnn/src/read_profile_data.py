from net_struct import *

# Set the cpu<->gpu transformation overhead
TRANSFORM_OVERHEAD = 3
# Set the Operator latency scale factor, deal with sum of ops is less than the end-to-end latency
OP_LATENCY_SCALE = 3
# CPU thread latency index
CPU_thread_index = 2

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


# Read Operator latency, for now we read 4 thread for CPU
# Note special case for 'concat'
def read_latency(file_path):
    f = open(file_path, 'r')
    operator_latency_dict = {}
    op_name_list = []
    for line in f.readlines():
        com = line.strip().split(" ")
        if len(com) < 4:
            continue
        op_latency = OperatorLatency()
        op_latency.CPU_latency = float(com[CPU_thread_index].strip()) / 1000 * OP_LATENCY_SCALE
        op_latency.GPU_latency = float(com[4].strip()) / 1000
        # Set GPU concat to a big value
        # if com[0].strip().split('/')[-1] == 'concat':
            # print("concat big value")
            # op_latency.GPU_latency = 500
        op_name = com[0].strip()
        op_name_list.append(op_name)
        operator_latency_dict[op_name] = op_latency
    # for k, v in operator_latency_dict.items():
    #     print("%s %s" % (k, v))
    return op_name_list, operator_latency_dict


def read_net_info(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    name_list = []
    name_op_dict = {}
    raw_info = []
    # Units name, input tensor shapes, output tensor shapes, parent idx, children idx
    for line in lines:
        com = line.split(" ")
        if len(com) < 5:
            print(com)
            print("Error")
            continue
        new_com = [c.strip() for c in com]
        raw_info.append(new_com)
    for com in raw_info:
        name = com[0]
        op = Operator(name)
        if com[1] != 'none':
            input_tensors = com[1].split(';')
            for it in input_tensors:
                if it.strip() != '':
                    op.input_tensors.append(it)
        if com[2] != 'none':
            output_tensors = com[2].split(';')
            for ot in output_tensors:
                if ot.strip() != '':
                    op.output_tensors.append(ot)
        if com[3] != 'none':
            parents = com[3].split(';')
            for p in parents:
                if p.strip() != '':
                    parents_name = raw_info[int(p.strip())][0]
                    op.parents.add(parents_name)
        if com[4] != 'none':
            children = com[4].split(';')
            for c in children:
                if c.strip() != '':
                    child_name = raw_info[int(c.strip())][0]
                    op.children.add(child_name)
        name_list.append(name)
        name_op_dict[name] = op
        # for op_name in name_list:
        #     print(name_op_dict[op_name].input_tensors)
    return name_list, name_op_dict


# We need three file to read the profiling info
# The 'raw_info_file_path' file describes the model structure
def gather_model_profile(raw_info_file_path, data_trans_file_path, inference_latency_file_path):
    data_trans_dict = read_data_trans(data_trans_file_path)
    op_name_list, latency_dict = read_latency(inference_latency_file_path)
    op_name_list, name_op_dict = read_net_info(raw_info_file_path)
    net_def = NetDef()
    # Gather three file into name_op_dict
    for op_name in op_name_list:
        op = name_op_dict[op_name]
        op_latency = latency_dict[op_name]
        op_type = op_name.split('/')[-1]
        op_def = OperatorDef()
        op_def.type = op_type
        # Set OperatorLatency transformation latency
        for tensor in op.input_tensors:
            if(len(tensor) >= 1):
                if tensor in data_trans_dict.keys():
                    op_latency.Transpose_latency_NCHW_to_NHWC += data_trans_dict[tensor][0]
                    op_latency.Transpose_latency_NHWC_to_NCHW += data_trans_dict[tensor][1]
                else:
                    op_latency.Transpose_latency_NCHW_to_NHWC = TRANSFORM_OVERHEAD
                    op_latency.Transpose_latency_NHWC_to_NCHW = TRANSFORM_OVERHEAD
        op_def.operatorLatency = op_latency
        op.op_def = op_def
        name_op_dict[op_name] = op
        net_def.op.append(op)

    return op_name_list, name_op_dict, net_def



if __name__ == "__main__":
    
    # op_name_list, op_dict, net_def = gather_model_profile(
    #     "/mnt/d/home/Projects/DAG-scheduler/mnn/inception-v3-info.txt",
    #     "/mnt/d/home/Projects/DAG-scheduler/mnn/redmi_data_trans.txt",
    #     "/mnt/d/home/Projects/DAG-scheduler/mnn/redmi_inception-v3-layerwise-latency.txt")
    op_name_list, op_dict, net_def = gather_model_profile(
        "/mnt/d/home/Projects/DAG-scheduler/mnn/pnasnet-mobile/pnasnet-info.txt",
        "/mnt/d/home/Projects/DAG-scheduler/mnn/redmi_data_trans.txt",
        "/mnt/d/home/Projects/DAG-scheduler/mnn/experimental_result_mnn/redmi-pnasnet-mobile-latency.csv")
    for op_name in op_name_list:
        print(op_dict[op_name])
    