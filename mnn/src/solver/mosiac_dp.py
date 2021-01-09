import os

from profile import read_profile_data
from utils import utils

MAX_COST = 1e6

class Slice:
    def __init__(self):
        self.cost = 0.0
        self.start_index = 0
        self.end_index = 0
        self.device = 0
    
    def __str__(self):
        return "(%d, %d), %f, %d" % (self.start_index, self.end_index, self.cost, self.device)


class SliceSet:
    def __init__(self):
        self.slice_set = set()
        self.cost = 0.0


def prepare_slice_latency(op_name_list, name_op_dict):
    acc_latency = []
    acc_cpu = 0.0
    acc_gpu = 0.0
    acc_latency.append((acc_cpu, acc_gpu, 0.0, 0.0))
    for op_name in op_name_list:
        op = name_op_dict[op_name]
        cpu_latency = op.op_def.operator_latency.CPU_latency
        gpu_latency = op.op_def.operator_latency.GPU_latency
        acc_cpu += cpu_latency
        acc_gpu += gpu_latency
        acc_latency.append((acc_cpu, acc_gpu, \
            op.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW, \
                op.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW))
    return acc_latency


def get_slice_latency(i, j, acc_latency):
    assert(i<j)
    cpu_i2j = acc_latency[j][0] - acc_latency[i][0]
    gpu_i2j = acc_latency[j][1] - acc_latency[i][1]
    return (cpu_i2j, gpu_i2j, acc_latency[j][2], acc_latency[j][3])


def find_efficient_slice_and_execution_plan(op_name_list, name_op_dict):
    acc_latency = prepare_slice_latency(op_name_list, name_op_dict)
    devices = [0, 1]
    slice_sets_list = []
    LENGTH = len(op_name_list)
    for l in range(0, LENGTH+1):
        slice_set = SliceSet()
        slice_sets_list.append(slice_set)
    for l in range(1, LENGTH):
        cost = 1000000
        slice_set = SliceSet()
        slice_set.cost = MAX_COST
        for k in range(0, l):
            # Note k+1 to l+1 get the cost from (k, l]
            (cpu_latency, gpu_latency, c2g, g2c) = get_slice_latency(k+1, l+1, acc_latency)
            device_latencies = [cpu_latency, gpu_latency]
            trans_cost = [g2c, c2g]
            sli = Slice()
            sli.start_index = k
            sli.end_index = l
            for d in devices:
                sli.device = d
                cost = slice_sets_list[k].cost + device_latencies[d]
                print("cost: %f" %(cost))
                if cost < sli.cost:
                    slice_sets_list = slice_sets_list[k]
                    slice_set.slice_set.add(sli)
                    slice_set.cost = cost
        slice_sets_list[l] = slice_set
    
    return slice_sets_list[LENGTH]


def get_sample_ops():
    latency_list = [(1,3,1,1), (1,3,1,1), (4,0.5,1,1), (4,1,1,1), (1,3,1,1), (1,3,1,1)]
    op_name_list= []
    name_op_dict = {}
    idx = 0
    for latency in latency_list:
        op = Operator("nade_%d" % idx)
        idx += 1
        op.op_def.operator_latency.CPU_latency = latency[0]
        op.op_def.operator_latency.GPU_latency = latency[1]
        op.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC = latency[2]
        op.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW = latency[3]
        op_name_list.append(op.name)
        name_op_dict[op.name] = op
    
    for op_name in op_name_list:
        print(name_op_dict[op_name])
    return op_name_list, name_op_dict


def my_serial_dp(op_name_list, name_op_dict):
    acc_latency = prepare_slice_latency(op_name_list, name_op_dict)
    devices = [0, 1]
    dp = []
    device_dp = []
    LENGTH = len(op_name_list)
    for i in range(LENGTH+1):
        if i == 0:
            dp.append(0)
            device_dp.append(0)
            continue
        (cpu_latency, gpu_latency, c2g, g2c) = get_slice_latency(0, i, acc_latency)
        dp.append(cpu_latency)
        device_dp.append(0)

    for l in range(1, LENGTH+1):
        cost = 1e6
        # print("*-" * 10)
        for k in range(l):
            (cpu_latency, gpu_latency, c2g, g2c) = get_slice_latency(k, l, acc_latency)
            device_latencies = [cpu_latency, gpu_latency]
            trans_costes = [g2c, c2g]
            # print("(%d %d)->(%f, %f)" % (k, l, cpu_latency, gpu_latency))
            for d in devices:
                trans_latency = 0.0
                if device_dp[k] != d:
                    trans_latency = trans_costes[d]
                slice_cost = dp[k] + device_latencies[d] + trans_latency
                # print("last device %d, device %d, dp[%d] %f, device_latency %f, trans_latency %f, cost %f" \
                #     % (device_dp[k], d, k, dp[k], device_latencies[d], trans_latency, cost))
                if slice_cost < cost:
                    tmp_device = d
                    cost = slice_cost
        device_dp[l] = tmp_device
        dp[l] = cost
        # print(dp[l])
        # print('-*' * 10)
    return dp[LENGTH], device_dp


def my_parallel_dp(op_name_list, name_op_dict):
    acc_latency = prepare_slice_latency(op_name_list, name_op_dict)
    devices = [0, 1]
    dp = []
    device_dp = []
    LENGTH = len(op_name_list)
    for i in range(LENGTH+1):
        if i == 0:
            dp.append(0)
            device_dp.append(0)
            continue
        (cpu_latency, gpu_latency, c2g, g2c) = get_slice_latency(0, i, acc_latency)
        dp.append(cpu_latency)
        device_dp.append(0)
    CPU_endpoint = 0.0
    GPU_endpoint = 0.0
    for l in range(1, LENGTH+1):
        cost = 1e6
        # print("*-" * 10)
        for k in range(l):
            (cpu_latency, gpu_latency, c2g, g2c) = get_slice_latency(k, l, acc_latency)
            device_latencies = [cpu_latency, gpu_latency]
            trans_costes = [g2c, c2g]
            # print("(%d %d)->(%f, %f)" % (k, l, cpu_latency, gpu_latency))
            for d in devices:
                trans_latency = 0.0
                if device_dp[k] != d:
                    trans_latency = trans_costes[d]
                slice_cost = dp[k] + device_latencies[d] + trans_latency
                # print("last device %d, device %d, dp[%d] %f, device_latency %f, trans_latency %f, cost %f" \
                #     % (device_dp[k], d, k, dp[k], device_latencies[d], trans_latency, cost))
                if slice_cost < cost:
                    tmp_device = d
                    cost = slice_cost
        device_dp[l] = tmp_device
        dp[l] = cost
        # print(dp[l])
        # print('-*' * 10)
    return dp[LENGTH], device_dp


def generate_mosaic_device_map(model, mobile, thread, op_name_list, device_dp):
    model_dir = os.path.join("../models/", model, mobile)
    file_name = "mosaic-placement-{}-cpu-{}.txt".format(model, thread)
    file_path = os.path.join(model_dir, file_name)
    f = open(file_path, 'w')
    lines = []
    for i in range(len(op_name_list)):
        device = 0
        if device_dp[i+1] == 1:
            device = 3
        line = "{} {}\n".format(op_name_list[i], device)
        lines.append(line)
    f.writelines(lines)
    f.flush()
    f.close()
    print("Write placement {}".format(file_path))
    os.system('adb push {} {}'.format(file_path, '/data/local/tmp/'))


def solve_mosaic_dp(model, mobile, thread):
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread)
    # find_efficient_slice_and_execution_plan(op_name_list, name_op_dict)
    result = my_serial_dp(op_name_list, name_op_dict)
    print("MOSAIC result: {}".format(result[0]))
    generate_mosaic_device_map(model, mobile, thread, op_name_list, result[1])
    

def test_my_dp():
    op_name_list, name_op_dict = get_sample_ops()
    # find_efficient_slice_and_execution_plan(op_name_list, name_op_dict)
    result = my_serial_dp(op_name_list, name_op_dict)
    print(result)


if __name__ == "__main__":
    # test_my_dp()
    model, mobile, thread, _ = utils.parse_model_mobile()
    solve_mosaic_dp(model, mobile, thread)
    
