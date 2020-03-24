
'''
In this file we read all the ops of a multi-run benchmark results 
with format (op_name, device, latency)
and analyze the CPU or GPU performance latency suppression ratio.
'''

from read_profile_data import *
import numpy as np


def read_multi_runs_latency(file_path):
    f_profile = open(file_path, 'r')
    profile_dict = {}
    for line in f_profile.readlines():
        com = line.strip().split(' ')
        op_name = com[0].strip()
        if op_name not in profile_dict.keys():
            profile_dict[op_name] = (com[0].strip(), com[1].strip(), [float(com[2].strip()), ])
        else:
            profile_dict[op_name][2].append(float(com[2].strip()))
    f_profile.close()
    return profile_dict


def filter_list(original_list):
    avg = np.average(original_list)
    new_list = []
    for num in original_list:
        if num <= avg*1.2:
            new_list.append(num)
    return new_list


def gather_profile(parallel_file_path, alone_file_path, co_run_file_path, result_file_path):
    parallel_profile_dict = read_multi_runs_latency(parallel_file_path)
    alone_profile_dict = read_multi_runs_latency(alone_file_path)
    co_run_profile_dict = read_multi_runs_latency(co_run_file_path)
    op_name_list, _ = read_net_info("/mnt/d/home/Projects/DAG-scheduler/mnn/inception-v3-info.txt")
    result_lines = []
    for op_name in op_name_list:
        parallel_avg = np.average(filter_list(parallel_profile_dict[op_name][2]))
        alone_avg = np.average(filter_list(alone_profile_dict[op_name][2]))
        co_run_avg = np.average(filter_list(co_run_profile_dict[op_name][2]))
        # print(parallel_profile_dict[op_name][1])
        line = "%s %s %f %f %f\n" % (op_name, parallel_profile_dict[op_name][1], parallel_avg, alone_avg, co_run_avg)
        result_lines.append(line)
    f_result = open(result_file_path, 'w')
    f_result.writelines(result_lines)
    f_result.flush()
    f_result.close()


def gather_1x1_profile(alone_file_path, co_run_file_path):
    alone_profile_dict = read_multi_runs_latency(alone_file_path)
    co_run_profile_dict = read_multi_runs_latency(co_run_file_path)
    for op_name, op_profile in alone_profile_dict.items():
        alone_avg = np.average(filter_list(alone_profile_dict[op_name][2]))
        co_run_avg = np.average(filter_list(co_run_profile_dict[op_name][2]))
        line = "%s %f %f" % (op_name, alone_avg, co_run_avg)
        print(line)


def gather_net_profile(cpu_alone_path, gpu_alone_path, parallel_path, result_file_path):
    cpu_alone_profile_dict = read_multi_runs_latency(cpu_alone_path)
    gpu_alone_profile_dict = read_multi_runs_latency(gpu_alone_path)
    parallel_profile_dict = read_multi_runs_latency(parallel_path)

    op_name_list, _ = read_net_info("/mnt/d/home/Projects/DAG-scheduler/mnn/inception-v3-info.txt")
    lines = []

    for op_name in op_name_list:
        cpu_alone_avg = np.average(filter_list(cpu_alone_profile_dict[op_name][2]))
        gpu_alone_avg = np.average(filter_list(gpu_alone_profile_dict[op_name][2]))
        parallel_avg = np.average(filter_list(parallel_profile_dict[op_name][2]))
        device_name = parallel_profile_dict[op_name][1]
        line = ''
        if device_name == 'CPU':
            line = "%s %s %f %f\n" % (op_name, device_name, cpu_alone_avg, parallel_avg)
        else:
            line = "%s %s %f %f\n" % (op_name, device_name, gpu_alone_avg, parallel_avg)
        lines.append(line)
    
    f_result = open(result_file_path, 'w')
    f_result.writelines(lines)
    f_result.flush()
    f_result.close()



def gather_multi_file_profile(files_list, raw_info_file_path, result_file_path):
    profile_dict_list = []
    for file_path in files_list:
        profile_dict_list.append(read_multi_runs_latency(file_path))
    op_name_list, _= read_net_info(raw_info_file_path)    
    lines = []
    count = 0
    for op_name in op_name_list:
        has_name = True
        line = op_name
        for profile_dict in profile_dict_list:
            if op_name not in profile_dict.keys():
                has_name = False
                break
            avg_latency = np.average(filter_list(profile_dict[op_name][2]))
            line += (" " + str(avg_latency))
        if has_name:
            count += 1
            line += "\n"
            lines.append(line)
    # print(lines)
    print(count)
    f_result = open(result_file_path, 'w')
    f_result.writelines(lines)
    f_result.flush()
    f_result.close()


if __name__ == "__main__":
    # gather_profile("inception-inteference/redmi-cpu-4-parallel-20avgpool.csv", \
    #     "inception-inteference/redmi-cpu-4-alone-multi-0302.csv", \
    #         "inception-inteference/redmi-cpu-4-affinity-multi-0228.csv", \
    #             "inception-inteference/cpu-4-gather-avg-multi0302.csv")
    # gather_1x1_profile("inception-inteference/redmi-3x3-co-run-with-avg-same-core.csv",\
    #     "inception-inteference/redmi-3x3-alone-data.csv")
    # gather_net_profile("experimental_result_mnn/redmi-inception-cpu-4.csv", \
    #     "experimental_result_mnn/redmi-inception-gpu.csv", \
    #     "experimental_result_mnn/tmp.csv", \
    #     "experimental_result_mnn/tmp2.csv")
    # gather_multi_file_profile(["pnasnet-mobile/redmi-pnasnet-mobile-cpu-1.csv", \
    #     "pnasnet-mobile/redmi-pnasnet-mobile-cpu-2.csv", \
    #     "pnasnet-mobile/redmi-pnasnet-mobile-cpu-4.csv", \
    #     "pnasnet-mobile/redmi-pnasnet-mobile-gpu-onwait.csv"], \
    #     "pnasnet-mobile/pnasnet-info.txt", \
    #     "pnasnet-mobile/redmi-pnasnet-mobile-latency-onwait.csv")
    # gather_multi_file_profile(["inception-v4/redmi-inception-v4-cpu-1.csv", \
    #     "inception-v4/redmi-inception-v4-cpu-2.csv", \
    #         "inception-v4/redmi-inception-v4-cpu-4.csv", \
    #             "inception-v4/redmi-inception-v4-gpu.csv"], \
    #                 "inception-v4/inception-v4-info.txt", \
    #                 "inception-v4/redmi-inception-v4-layerwise-latency.csv")
    # gather_multi_file_profile(["lanenet/oneplus-3-lanenet-cpu-1.csv", \
    #     "lanenet/oneplus-3-lanenet-cpu-2.csv", \
    #         "lanenet/oneplus-3-lanenet-cpu-4.csv", \
    #             "lanenet/oneplus-3-lanenet-gpu.csv"], \
    #                 "lanenet/lanenet-info.txt", \
    #                 "lanenet/oneplus-3-lanenet-layerwise-latency.csv")
    gather_multi_file_profile(["lanenet/oneplus3-lanenet-cpu-1.csv", \
        "lanenet/oneplus3-lanenet-cpu-2-serial-hybrid.csv"], \
        "lanenet/", "lanenet/oneplus3-lanenet-cpu-2-serial-hybrid-compare.csv")

# InceptionV3\/InceptionV3\/Mixed_6e\/Branch_2\/Conv2d_0a_1x1\/Conv2D
