
import numpy as np
import os

from utils import utils
from analyze import measure_interference
from profile import read_profile_data

if __name__ == "__main__":
    model, mobile, thread, CPU_little_thread_index = utils.parse_model_mobile()
    # sh_cmd='adb shell "cd /data/local/tmp && source set_env.sh && ./grun_mnn.sh {} {}"'.format(model, thread)
    # print(sh_cmd)
    # os.system(sh_cmd)

    model_dir = os.path.join(utils.get_project_path(), "mnn/models/", model)
    folder_path = os.path.join(model_dir, mobile)
    op_name_list, name_op_dict, net_def = read_profile_data.gather_model_profile(
            os.path.join(model_dir, model + "-info.txt"),
            os.path.join(model_dir, mobile, model+'-'+mobile+'-data-trans.csv'),
            os.path.join(model_dir, mobile, mobile+"-"+model+"-layerwise-latency.csv"),
            thread, CPU_little_thread_index=CPU_little_thread_index)
    result_file_name = os.path.join(model_dir, mobile, model+'-'+mobile+'-cpu-'+str(thread)+'-compare.csv')
    profile_file_name = os.path.join(model_dir, mobile, "profile.txt")
    sh_cmd = "adb pull /data/local/tmp/profile.txt {}".format(profile_file_name)
    print(sh_cmd)
    os.system(sh_cmd)
    parallel_file_name = os.path.join(model_dir, mobile, "tmp.csv")
    sh_cmd = 'cat {} | grep Iter | awk \'{{print $3, $5, $6, $7, $8}}\' > {}'.format(profile_file_name, parallel_file_name)
    print(sh_cmd)
    os.system(sh_cmd)

    parall_op_latency_dict = measure_interference.read_multi_runs_latency(parallel_file_name)

    # op_name, device, alone, parallel
    lines = []
    cpu_ratio_list = []
    gpu_ratio_list = []
    convert_latency_list = []
    cpu_latency_list = []
    alone_cpu_latency_list = []
    gpu_latency_list = []
    alone_gpu_latency_list = []
    # Analyze CPU and GPU op's real latency
    for op_name in op_name_list:
        op = name_op_dict[op_name]
        parallel_op_latency = parall_op_latency_dict[op_name]
        parallel_latency = np.average(parallel_op_latency[2])
        device = parallel_op_latency[1]
        alone_latency = 0.0
        if device == "CPU":
            alone_latency = op.op_def.operator_latency.CPU_latency*1000
            alone_cpu_latency_list.append(alone_latency)
        elif device == "OpenCL" or device == "N/A":
            alone_latency = op.op_def.operator_latency.GPU_latency*1000
            alone_gpu_latency_list.append(alone_latency)
        else:
            print(parallel_latency)
        ratio = 0
        if alone_latency > 0:
            ratio = float(parallel_latency)/ float(alone_latency)
        if device == "CPU":
            cpu_ratio_list.append(ratio)
            cpu_latency_list.append(parallel_latency)
        elif device == "OpenCL":
            gpu_ratio_list.append(ratio)
            gpu_latency_list.append(parallel_latency)
        line = "{},{},{},{},{}\n".format(op_name, device, alone_latency, parallel_latency, ratio)
        lines.append(line)
    utils.write_lines(result_file_name, lines)
    # Analyze the convert latency
    for k, v in parall_op_latency_dict.items():
        if(v[1]=='Convert'):
            convert_latency_list.append(np.average(v[2]))
    # print(cpu_latency_list)
    # print(gpu_latency_list)
    # print(convert_latency_list)
    print("CPU sum of latency {} ms, GPU sum of latency {} ms, convert sum of latency {} ms".format(\
        np.sum(cpu_latency_list)/1000.0, np.sum(gpu_latency_list) / 1000.0, np.sum(convert_latency_list) / 1000.0))
    print("Alone CPU sum of latency {} ms, Alone GPU sum of latency {} ms,".format(\
        np.sum(alone_cpu_latency_list), np.sum(alone_gpu_latency_list)))
    print("Every op: CPU size {} average ratio {}, GPU size {} average ratio {}".format(\
        len(cpu_ratio_list) ,np.average(cpu_ratio_list), len(gpu_ratio_list), np.average(gpu_ratio_list)))
    print("Total op: CPU average ratio {}, GPU average ratio {}".format(\
        np.sum(cpu_latency_list)/ np.sum(alone_cpu_latency_list),
        np.average(gpu_latency_list)/ np.average(alone_gpu_latency_list)))
    print("Total convert size {}\n".format(len(convert_latency_list)))
    print(result_file_name)
