
import numpy as np

from utils import *
from measure_inteference import *


if __name__ == "__main__":
    model, mobile, thread = parse_model_mobile()
    model_dir = os.path.join("/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/", model)
    folder_path = os.path.join(model_dir, mobile)
    op_name_list, name_op_dict, net_def = gather_model_profile(
            os.path.join(model_dir, model + "-info.txt"),
            os.path.join(model_dir, mobile, model+'-'+mobile+'-data-trans.csv'),
            os.path.join(model_dir, mobile, mobile+"-"+model+"-layerwise-latency.csv"),
            thread)
    result_file_name = os.path.join(model_dir, mobile, model+'-'+mobile+'-cpu-'+str(thread)+'compare.csv')
    profile_file_name = os.path.join(model_dir, mobile, "profile.txt")
    sh_cmd = "adb pull /data/local/tmp/profile.txt {}".format(profile_file_name)
    print(sh_cmd)
    os.system(sh_cmd)
    parallel_file_name = os.path.join(model_dir, mobile, "tmp.csv")
    sh_cmd = 'cat {} | grep Iter | awk \'{{print $3, $5, $6, $7, $8}}\' > {}'.format(profile_file_name, parallel_file_name)
    print(sh_cmd)
    os.system(sh_cmd)
    parall_op_latency_dict = read_multi_runs_latency(parallel_file_name)
    # op_name, device, alone, parallel
    lines = []
    for op_name in op_name_list:
        op = name_op_dict[op_name]
        parallel_op_latency = parall_op_latency_dict[op_name]
        parallel_latency = np.average(parallel_op_latency[2])
        device = parallel_op_latency[1]
        alone_latency = 0.0
        if device == "CPU":
            alone_latency = op.op_def.operatorLatency.CPU_latency*1000
        elif device == "OpenCL":
            alone_latency = op.op_def.operatorLatency.GPU_latency*1000
        line = "{},{},{},{}\n".format(op_name, device, alone_latency, parallel_latency)
        lines.append(line)
    write_lines(result_file_name, lines)
    print(result_file_name)
    
    