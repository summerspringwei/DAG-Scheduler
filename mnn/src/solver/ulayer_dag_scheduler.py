
import os

from utils import utils
from profile import read_profile_data
from profile import net_struct

def ulayer_scheduler(op_name_list, name_op_dict, model, mobile, thread):
    """ Scheduler based on bottom level priority and 
    """
    device_map_lines = []
    for op_name in op_name_list:
        op_name_low_case = op_name.lower()
        if "conv" in op_name_low_case or "depthwise" in op_name_low_case or \
            "pointwise" in op_name_low_case or 'expand' in op_name_low_case or\
                "squeeze" in op_name_low_case:
            op = name_op_dict[op_name]
            cpu_latency = op.op_def.operator_latency.CPU_latency
            gpu_latency = op.op_def.operator_latency.GPU_latency
            ratio = 0.0
            min_latency = gpu_latency
            for r in [0.0, 0.25, 0.5, 0.75, 1]:
                if max(r* cpu_latency, (1-r)*gpu_latency) < min_latency:
                    ratio = r
            # ratio = 0.25
            line = "{} {}\n".format(op_name, ratio)
            device_map_lines.append(line)
    # TODO(xcw) Compute the ratio of workload on CPU and GPU
    file_name = "ulayer-placement-{}-cpu-{}.txt".format(model, thread)
    file_path = os.path.join(utils.get_project_path(), "mnn/models/", model, mobile, file_name)
    f = open(file_path, 'w')
    f.writelines(device_map_lines)
    f.flush()
    f.close()
    sh_cmd = "adb push {} /data/local/tmp/".format(file_path)
    os.system(sh_cmd)
    model_name = model.split("-")[-1]
    run_cmd = 'adb shell "cd /data/local/tmp/ && ./acl-run.sh {} CL ulayer {} {}"'.format(model_name, thread, file_name)
    os.system(run_cmd)


if __name__=="__main__":
    model, mobile, thread, little_idx = utils.parse_model_mobile()
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread, CPU_little_thread_index=little_idx)
    ulayer_scheduler(op_name_list, name_op_dict, model, mobile, thread)
    