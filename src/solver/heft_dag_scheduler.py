
from utils import utils
from profile import read_profile_data
from profile import subgraph
from profile import net_struct
from solver import scheduler_utils
from operator import attrgetter

import os


def get_avaliable_time_slice(occupy_list, start_point, latency):
    for i in range(len(occupy_list) - 1):
        device_avaliable_time = max(occupy_list[i][1], start_point)
        if device_avaliable_time + latency <= occupy_list[i+1][0]:
            return (i, device_avaliable_time, device_avaliable_time + latency)
    device_avaliable_time = max(occupy_list[-1][1], start_point)
    return len(occupy_list), device_avaliable_time, device_avaliable_time + latency


def heft_scheduler(op_name_list, name_op_dict, model, mobile, thread, CPU_little_thread_index=0):
    """ Scheduler based on bottom level priority and 
    """
    gpu_not_support_op_names = ['final_layer/Mean', 'final_layer/Mean/reduction_indices', \
        'final_layer/Relu', 'final_layer/FC/weights', 'final_layer/FC/biases']
    # Need to record the execute order
    input_op_names = subgraph.find_input_nodes(op_name_list, name_op_dict)
    ops_queue = [name_op_dict[op_name] for op_name in input_op_names]
    print("input op name: ", input_op_names)
    assert(len(ops_queue) > 0)
    print("Start bottom level macro scheduler")
    scheduler_utils.compute_bottom_level(op_name_list, name_op_dict, scheduler_utils.BottomLevelFuncType.RANK)
    ops_list = [name_op_dict[op_name] for op_name in op_name_list]
    ops_list = sorted(ops_list, key=attrgetter("bottom_level"), reverse=True)
    idx = 0
    CPU_occupy_list, GPU_occupy_list = [(0, 0)], [(0, 0)] # Stores tuple (start_time, end_time)
    cpu_end, gpu_end = 0, 0
    task_execution_list = []
    for op in ops_list:
        CPU_latency, GPU_latency = scheduler_utils.get_ops_total_latency(op, name_op_dict)
        cpu_idx, cpu_start, cpu_end = get_avaliable_time_slice(CPU_occupy_list, op.earlist_start_point, CPU_latency)
        gpu_idx, gpu_start, gpu_end = get_avaliable_time_slice(GPU_occupy_list, op.earlist_start_point, GPU_latency)
        if cpu_end < gpu_end or op.name in gpu_not_support_op_names:
            CPU_occupy_list.insert(cpu_idx, (cpu_start, cpu_end))
            scheduler_utils.update_children_start_point(op, name_op_dict, cpu_start, CPU_latency)
            op.op_def.device_type = net_struct.DeviceType.CPU
            task_execution_list.append((cpu_start, op.name, net_struct.DeviceType.CPU))
        else:
            GPU_occupy_list.insert(gpu_idx, (gpu_start, gpu_end))
            scheduler_utils.update_children_start_point(op, name_op_dict, gpu_start, GPU_latency)
            op.op_def.device_type = net_struct.DeviceType.GPU
            task_execution_list.append((gpu_start, op.name, net_struct.DeviceType.GPU))
    sorted(task_execution_list, key=lambda task: task[0])
    
    logger = utils.get_logger()
    logger.info("HEFT scheduler result: {}".format(max(cpu_end, gpu_end)))
    # logger.info("All task execution list")
    # logger.info(task_execution_list)
    
    # Write results to file
    lines = []
    for (start_time, op_name, device) in task_execution_list:
        lines.append("{} {}\n".format(op_name, device))
    
    file_name = "heft-placement-{}-cpu-{}.txt".format(model, thread)
    file_path = os.path.join(utils.get_project_path(), "mnn/models/", model, mobile, file_name)

    new_lines = []
    if CPU_little_thread_index != None:
        file_path = os.path.join(utils.get_project_path(), "mnn/models/", model, mobile,\
            "mDeviceMap-{}-cpu-big-{}-little-{}.txt".format(model, thread, CPU_little_thread_index))
        # Little core device type is 2
        for line in lines:
            line = line.replace(' 3', ' 2')
            new_lines.append(line)
        lines = new_lines
    utils.write_lines(file_path, lines)
    sh_cmd = "adb push {} /data/local/tmp/".format(file_path)
    print(sh_cmd)
    os.system(sh_cmd)
    # Special for ACL
    if model.find("acl") >= 0:
      model_name = model.split('-')[-1]
      run_cmd = 'adb shell "cd /data/local/tmp/ && ./acl-run.sh {} CL parallel {} {}"'.format(model_name, thread, "greedy-placement-{}-cpu-{}.txt".format(model, thread))
      os.system(run_cmd)


def test_heft_scheduler():
    model, mobile, thread, little_idx = utils.parse_model_mobile()
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread, CPU_little_thread_index=little_idx)
    heft_scheduler(op_name_list, name_op_dict, model, mobile, thread, CPU_little_thread_index=little_idx)
    


if __name__=="__main__":
    test_heft_scheduler()
