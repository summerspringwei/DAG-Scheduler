
from utils import utils
from profile import read_profile_data
from profile import subgraph
from profile import net_struct
from solver import scheduler_utils
from operator import attrgetter



def get_avaliable_time_slice(occupy_list, start_point, latency):
    for i in range(len(occupy_list) - 1):
        device_avaliable_time = max(occupy_list[i][1], start_point)
        if device_avaliable_time + latency <= occupy_list[i+1][0]:
            return (i, device_avaliable_time, device_avaliable_time + latency)
    device_avaliable_time = max(occupy_list[-1][1], start_point)
    return len(occupy_list), device_avaliable_time, device_avaliable_time + latency


def heft_scheduler(op_name_list, name_op_dict, model, mobile, thread):
    """ Scheduler based on bottom level priority and 
    """
    ops_not_support_by_GPU = []
    # Record the CPU queue and GPU queue finish timestamp
    CPU_end_point = 0.0
    GPU_end_point = 0.0
    # Need to record the execute order
    op_execution_order_list = []
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
    CPU_task_list, GPU_task_list = [""], [""]
    task_execution_list = []
    for op in ops_list:
        CPU_latency, GPU_latency = scheduler_utils.get_ops_total_latency(op, name_op_dict)
        cpu_idx, cpu_start, cpu_end = get_avaliable_time_slice(CPU_occupy_list, op.earlist_start_point, CPU_latency)
        gpu_idx, gpu_start, gpu_end = get_avaliable_time_slice(GPU_occupy_list, op.earlist_start_point, GPU_latency)
        if cpu_end < gpu_end:
            CPU_occupy_list.insert(cpu_idx, (cpu_start, cpu_end))
            CPU_task_list.insert(cpu_idx, op.name)
            scheduler_utils.update_children_start_point(op, name_op_dict, cpu_start, CPU_latency)
            op.op_def.device_type = net_struct.DeviceType.CPU
            task_execution_list.append((cpu_start, op.name, net_struct.DeviceType.CPU))
        else:
            GPU_occupy_list.insert(gpu_idx, (gpu_start, gpu_end))
            GPU_task_list.insert(gpu_idx, op.name)
            scheduler_utils.update_children_start_point(op, name_op_dict, gpu_start, GPU_latency)
            op.op_def.device_type = net_struct.DeviceType.GPU
            task_execution_list.append((gpu_start, op.name, net_struct.DeviceType.GPU))
    sorted(task_execution_list, key=lambda task: task[0])
    
    logger = utils.get_logger()
    logger.info(task_execution_list)
    logger.info("CPU task occupy")
    logger.info(CPU_occupy_list)
    logger.info(CPU_task_list)
    logger.info("GPU task occupy")
    logger.info(GPU_occupy_list)
    logger.info(GPU_task_list)


def test_heft_scheduler():
    model, mobile, thread, _ = utils.parse_model_mobile()
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread)
    heft_scheduler(op_name_list, name_op_dict, model, mobile, thread)
    


if __name__=="__main__":
    test_heft_scheduler()
