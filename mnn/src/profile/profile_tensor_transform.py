import os

from profile import read_profile_data
from utils import utils

def gather_tensors(model, mobile):
    model_dir = os.path.join("../models/", model)
    raw_info_file_path = os.path.join(model_dir, model + "-info.txt")
    cpu_to_gpu_file_path = os.path.join(model_dir, "mDeviceMap-"+model+"-cpu-to-gpu-tensor-trans.txt")
    gpu_to_cpu_file_path = os.path.join(model_dir, "mDeviceMap-"+model+"-gpu-to-cpu-tensor-trans.txt")
    gpu_not_support_op_names = ['final_layer/Mean', 'final_layer/Mean/reduction_indices', \
        'final_layer/Relu', 'final_layer/FC/weights', 'final_layer/FC/biases']
    name_op_list, name_op_dict = read_profile_data.read_net_info(raw_info_file_path)
    tensor_set = set()
    used_op = set()
    cpu_to_gpu_lines = []
    gpu_to_cpu_lines = []
    for op_name in name_op_list:
        if op_name in gpu_not_support_op_names:
            cpu_to_gpu_lines.append("%s %d\n" % (op_name, 0))
            gpu_to_cpu_lines.append("%s %d\n" % (op_name, 0))
            used_op.add(op_name)
            continue
        op = name_op_dict[op_name]
        for ot in op.output_tensors:
            tensor_set.add(ot)
        # Set parent to cpu and child to gpu
        if op_name not in used_op:
            children = op.children
            for child in children:
                if child not in used_op:
                    cpu_to_gpu_lines.append("%s %d\n" % (op_name, 0))
                    cpu_to_gpu_lines.append("%s %d\n" % (child, 3))
                    gpu_to_cpu_lines.append("%s %d\n" % (op_name, 3))
                    gpu_to_cpu_lines.append("%s %d\n" % (child, 0))
                    used_op.add(child)
            used_op.add(op_name)
                
    f_cpu_to_gpu = open(cpu_to_gpu_file_path, 'w')
    f_cpu_to_gpu.writelines(cpu_to_gpu_lines)
    f_cpu_to_gpu.flush()
    f_cpu_to_gpu.close()
    f_gpu_to_cpu = open(gpu_to_cpu_file_path, 'w')
    f_gpu_to_cpu.writelines(gpu_to_cpu_lines)
    f_gpu_to_cpu.flush()
    f_gpu_to_cpu.close()


if __name__ == "__main__":
    model, mobile, thread, _ = utils.parse_model_mobile()
    gather_tensors(model, mobile)
    