
import os
import logging

from profile import read_profile_data
from solver import generate_LP
from solver import parse_lp_solution
from utils import utils
from profile import subgraph

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
datefmt='%Y-%m-%d:%H:%M:%S',
level=logging.INFO)
logging.root.setLevel(logging.INFO)
logger = logging.getLogger()


def replace_GPU_with_little(model_dir, mobile, thread, lines, CPU_little_thread_index):
    new_lines = []
    device_map_file_path = os.path.join(model_dir, mobile, \
        "mDeviceMap-{}-cpu-big-{}-little-{}.txt".format(model, thread, CPU_little_thread_index))
    # Little core device type is 2
    for line in lines:
        line = line.replace(' 3', ' 2')
        new_lines.append(line)

    return device_map_file_path, new_lines


def push_device_placement_file(file_path, model, mobile, thread):
    sh_cmd = "adb push {} /data/local/tmp/".format(file_path)
    logger.info(sh_cmd)
    os.system(sh_cmd)
    # Special for ACL
    model_name = model.split('-')[-1]
    file_name = "mDeviceMap-{}-cpu-{}.txt".format(model, thread)
    run_cmd = 'adb shell "cd /data/local/tmp/ && ./acl-run.sh {} CL parallel {} {}"'.format(model_name, thread, file_name)
    os.system(run_cmd)


def solve_model(model, mobile, thread, module_list, unsupported_op_list, \
        mode=generate_LP.LPMode.Mode_Subgraph, CPU_little_thread_index=None):
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread, \
        SCALE=1.0, CPU_little_thread_index=CPU_little_thread_index)
    net_def = None
    model_dir = os.path.join("../models/", model)
    net_def = None
    folder_path = os.path.join(model_dir, mobile)
    lines, intersection_list = generate_LP.solve_glpk(op_name_list, name_op_dict, net_def, module_list, folder_path, model, mode=mode)
    lines, untreated_op_latency = subgraph.insert_untreated_ops(lines, op_name_list, name_op_dict)
    lp_total = parse_lp_solution.sum_lp_objectives(folder_path)
    logger.info("LP+serial total: {}".format(lp_total+untreated_op_latency))
    logger.info("LP+serial+intersection total: {}".format(sum([(endpoint+intersection) for endpoint, intersection in intersection_list]) + untreated_op_latency))

    device_map_file_path = os.path.join(model_dir, mobile, "mDeviceMap-{}-cpu-{}.txt".format(model, thread))
    if CPU_little_thread_index != None:
        device_map_file_path, lines = replace_GPU_with_little(model_dir, \
            mobile, thread, lines, CPU_little_thread_index)
    utils.write_lines(device_map_file_path, lines)
    push_device_placement_file(device_map_file_path, model, mobile, thread)


def solve_pnasnet(model, mobile, thread, CPU_little_thread_index=None):
    # Using module prefix to form the subgraph
    pnasnet_module_list = ['cell_stem_0/', 'cell_stem_1/']
    if model in ['pnasnet-large', 'acl-pnasnet_large']:
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
    elif model in ['pnasnet-mobile', 'acl-pnasnet_mobile']:
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(9)])
    elif model in ['nasnet-large', 'acl-nasnet_large']:
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(18)])
        pnasnet_module_list.insert(8, 'reduction_cell_0/')
        pnasnet_module_list.insert(15, 'reduction_cell_1/')
        logger.info('aaa', pnasnet_module_list)
    elif model in ['nasnet-mobile', 'acl-nasnet_mobile']:
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
        pnasnet_module_list.insert(6, 'reduction_cell_0/')
        pnasnet_module_list.insert(11, 'reduction_cell_1/')
    else:
        logger.info("Model %s does not suport yet." % (model))
        return
    
    unsupported_op_names = ["final_layer/Relu", "final_layer/Mean/reduction_indices", \
        "final_layer/Relu___tr4final_layer/Mean", "final_layer/Mean", \
        "final_layer/FC/weights", "final_layer/FC/MatMul", \
        "final_layer/FC/biases", "final_layer/FC/BiasAdd", "final_layer/predictions"]
    solve_model(model, mobile, thread, pnasnet_module_list, unsupported_op_names, \
        mode=generate_LP.LPMode.Mode_Subgraph, CPU_little_thread_index=CPU_little_thread_index)
    

def solve_inception(model, mobile, thread, CPU_little_thread_index=None):
    if model == "inception-v3":
        inception_prefix = "InceptionV3/InceptionV3/"
        inception_module_list = ["Mixed_5b/", "Mixed_5c/", "Mixed_5d/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/", "Mixed_7a/", "Mixed_7b/", "Mixed_7c/"]
    elif model == "acl-inception_v3":
        inception_prefix = ""
        inception_module_list = ["Mixed_5b/", "Mixed_5c/", "Mixed_5d/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/", "Mixed_7a/", "Mixed_7b/", "Mixed_7c/"]
    elif model == "inception-v4":
        inception_prefix = "InceptionV4/InceptionV4/"
        inception_module_list = ["Mixed_4a/", "Mixed_5b/", "Mixed_5c/", "Mixed_5d/", "Mixed_5e/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/","Mixed_6f/","Mixed_6g/","Mixed_6h/",\
            "Mixed_7a/","Mixed_7b/","Mixed_7c/","Mixed_7d/",]
    elif model == "acl-inception_v4":
        inception_prefix = ""
        inception_module_list = ["Mixed_4a/", "Mixed_5b/", "Mixed_5c/", "Mixed_5d/", "Mixed_5e/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/","Mixed_6f/","Mixed_6g/","Mixed_6h/",\
            "Mixed_7a/","Mixed_7b/","Mixed_7c/","Mixed_7d/",]
    inception_module_list_full = [inception_prefix + module for module in inception_module_list]
    unsupported_op_names = []
    solve_model(model, mobile, thread, inception_module_list_full, unsupported_op_names, \
        mode=generate_LP.LPMode.Mode_Subgraph, CPU_little_thread_index=CPU_little_thread_index)


def solve_whole_model(model, mobile, thread, CPU_little_thread_index=None):
    module_list = ["model"]
    unsupported_op_names = []
    solve_model(model, mobile, thread, module_list, unsupported_op_names, \
        mode=generate_LP.LPMode.Mode_AUTO_Subgraph, CPU_little_thread_index=CPU_little_thread_index)


if __name__ == "__main__":
    # model, mobile, thread, CPU_little_thread_index = utils.parse_model_mobile()
    model, mobile, thread = "acl-nasnet_large", "huawei_p40", 2
    CPU_little_thread_index = None
    solve_whole_model(model, mobile, thread, CPU_little_thread_index=CPU_little_thread_index)
    exit(0)
    if model in ['pnasnet-mobile', 'pnasnet-large', 'nasnet-large', 'nasnet-mobile', 
                 'acl-pnasnet_mobile', 'acl-pnasnet_large', 'acl-nasnet_large', 'acl-nasnet_mobile']:
        solve_pnasnet(model, mobile, thread, CPU_little_thread_index=CPU_little_thread_index)
    elif model in ['inception-v3', 'inception-v4', 'acl-inception_v3', 'acl-inception_v4']:
        solve_inception(model, mobile, thread, CPU_little_thread_index=CPU_little_thread_index)
    else:
        solve_whole_model(model, mobile, thread, CPU_little_thread_index=CPU_little_thread_index)