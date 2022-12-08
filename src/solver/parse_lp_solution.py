import os
import xml.etree.ElementTree as ET
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger()

from profile import subgraph

# DO NOT MODIFY THE VALUE OF `CPU` AND `GPU`
CPU = 1
GPU = 2
class LPMode:
    Mode_Subgraph = 0
    Mode_Operator = 1
    Mode_AUTO_Subgraph = 2

def parse_s_var(s):
    device = 0
    sc = s.split('_')
    if int(sc[1]) == CPU:
        device = 0
    elif int(sc[1]) == GPU:
        device = 3
    op_idx = sc[2]
    return op_idx, device

def parse_glpk_result_old(one_module_names_idx_dict, result_file_path):
    f = open(result_file_path, 'r')
    name_device_tuple_list = []
    lines = f.readlines()
    for line in lines:
        # Find lines with s_
        line = line.strip()
        if line.find('s_') >= 0:
            com = line.split(' ')
            striped_com = []
            for c in com:
                if c != '':
                    striped_com.append(c)
            
            if len(striped_com) == 6 and striped_com[3] == '1':
                # print(striped_com)
                s = striped_com[1]
                op_idx, device = parse_s_var(s)
                for name, idx in one_module_names_idx_dict.items():
                    if int(op_idx) == idx:
                        name_device_tuple_list.append((name, device))
                        break
    for (name, device) in name_device_tuple_list:
        print("%s %d" %(name, device))
    return name_device_tuple_list


def parse_cplex_result(one_module_names_idx_dict, result_file_path):
    tree = ET.parse(result_file_path)
    root = tree.getroot()
    name_device_tuple_list = []
    for child in root:
        # Get variables in cplex xml
        if child.tag == "variables":
            for variable in child:
                s = variable.attrib['name']
                value = variable.attrib['value']
                # Get variable 
                if value == '1' and s.find('s_') >= 0:
                    op_idx, device = parse_s_var(s)
                    for name, idx in one_module_names_idx_dict.items():
                        if int(op_idx) == idx:
                            name_device_tuple_list.append((name, device))
                            break
    for (name, device) in name_device_tuple_list:
        print("%s %d" %(name, device))
    return name_device_tuple_list


def get_glpk_variables(result_file_path):
    s_dict = {}
    t_dict = {}
    f = open(result_file_path, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.find('s_') >= 0 or line.find('t_') >=0:
            com = line.split(' ')
            striped_com = []
            for c in com:
                if c != '':
                    striped_com.append(c)
            if striped_com[1].find('s_')>=0 and striped_com[3] == '1':
                sc = striped_com[1].split('_')
                device, idx = sc[1], sc[2]
                s_dict[int(device), int(idx)] = int(striped_com[3])
            elif striped_com[1].find('t_')>=0:
                sc = striped_com[1].split('_')
                device, idx = sc[1], sc[2]
                t_dict[(int(device), int(idx))] = float(striped_com[2])
    f.close()
    return s_dict, t_dict

def get_cplex_variables(result_file_path):
    s_dict = {}
    t_dict = {}
    
    tree = ET.parse(result_file_path)
    root = tree.getroot()
    all_variables = [child for child in root if child.tag == "variables"][-1]
    
    for variable in all_variables:
        name = variable.attrib['name'].strip()
        value = variable.attrib['value'].strip()
        # logger.info("{} {}".format(name, value))
        # Get variable 
        # cplex has bug that binary value may be 1.00000x or 0.999999x
        if round(float(value)) == 1 and name.find('s_') >= 0:
            sc = name.split('_')
            device, idx = sc[1], sc[2]
            s_dict[int(device), int(idx)] = round(float(value))
        if name.find('t_') >= 0:
            sc = name.split('_')
            device, idx = sc[1], sc[2]
            t_dict[(int(device), int(idx))] = float(value)
    return s_dict, t_dict


def parse_glpk_result(one_module_names_idx_dict, result_file_path):
    name_device_tuple_list = []
    s_dict, _ = get_glpk_variables(result_file_path)
    idx_to_name_dict = {}
    for name, idx in one_module_names_idx_dict.items():
        idx_to_name_dict[idx] = name
    for device, idx in s_dict:
        name = idx_to_name_dict[idx]
        if device == CPU:
            device = 0
        elif device == GPU:
            device = 3
        name_device_tuple_list.append((name, device))

    return name_device_tuple_list

def parse_ilp_timeline(one_module_names_idx_dict, result_file_path, op_dict, mode=LPMode.Mode_Subgraph):
    idx_name_dict = {v:k for k,v in one_module_names_idx_dict.items()}
    print(idx_name_dict)
    
    # s_glpk_dict, t_glpk_dict = get_glpk_variables(result_file_path)
    s_cplex_dict, t_cplex_dict = get_cplex_variables(result_file_path+".xml")
    # logger.info(s_glpk_dict)
    # logger.info(s_cplex_dict)
    # logger.info(t_glpk_dict)
    # logger.info(t_cplex_dict)
    # assert(s_glpk_dict==s_cplex_dict)
    result = []
    s_dict, t_dict = s_cplex_dict, t_cplex_dict
    op_execution_order_list = []
    for (device, idx), _ in s_dict.items():
        start_time = t_dict[(device, idx)]
        op_name = idx_name_dict[idx]
        op = op_dict[op_name]
        device_latency = 0.0
        if device == CPU:
            device_latency = op.op_def.operator_latency.CPU_latency
        elif device == GPU:
            device_latency = op.op_def.operator_latency.GPU_latency
        # Compute data trans latency
        acc_data_trans_latency = 0.0
        for (addr, data_trans) in op.op_def.operator_latency.input_data_trans_latency.items():
            data_trans_latency = data_trans[device-1]
            for op_parent_name in op.parents:
                op_parent = op_dict[op_parent_name]
                if mode==LPMode.Mode_Subgraph and not isinstance(op_parent, subgraph.Subgraph):
                    continue
                if op_parent_name not in one_module_names_idx_dict.keys():
                    continue
                parent_idx = one_module_names_idx_dict[op_parent_name]
                if device == CPU and (GPU, parent_idx) not in s_dict.keys():
                    continue
                elif device == GPU and (CPU, parent_idx) not in s_dict.keys():
                    continue
                parent_output_tensors_addr = [paddr for (paddr, _) in op_parent.output_tensors]
                if addr in parent_output_tensors_addr:
                    acc_data_trans_latency += data_trans_latency
                    break
        
        result.append((op_name, device, start_time, device_latency, acc_data_trans_latency))
        op_execution_order_list.append((op_name, device, start_time))
    # Get the schedule result from t_device_op
    result = sorted(result, key=lambda x: x[2])
    op_execution_order_list = sorted(op_execution_order_list, key=lambda x: x[2])
    for t in op_execution_order_list:
        logger.info(t)
    cpu_data, gpu_data, convert_data, convert_device = [], [], [], []
    for (op_name, device, start_time, device_latency, acc_data_trans_latency) in result:
        logger.info("name:{}, device:{}, start_time:{}, latency:{}, data_trans:{}"\
            .format(op_name, device, start_time, device_latency, acc_data_trans_latency))
        if device == 1:
            cpu_data.append((start_time, device_latency))
        elif device == 2:
            gpu_data.append((start_time, device_latency))
        if acc_data_trans_latency > 0:
            convert_data.append((start_time+device_latency, acc_data_trans_latency))
            convert_device.append(device)
    
    return cpu_data, gpu_data, convert_data, convert_device, op_execution_order_list



def sum_lp_objectives(folder_path):
    grep_cmd = "cat %s | grep Objective | awk '{print $4}'" % os.path.join(folder_path, "lp-result-subgraphs-*")
    print("Execute "+grep_cmd)
    lp_result = os.popen(grep_cmd).read()
    print(lp_result)
    com = lp_result.split("\n")
    total = 0.
    for c in com:
        if c == '' or len(c) == 0:
            continue
        total += float(c)
    return total


def check_device_placement_equal(glpk_name_device_tuple_list, cplex_name_device_tuple_list):
    glpk_dict = {}
    for name, device in glpk_name_device_tuple_list:
        glpk_dict[name] = device
    equal = True
    for name, device in cplex_name_device_tuple_list:
        if glpk_dict[name] != device:
            logger.info("Not equal: {} {} {}".format(name, glpk_dict[name], device))
            equal = False
    return equal
