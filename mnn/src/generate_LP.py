import queue
import os

import read_profile_data
from read_net_structure import *
from utils import *
from draw_gantt import *

# DO NOT MODIFY THE VALUE OF `CPU` AND `GPU`
CPU = 1
GPU = 2

# `M` and `K` can be changed based on the latency of the subgraph
M = 1000
K = 2000
# GPU has to do the data transformation for CPU
# there for the GPU execution time also increases
# we use a scale factor to simulate the GPU execution time increasing
GPU_TRANSFORM_SCALE_FACTOR = 1

# Read one module names and associate a name with one index
def associate_op_name_with_idx(file_path):
    f = open(file_path)
    one_module_names_idx_dict = {}
    idx = 1
    for line in f.readlines():
        one_module_names_idx_dict[line.strip()] = idx
        idx += 1
    return one_module_names_idx_dict


def associate_op_name_list_with_idx(op_name_list):
    idx = 1
    one_module_names_idx_dict = {}
    for op_name in op_name_list:
        one_module_names_idx_dict[op_name] = idx
        idx += 1
    return one_module_names_idx_dict


# Used to generate (s[y][i]*c[u][j][i] for all i belongs to j's parents)
def get_parent_idxes_and_data_trans(op_name, one_module_names_idx_dict, op_dict, device):
    op = op_dict[op_name]
    assert(device in [CPU, GPU])
    # For all 
    acc_data_trans_latency = 0.0
    parent_idx_data_trans = []
    for (addr, data_trans) in op.op_def.operatorLatency.input_data_trans_latency.items():
        data_trans_latency = data_trans[device-1]
        for op_parent_name in op.parents:
            op_parent = op_dict[op_parent_name]
            if not isinstance(op_parent, Subgraph):
                continue
            parent_idx = one_module_names_idx_dict[op_parent_name]
            parent_output_tensors_addr = [paddr for (paddr, _) in op_parent.output_tensors]
            if addr in parent_output_tensors_addr:
                acc_data_trans_latency += data_trans_latency
                parent_idx_data_trans.append((parent_idx, data_trans_latency))
    return (acc_data_trans_latency, parent_idx_data_trans)


def get_input_tensor_data_trans_latency(op_name, op_dict, device):
    op = op_dict[op_name]
    acc_data_trans_latency = 0.0
    assert(device in [CPU, GPU])
    for (addr, data_trans) in op.op_def.operatorLatency.input_data_trans_latency.items():
        acc_data_trans_latency += data_trans[device]
    return acc_data_trans_latency


# Generate constraints for the "tt > node finish time"
def generate_final_latency_for_one_node(op_name, one_module_names_idx_dict,
                                        device_list, op_dict):
    constraints = []
    op = op_dict[op_name]
    idx = one_module_names_idx_dict[op_name]
    
    for device in device_list:
        assert(device in [CPU, GPU])
        (acc_data_trans_latency, parent_idx_data_trans) = get_parent_idxes_and_data_trans(op_name, one_module_names_idx_dict, op_dict, device)    
        c1 = "tt + %d s_%d_%d > 0.0\n" % (M, device, idx)
        c2 = ""

        if device == CPU:
            device_latency = op.op_def.operatorLatency.CPU_latency
        elif device == GPU:
            device_latency = op.op_def.operatorLatency.GPU_latency
        
        lp_data_trans = ""
        for (parent_idx, data_trans_latency) in parent_idx_data_trans:
            lp_data_trans += " + %f s_%d_%d " % (data_trans_latency, device, parent_idx)
        c2 = "tt - t_%d_%d - %f s_%d_%d %s > %f\n" \
            % (device, idx, M + acc_data_trans_latency, device, idx, lp_data_trans, (device_latency - M))
        constraints.extend([c1, c2])
    return constraints


# One node can only be executed once, so the sum of s_i_j is equal to 1
def generate_node_execute_once(one_module_names_idx_dict, device_list):
    constraints = []
    for _, idx in one_module_names_idx_dict.items():
        s = ""
        for device in device_list:
            if device == device_list[-1]:
                s += ("s_%d_%d = 1\n" % (device, idx))
            else:
                s += ("s_%d_%d + " % (device, idx))
        constraints.append(s)
    return constraints


# Find if op_name_b is op_name_a's ancestor
# if not ancestor, a and b can not execute on device at the same time
# else, use DAG constrains
def have_relative_relation(one_module_names_idx_dict, op_name_a, op_name_b,
                           op_dict):
    op_queue = queue.Queue()
    for parent in op_dict[op_name_a].parents:
        if parent in one_module_names_idx_dict.keys():
            op_queue.put(parent)
    while not op_queue.empty():
        op_parent_name = op_queue.get()
        if op_parent_name == op_name_b:
            return True
        op_parent = op_dict[op_parent_name]
        for parent in op_parent.parents:
            if parent in one_module_names_idx_dict.keys():
                op_queue.put(parent)
    return False


# Find if op_name_b is op_name_a's parent
def have_parent_relation(op_name_a, op_name_b, op_dict):
    op_a = op_dict[op_name_a]
    if op_name_b in op_a.parents:
        return True
    else:
        return False


# Generate constraints for parent and child operations.
# Parent and child constraints can both cover the DAG constrains and the device execute one device at a time constrains
# We have considered the format convert
def generate_parent_and_child_constraints(one_module_names_idx_dict,
                                          device_list, op_name_child,
                                          op_name_parent, op_dict):
    idx_parent = one_module_names_idx_dict[op_name_parent]
    idx_child = one_module_names_idx_dict[op_name_child]
    # idx_parent_parent = 0
    constraints = []
    for device1 in device_list:
        for device2 in device_list:
            c = ""
            op_parent_latency = op_dict[op_name_parent].op_def.operatorLatency
            
            device_latency = 0.0
            if device2 == CPU:
                device_latency = op_parent_latency.CPU_latency
            elif device2 == GPU:
                device_latency = op_parent_latency.GPU_latency
            (acc_data_trans_latency, parent_idx_data_trans) = get_parent_idxes_and_data_trans(op_name_parent, one_module_names_idx_dict, op_dict, device2)
            lp_data_trans = ""
            for (parent_idx, data_trans_latency) in parent_idx_data_trans:
                lp_data_trans += " + %f s_%d_%d " % (data_trans_latency, device2, parent_idx)
            c = "t_%d_%d - t_%d_%d - %f s_%d_%d %s > %f\n" \
                % (device1, idx_child, device2, idx_parent, (M + acc_data_trans_latency), device2, idx_parent, \
                    lp_data_trans, device_latency - M)
            
            constraints.append(c)
    return constraints


def get_parent_idxes(one_module_names_idx_dict, op_name, op_dict):
    idx_parent = []
    for parent in op_dict[op_name].parents:
        if parent in one_module_names_idx_dict.keys():
            idx_parent.append(one_module_names_idx_dict[parent])
    return idx_parent


def get_parent_idx(one_module_names_idx_dict, op_name, op_dict):
    idx_parent = 0
    for parent in op_dict[op_name].parents:
        if parent in one_module_names_idx_dict.keys():
            idx_parent = one_module_names_idx_dict[parent]
    return idx_parent


def generate_one_node_at_a_device(one_module_names_idx_dict, op_name_a,
                                  op_name_b, device_list, op_dict):
    idx_a = one_module_names_idx_dict[op_name_a]
    idx_b = one_module_names_idx_dict[op_name_b]
    # idx_a_parent = get_parent_idx(one_module_names_idx_dict, op_name_a,
    #                               op_dict)
    idx_b_parent = get_parent_idx(one_module_names_idx_dict, op_name_b,
                                  op_dict)
    b_cpu_latency = op_dict[op_name_b].op_def.operatorLatency.CPU_latency
    # a_cpu_latency = op_dict[op_name_a].op_def.operatorLatency.CPU_latency
    b_gpu_latency = op_dict[op_name_b].op_def.operatorLatency.GPU_latency
    # a_gpu_latency = op_dict[op_name_a].op_def.operatorLatency.GPU_latency
    constraints = []
    u_variable = []

    for device in device_list:
        c = ""
        li = [idx_a, idx_b]
        li.sort()
        [u_idx_a, u_idx_b] = li
        u_val_str = "u_%d_%d_%d" % (device, u_idx_a, u_idx_b)
        cond_val_str = "r_%d_%d_%d" % (device, u_idx_a, u_idx_b)
        assert(idx_a != idx_b)
        if u_val_str not in u_variable:
            u_variable.append(u_val_str+"\n")
            u_variable.append(cond_val_str+"\n")
        # Judge whether the two op are executed on the same device
        # 0 <= s1 + s2 -2 + Kx <= K - 1
        # from: https://blog.adamfurmanek.pl/2015/09/12/ilp-part-4/
        judge_constraint1 = "s_%d_%d + s_%d_%d + %d %s >= 2\n" % (device, idx_a, device, idx_b, M, cond_val_str)
        judge_constraint2 = "- s_%d_%d - s_%d_%d - %d %s >= %d\n" % (device, idx_a, device, idx_b, M, cond_val_str, - M - 1)

        device_latency = 0.0
        if device == CPU:
            device_latency = b_cpu_latency
        elif device ==GPU:
            device_latency = b_gpu_latency
        
        (acc_data_trans_latency, parent_idx_data_trans) = get_parent_idxes_and_data_trans(op_name_b, one_module_names_idx_dict, op_dict, device)
        lp_data_trans = ""
        for (parent_idx, data_trans_latency) in parent_idx_data_trans:
            lp_data_trans += " + %f s_%d_%d " % (data_trans_latency, device, parent_idx)
        
        if idx_a > idx_b:
            c = "t_%d_%d - t_%d_%d + %f %s - %f s_%d_%d %s + %f %s > %f\n" \
                % (device, idx_a, device, idx_b, M, u_val_str, \
                acc_data_trans_latency, device, idx_b, lp_data_trans, K, cond_val_str, device_latency)
        else:
            c = "t_%d_%d - t_%d_%d - %f %s - %f s_%d_%d %s + %f %s > %f\n" \
                 % (device, idx_a, device, idx_b, M, u_val_str, \
                    acc_data_trans_latency, device, idx_b, lp_data_trans, K, cond_val_str, device_latency - M)
        constraints.extend([judge_constraint1, judge_constraint2, c])

    return constraints, u_variable


# One device can only execute one op at a time
# If op_a is op_b's child, then we can simply use the DAG constraints
def generate_device_execute_once_at_a_time(one_module_names_idx_dict,
                                           device_list, op_dict):
    constraints = []
    u_variables = []
    for op_name_a, idx_a in one_module_names_idx_dict.items():
        for op_name_b, idx_b in one_module_names_idx_dict.items():
            if op_name_a == op_name_b:
                continue
            # Generate DAG constraints
            if have_parent_relation(op_name_a, op_name_b, op_dict):
                constraints.extend( \
                    generate_parent_and_child_constraints(one_module_names_idx_dict, device_list, op_name_a, op_name_b, op_dict))
            elif have_parent_relation(op_name_b, op_name_a, op_dict):
                constraints.extend( \
                    generate_parent_and_child_constraints(one_module_names_idx_dict, device_list, op_name_b, op_name_a, op_dict))
            # Generate one device constraints
            if not have_relative_relation(one_module_names_idx_dict, op_name_a, op_name_b, op_dict) and \
                not have_relative_relation(one_module_names_idx_dict, op_name_b, op_name_a, op_dict):
                constraints_one_device, u_variable_one_device =  generate_one_node_at_a_device( \
                        one_module_names_idx_dict, op_name_a, op_name_b, device_list, op_dict)
                constraints.extend(constraints_one_device)
                u_variables.extend(u_variable_one_device)
    return constraints, u_variables


def generate_concat():
    pass


def generate_binary(one_module_names_idx_dict, device_list):
    binary_content = ["Binary\n"]
    for op_name, idx in one_module_names_idx_dict.items():
        for device in device_list:
            c = "s_%d_%d\n" % (device, idx)
            binary_content.append(c)
    return binary_content


def print_op_profile(one_module_names_idx_dict, op_dict):
    op_profile_list = []
    for op_name, idx in one_module_names_idx_dict.items():
        op = op_dict[op_name]
        op_profile_list.append((op_name, idx, op.op_def.operatorLatency))
    op_profile_list = sorted(op_profile_list,
                             key=lambda op_profile: op_profile[1])

    for op_profile in op_profile_list:
        (op_name, idx, op.op_def.operatorLatency) = op_profile
        print("%s %d %s" % (op_name, idx, op.op_def.operatorLatency))


def generateLP(one_module_names_idx_dict, op_name_list, op_dict, net_def):

    # Print the relationship between op_name and index
    print_op_profile(one_module_names_idx_dict, op_dict)

    # Set s_cpu_%d of 'concat' to 1
    concat_constraint = ""
    for op_name, idx in one_module_names_idx_dict.items():
        if op_name.strip().split("/")[-1] == "concat":
            concat_constraint = ("s_%d_%d = 1\n" % (CPU, idx))
    LP_contents = []
    LP_objective = "Minimize\n\tvalue: tt\n\n"
    LP_constraints = ["Subject to\n"]
    LP_constraints.append(concat_constraint)
    # Generate for all op one all devices
    # 1 for CPU, 2 for GPU
    device_list = [CPU, GPU]
    for op_name, idx in one_module_names_idx_dict.items():
        LP_constraints.extend(
            generate_final_latency_for_one_node(op_name,
                                                one_module_names_idx_dict,
                                                device_list, op_dict))

    LP_constraints.extend(
        generate_node_execute_once(one_module_names_idx_dict, device_list))
    device_once_at_a_time, u_variable = generate_device_execute_once_at_a_time(
        one_module_names_idx_dict, device_list, op_dict)
    LP_constraints.extend(device_once_at_a_time)

    binary_content = generate_binary(one_module_names_idx_dict, device_list)
    binary_content.extend(u_variable)
    # Remove the dulplicate variables
    binary_content = sorted(set(binary_content), key=binary_content.index)
    LP_constraints = sorted(set(LP_constraints), key=LP_constraints.index)

    LP_contents.extend(LP_objective)
    LP_contents.extend(LP_constraints)
    LP_contents.extend(binary_content)
    LP_contents.append("\nEnd\n")
    
    return LP_contents


def write_LP_contents(LP_contents, file_name):
    f = open(file_name, "w")
    f.writelines(LP_contents)
    f.flush()
    f.close()


def run_glpsol(lp_file_path, result_file_path):
    glpsol_file_path = "glpsol"
    cmd_str = '%s --lp %s -o %s' % (glpsol_file_path, lp_file_path, result_file_path)
    print("Execute %s" % (cmd_str))
    os.system(cmd_str)
    print("Run solver done!")


# The result file are in follows:
# '4 s_1_6        *              1             0             1 '
def parse_glpk_result(one_module_names_idx_dict, result_file_path):
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
                sc = s.split('_')
                if int(sc[1]) == CPU:
                    device = 0
                elif int(sc[1]) == GPU:
                    device = 3
                op_idx = sc[2]
                for name, idx in one_module_names_idx_dict.items():
                    if int(op_idx) == idx:
                        name_device_tuple_list.append((name, device))
                        break
    for (name, device) in name_device_tuple_list:
        print("%s %d" %(name, device))
    return name_device_tuple_list


def parse_glpk_timeline(one_module_names_idx_dict, result_file_path, op_dict):
    idx_name_dict = {v:k for k,v in one_module_names_idx_dict.items()}
    print(idx_name_dict)
    s_dict = {}
    t_dict = {}
    f = open(result_file_path, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.find('s_') >=0 or line.find('t_') >=0:
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
    result = []
    op_execution_order_list = []
    for (device, idx), _ in s_dict.items():
        start_time = t_dict[(device, idx)]
        op_name = idx_name_dict[idx]
        op = op_dict[op_name]
        device_latency = 0.0
        if device == CPU:
            device_latency = op.op_def.operatorLatency.CPU_latency
        elif device == GPU:
            device_latency = op.op_def.operatorLatency.GPU_latency
        # Compute data trans latency
        acc_data_trans_latency = 0.0
        for (addr, data_trans) in op.op_def.operatorLatency.input_data_trans_latency.items():
            data_trans_latency = data_trans[device-1]
            for op_parent_name in op.parents:
                op_parent = op_dict[op_parent_name]
                if not isinstance(op_parent, Subgraph):
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
    
    result = sorted(result, key=lambda x: x[2])
    op_execution_order_list = sorted(op_execution_order_list, key=lambda x: x[2])
    for t in op_execution_order_list:
        print(t)
    cpu_data, gpu_data, convert_data = [], [], []
    for (op_name, device, start_time, device_latency, acc_data_trans_latency) in result:
        print("name:{}, device:{}, start_time:{}, latency:{}, data_trans:{}"\
            .format(op_name, device, start_time, device_latency, acc_data_trans_latency))
        if device == 1:
            cpu_data.append((start_time, device_latency))
        elif device == 2:
            gpu_data.append((start_time, device_latency))
        if acc_data_trans_latency >0 :
            convert_data.append((start_time+device_latency, acc_data_trans_latency))
    
    return cpu_data, gpu_data, convert_data, op_execution_order_list



def get_inception_one_module_name(op_name_prefix, op_name_list):
    module_op_names = []
    branches = set()
    for op_name in op_name_list:
        if op_name.find(op_name_prefix) == 0:
            module_op_names.append(op_name)
            if op_name.find("Branch") > 0:
                branches.add(op_name.split("/")[3])
    
    return module_op_names, list(branches)


def solve_glpk(op_name_list, name_op_dict, net_def, module_name_list, folder_path, model_name):
    lines = []
    for module_name in module_name_list:
        # For one module with multiple subgraphs, we need build subgraph and update the op_dict
        parent_subgraph = Subgraph(module_name)
        if model_name != None and model_name.find("inception") >=0:
            one_module_name, branches = get_inception_one_module_name(module_name, op_name_list)
            parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, branches, pattern=module_name)
        elif model_name != None and (model_name.find("pnasnet") >=0 or model_name.find("nasnet") >=0):
            parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, pnasnet_mobile_subgraph_subprefix(), pattern=module_name)
        # one_module_names_idx_dict = associate_op_name_with_idx(
        #     "/mnt/d/home/Projects/DAG-scheduler/mnn/pnasnet-mobile/pnasnet-cell-0-subgraph-names.txt")
        one_module_names_idx_dict = associate_op_name_list_with_idx(parent_subgraph.op_name_list)
        # Generate LP constraints and write them to a file
        LP_contents = generateLP(one_module_names_idx_dict, op_name_list, name_op_dict, net_def)
        # write_LP_contents(LP_contents, "inception-one-module-mix5c.lp")
        module_name_striped = module_name.replace('/','-')
        if module_name_striped[-1] == '-':
            module_name_striped = module_name_striped[0:len(module_name_striped)-1]
            module_name_striped = module_name_striped.split('-')[-1]
        lp_file_path = os.path.join(folder_path, "subgraphs-" + module_name_striped + ".lp")
        result_file_path = os.path.join(folder_path, "lp-result-subgraphs-" + module_name_striped+ ".txt")
        write_LP_contents(LP_contents, lp_file_path)
        # Solve the LP
        run_glpsol(lp_file_path, result_file_path)
        # Parse subgraph device placement result
        # name_device_tuple_list = parse_glpk_result(one_module_names_idx_dict, result_file_path)
        # print(name_device_tuple_list)
        cpu_data, gpu_data, convert_data, op_execution_order_list = parse_glpk_timeline(one_module_names_idx_dict, result_file_path, name_op_dict)
        print("module_name")

        tmp_module_name_list = []
        for c in module_name.split("/"):
            if len(c.strip()) > 0:
                tmp_module_name_list.append(c)
        
        draw_gantt(cpu_data, gpu_data, convert_data, os.path.join(folder_path, tmp_module_name_list[-1]))
            
        device_placement_file_path = os.path.join(folder_path, "mDeviceMap-"+ "subgraphs-" + model_name + "-" + module_name_striped +".txt")
        results = write_subgraph_device_placement_result([name for (name, device, start_time) in op_execution_order_list if device == CPU],\
            [name for (name, device, start_time) in op_execution_order_list if device == GPU], \
            name_op_dict, device_placement_file_path, op_execution_order_list=op_execution_order_list)
        lines.extend(results)
        # print("Write result to %s" % (device_placement_file_path))
    return lines


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



def insert_untreated_ops(lines, op_name_list, name_op_dict, unsupported_op_names=[]):
    solved_op_name_list = []
    for l in lines:
        solved_op_name_list.append(l.split(' ')[0])
    print(len(lines))
    print(len(op_name_list))
    
    untreated_op_name_list = set(op_name_list).difference(set(solved_op_name_list))
    untreated_op_latency = 0.0
    # print(untreated_op_name_list)
    for op_name in op_name_list:
        if op_name in untreated_op_name_list:
            op = name_op_dict[op_name]
            parents_set = set(op.parents)
            if len(op.parents) == 0:
                lines.insert(0, "%s %d\n" % (op_name, 0))
            else:
                index = 0
                for line in lines:
                    index += 1
                    op_tmp_name = line.strip().split(" ")[0]
                    if op_tmp_name in parents_set:
                        parents_set.remove(op_tmp_name)
                        if len(parents_set) == 0:
                            cpu_latency = op.op_def.operatorLatency.CPU_latency
                            gpu_latency = op.op_def.operatorLatency.CPU_latency
                            if op_name not in unsupported_op_names and gpu_latency < cpu_latency:
                                lines.insert(index, "%s %d\n" % (op_name, 3))
                                untreated_op_latency += gpu_latency
                            else:
                                lines.insert(index, "%s %d\n" % (op_name, 0))
                                untreated_op_latency += cpu_latency
                            print(index, "%s %d\n" % (op_name, 0))
                            break
    return lines, untreated_op_latency


def solve_pnasnet(model, mobile, thread):
    # Read profile data
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict, net_def = gather_model_profile(
        os.path.join(model_dir, model + "-info.txt"),
        os.path.join(model_dir, mobile, model+'-'+mobile+'-data-trans.csv'),
        os.path.join(model_dir, mobile, mobile+"-"+model+"-layerwise-latency.csv"), thread, SCALE=1.0)
    
    # Using module prefix to form the subgraph
    pnasnet_module_list = ['cell_stem_0/', 'cell_stem_1/']
    if model == 'pnasnet-large':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
    elif model == 'pnasnet-mobile':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(9)])
    elif model == 'nasnet-large':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(18)])
        pnasnet_module_list.insert(6, 'reduction_cell_0/')
        pnasnet_module_list.insert(13, 'reduction_cell_1/')
        
    elif model == 'nasnet-mobile':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
        pnasnet_module_list.insert(6, 'reduction_cell_0/')
        pnasnet_module_list.insert(11, 'reduction_cell_1/')
    else:
        print("Model %s does not suport yet." % (model))
        return
    
    # Using GLPK solve device placement here
    folder_path = os.path.join(model_dir, mobile)
    lines = solve_glpk(op_name_list, name_op_dict, net_def, pnasnet_module_list, folder_path, model)
    unsupported_op_names = ["final_layer/Relu", "final_layer/Mean/reduction_indices", \
        "final_layer/Relu___tr4final_layer/Mean", "final_layer/Mean", \
        "final_layer/FC/weights", "final_layer/FC/MatMul", \
        "final_layer/FC/biases", "final_layer/FC/BiasAdd", "final_layer/predictions"]
    # Deal with ops that are not in the module prefix
    lines, untreated_op_latency = insert_untreated_ops(lines, op_name_list, name_op_dict, unsupported_op_names=unsupported_op_names)
    lp_total = sum_lp_objectives(folder_path)
    
    # Write results
    device_map_file_path = os.path.join(model_dir, mobile, "mDeviceMap-{}-cpu-{}.txt".format(model, thread))
    f = open(device_map_file_path, 'w')
    f.writelines(lines)
    f.flush()
    f.close()
    print(untreated_op_latency)
    print("LP+serial total: {}".format(lp_total+untreated_op_latency))


def solve_inception(model, mobile, thread):
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict, net_def = gather_model_profile(
        os.path.join(model_dir, model + "-info.txt"),
        os.path.join(model_dir, mobile, model+'-'+mobile+'-data-trans.csv'),
        os.path.join(model_dir, mobile, mobile+"-"+model+"-layerwise-latency.csv"),
        thread, SCALE=1.0)
    
    if model == "inception-v3":
        inception_prefix = "InceptionV3/InceptionV3/"
        inception_module_list = ["Mixed_5b/", "Mixed_5c/", "Mixed_5d/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/", "Mixed_7a/", "Mixed_7b/", "Mixed_7c/"]
    elif model == "inception-v4":
        inception_prefix = "InceptionV4/InceptionV4/"
        inception_module_list = ["Mixed_4a/", "Mixed_5b/", "Mixed_5c/", "Mixed_5d/", "Mixed_5e/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/","Mixed_6f/","Mixed_6g/","Mixed_6h/",\
            "Mixed_7a/","Mixed_7b/","Mixed_7c/","Mixed_7d/",]
    inception_module_list = [inception_prefix + module for module in inception_module_list]
    folder_path = os.path.join(model_dir, mobile)
    lines = solve_glpk(op_name_list, name_op_dict, net_def, inception_module_list, folder_path, model)
    lines, untreated_op_latency = insert_untreated_ops(lines, op_name_list, name_op_dict)
    lp_total = sum_lp_objectives(folder_path)
    print("LP+serial total: {}".format(lp_total+untreated_op_latency))

    device_map_file_path = os.path.join(model_dir, mobile, "mDeviceMap-{}-cpu-{}.txt".format(model, thread))
    f = open(device_map_file_path, 'w')
    f.writelines(lines)
    f.flush()
    f.close()



if __name__ == "__main__":    
    model, mobile, thread = parse_model_mobile()
    if model in ['pnasnet-mobile', 'pnasnet-large', 'nasnet-large', 'nasnet-mobile']:
        solve_pnasnet(model, mobile, thread)
    elif model in ['inception-v3', 'inception-v4']:
        solve_inception(model, mobile, thread)
    