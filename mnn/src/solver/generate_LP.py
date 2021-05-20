import queue
import os
import pysnooper
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger()


from profile import read_profile_data
from profile import graph_partition
from profile import find_critical_node
from profile import subgraph
from utils import utils
from visualization import *
from solver import parse_lp_solution
from solver import uprank_partitioning


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

class LPMode:
    Mode_Subgraph = 0
    Mode_Operator = 1
    Mode_AUTO_Subgraph = 2
    Mode_Model_Operator = 3


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
def get_parent_idxes_and_data_trans(op_name, one_module_names_idx_dict, op_dict, device, mode=LPMode.Mode_Subgraph):
    op = op_dict[op_name]
    assert(device in [CPU, GPU])
    # For all 
    acc_data_trans_latency = 0.0
    parent_idx_data_trans = []
    for (addr, data_trans) in op.op_def.operator_latency.input_data_trans_latency.items():
        data_trans_latency = data_trans[device-1]
        for op_parent_name in op.parents:
            op_parent = op_dict[op_parent_name]
            if mode==LPMode.Mode_Subgraph and not isinstance(op_parent, subgraph.Subgraph):
                continue
            if op_parent_name not in one_module_names_idx_dict.keys():
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
    for (addr, data_trans) in op.op_def.operator_latency.input_data_trans_latency.items():
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
            device_latency = op.op_def.operator_latency.CPU_latency
        elif device == GPU:
            device_latency = op.op_def.operator_latency.GPU_latency
        
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
            op_parent_latency = op_dict[op_name_parent].op_def.operator_latency
            
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
    b_cpu_latency = op_dict[op_name_b].op_def.operator_latency.CPU_latency
    # a_cpu_latency = op_dict[op_name_a].op_def.operator_latency.CPU_latency
    b_gpu_latency = op_dict[op_name_b].op_def.operator_latency.GPU_latency
    # a_gpu_latency = op_dict[op_name_a].op_def.operator_latency.GPU_latency
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
        op_profile_list.append((op_name, idx, op.op_def.operator_latency))
    op_profile_list = sorted(op_profile_list,
                             key=lambda op_profile: op_profile[1])

    for op_profile in op_profile_list:
        (op_name, idx, op.op_def.operator_latency) = op_profile
        print("%s %d %s" % (op_name, idx, op.op_def.operator_latency))


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



def run_glpsol(lp_file_path, result_file_path):
    glpsol_file_path = "glpsol"
    cmd_str = '%s --lp %s -o %s' % (glpsol_file_path, lp_file_path, result_file_path)
    logger.info("Execute %s" % (cmd_str))
    os.system(cmd_str)
    logger.info("Run glpk solver done!")

# ./cplex -c "read /Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/inception-v4/redmi/subgraphs-Mixed_7b.lp" "mipopt" "write tmp.txt" "sol" "y" "quit"
# TODO(xcw) Add support for IBM CPLEX solver
def run_cplex(lp_file_path, result_file_path):
    cplex_file_path = "cplex"
    cmd_str = '{} -c "read {}" "mipopt" "write {}" "sol" "y" "quit"' \
        .format(cplex_file_path, lp_file_path, result_file_path)
    logger.info("Execute %s" % (cmd_str))
    os.system(cmd_str)
    logger.info("Run cplex solver done!")


# The result file are in follows:
# '4 s_1_6        *              1             0             1 '

# @pysnooper.snoop()
def compute_data_trans_intersection(cpu_data, gpu_data, convert_data, convert_device):
    def add_latency(data):
        return [(start, start+latency) for (start, latency) in data]
    def max_time(data):
        if len(data) == 0:
            return 0
        else:
            return max([endtime for (_, endtime) in data])
    cpu_data = add_latency(cpu_data)
    gpu_data = add_latency(gpu_data)
    convert_data = add_latency(convert_data)
    print(cpu_data)
    print(gpu_data)
    print(convert_data)
    sum_of_intersection = 0
    for (cs, ce), device in list(zip(convert_data, convert_device)):
        for (gs, ge) in gpu_data:
            # if op execute on GPU, skip 
            if device == 2:
                continue
            if cs >= gs and cs <= ge:
                sum_of_intersection += (min(ce, ge) - cs)
            elif gs > cs and gs < ce:
                sum_of_intersection += (min(ce, ge) - gs)
    cpu_max, gpu_max, convert_max = max_time(cpu_data), max_time(gpu_data), max_time(convert_data)
    endpoint = max([cpu_max, gpu_max, convert_max])

    return endpoint, sum_of_intersection


def solve_glpk(op_name_list, name_op_dict, net_def, module_name_list, folder_path, model_name, mode=LPMode.Mode_Subgraph):
    folder_path = os.path.join(folder_path, "ilp")
    if not os.path.exists(folder_path):
        os.system("mkdir {}".format(folder_path))
    lines = []
    intersection_list = []
    if mode == LPMode.Mode_AUTO_Subgraph:
        subgraph_list = uprank_partitioning.uprank_partitioning(op_name_list, name_op_dict)
        # we partition the graph from bottom to up
        subgraph_list.reverse()
        module_name_list.clear()
        for idx in range(len(subgraph_list)):
            module_name_list.append("subgraph-{}".format(idx))
        print("Uprank partitioning result:")
        for sb_name in subgraph_list:
            print("{} {}".format(len(sb_name), sb_name))
        sum = 0
        for g in subgraph_list:
            sum += len(g)
        print(sum)
        assert(sum == len(op_name_list))
    exit(0)
    module_idx = 0
    for module_name in module_name_list:
        one_module_names_idx_dict = {}
        if mode == LPMode.Mode_Subgraph:
            # For one module with multiple subgraphs, we need build subgraph and update the op_dict
            parent_subgraph = subgraph.Subgraph(module_name)
            if model_name != None and model_name.find("inception") >=0:
                one_module_name, branches = subgraph.get_inception_one_module_name(module_name, op_name_list)
                parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, branches, pattern=module_name)
            elif model_name != None and (model_name.find("pnasnet") >=0 or model_name.find("nasnet") >=0):
                parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, subgraph.pnasnet_mobile_subgraph_subprefix(), pattern=module_name)
            # one_module_names_idx_dict = associate_op_name_with_idx(
            #     "/mnt/d/home/Projects/DAG-scheduler/mnn/pnasnet-mobile/pnasnet-cell-0-subgraph-names.txt")
            logger.info("aaa {}".format(parent_subgraph.op_name_list))
            for subgraph_name in parent_subgraph.op_name_list:
                logger.info(name_op_dict[subgraph_name])
            one_module_names_idx_dict = associate_op_name_list_with_idx(parent_subgraph.op_name_list)
        elif mode == LPMode.Mode_Operator:
            one_module_names_idx_dict = associate_op_name_list_with_idx(subgraph.filter_op_name_list(op_name_list, module_name))
        elif mode == LPMode.Mode_AUTO_Subgraph:
            one_module_names_idx_dict = associate_op_name_list_with_idx(subgraph_list[module_idx])
            module_idx += 1
        elif mode == LPMode.Mode_Model_Operator:
            one_module_names_idx_dict = associate_op_name_list_with_idx(op_name_list)
        # Generate LP constraints and write them to a file
        
        LP_contents = generateLP(one_module_names_idx_dict, op_name_list, name_op_dict, net_def)

        module_name_striped = module_name.replace('/','-')
        if module_name_striped[-1] == '-':
            module_name_striped = module_name_striped[0:len(module_name_striped)-1]
            module_name_striped = module_name_striped.split('-')[-1]
        
        # Move ILP result to a subdirectory
        lp_file_path = os.path.join(folder_path, "subgraphs-" + module_name_striped + ".lp")
        result_file_path = os.path.join(folder_path, "lp-result-subgraphs-" + module_name_striped+ ".txt")
        logger.info("Write Integer Linear Programming models to {}".format(lp_file_path))
        utils.write_lines(lp_file_path, LP_contents)
        # Solve the LP
        # run_glpsol(lp_file_path, result_file_path)
        run_cplex(lp_file_path, result_file_path+".xml")
        # Parse subgraph device placement result
        cplex_name_device_tuple_list = parse_lp_solution.parse_cplex_result(one_module_names_idx_dict, result_file_path+".xml")
        # glpk_name_device_tuple_list = parse_lp_solution.parse_glpk_result(one_module_names_idx_dict, result_file_path)
        # equal = parse_lp_solution.check_device_placement_equal(glpk_name_device_tuple_list, cplex_name_device_tuple_list)
        # if not equal:
        #     logger.info("ILP result not equal\n")
        name_device_tuple_list = cplex_name_device_tuple_list
        
        logger.info(name_device_tuple_list)
        cpu_data, gpu_data, convert_data, convert_device, \
                op_execution_order_list = \
                    parse_lp_solution.parse_ilp_timeline(one_module_names_idx_dict, result_file_path, name_op_dict, mode=mode)
        endpoint, sum_of_intersection = compute_data_trans_intersection(cpu_data, gpu_data, convert_data, convert_device)
        intersection_list.append((endpoint, sum_of_intersection))
        print("endpoint: %f , intersection: %f" % (endpoint, sum_of_intersection))
        print("module_name")
        
        tmp_module_name_list = []
        for c in module_name.split("/"):
            if len(c.strip()) > 0:
                tmp_module_name_list.append(c)
        gantt_file_path = os.path.join(folder_path, tmp_module_name_list[-1])
        draw_gantt(cpu_data, gpu_data, convert_data, gantt_file_path)
        logger.info("open {}".format(gantt_file_path))
        # os.system("open {}.pdf".format(gantt_file_path))
        # os.system("sleep 1")
        # exit(0)
        # device_placement_file_path = os.path.join(folder_path, "mDeviceMap-"+ "subgraphs-" + model_name + "-" + module_name_striped +".txt")
        results = subgraph.write_subgraph_device_placement_result([name for (name, device, start_time) in op_execution_order_list if device == CPU],\
            [name for (name, device, start_time) in op_execution_order_list if device == GPU], \
            name_op_dict, op_execution_order_list=op_execution_order_list)
        lines.extend(results)
        # print("Write result to %s" % (device_placement_file_path))
    return lines, intersection_list
