
from profile import net_struct
from profile import read_profile_data
from utils import utils

import queue
import re
import os

def pnasnet_mobile_subgraph_subprefix():
    subprefix_list = []
    for i in range(5):
        subprefix_list.append('comb_iter_%d/' % (i))
    return subprefix_list


def filter_op_name_list(op_name_list, pattern):
    filtered_op_name_list = []
    if pattern != None:
        for op_name in op_name_list:
            if op_name.find(pattern) == 0:
                filtered_op_name_list.append(op_name)
    return filtered_op_name_list


def filter_op_name_not_in_pattern(op_name_list, pattern):
    filtered_op_name_list = []
    if pattern != None:
        for op_name in op_name_list:
            if op_name.find(pattern) != 0:
                filtered_op_name_list.append(op_name)
    return filtered_op_name_list


def put_op_parents_and_children(op_queue, op, untreated_op_name_list):
    for parent_name in op.parents:
        if parent_name in untreated_op_name_list:
            op_queue.put(parent_name)
    for child_name in op.children:
        if child_name in untreated_op_name_list:
            op_queue.put(child_name)


def build_subgraph_relationship(subgraph_list, name_op_dict):
    """Setup relationship for subgraphs
    Add subgraph name into subgraph's parents and children set
    """
    for parent_subgraph_name in subgraph_list:
        for child_subgraph_name in subgraph_list:
            if parent_subgraph_name == child_subgraph_name:
                continue
            parent_subgraph = name_op_dict[parent_subgraph_name]
            child_subgraph = name_op_dict[child_subgraph_name]
            for p in list(child_subgraph.parents):
                if p in parent_subgraph.op_name_list:
                    child_subgraph.parents.add(parent_subgraph.name)
                    parent_subgraph.children.add(child_subgraph.name)
                    break
            for c in list(parent_subgraph.children):
                if c in child_subgraph.op_name_list:
                    parent_subgraph.children.add(child_subgraph.name)
                    child_subgraph.parents.add(parent_subgraph.name)


class Subgraph(net_struct.Operator):

    def __init__(self, name):
        super().__init__(name)
        self.op_name_list = []
        # self.subgraph_list = []
        self.name_op_dict = {}
        self.op_def = net_struct.OperatorDef()
        self.op_def.type = "OpType_Subgraph"

    def merge_subgraph(self, sub):
        self.op_name_list.extend(sub.op_name_list)
        self.name_op_dict.update(sub.name_op_dict)
    
    
    # Set subgraph's parents and children with operators
    def _findGraphInputsOutputs(self):
        internal_parent = set()
        internal_children = set()
        for op_name in self.op_name_list:
            op = self.name_op_dict[op_name]
            internal_parent = internal_parent.union(op.parents)
            internal_children = internal_children.union(op.children)
        common_ops = internal_parent.intersection(internal_children)
        internal_parent.difference_update(common_ops)
        internal_children.difference_update(common_ops)
        self.parents = internal_parent.difference(set(self.op_name_list))
        self.children = internal_children.difference(set(self.op_name_list))
        
        # for parent in self.parents:
        #     op_parent = self.name_op_dict[parent]
        #     for op_name in self.op_name_list:
        #         if op_name in op_parent.children:
        #             op = self.name_op_dict[op_name]
        #             self.input_tensors.extend(op.input_tensors)
        # for child in self.children:
        #     op_child = self.name_op_dict[child]
        #     if op_name in self.op_name_list:
        #         if op_name in op_child.parents:
        #             op = self.name_op_dict[op_name]
        #             self.output_tensors.extend(op.output_tensors)

    
    # Fill this subgraph with operators
    def _set_op_list_with_filter(self, op_name_list, name_op_dict, pattern=None):
        # Filter op, preserve execution order
        self.name_op_dict = name_op_dict
        if pattern != None:
            for op_name in op_name_list:
                if op_name.find(pattern) == 0:
                    self.op_name_list.append(op_name)
                    # self.name_op_dict[op_name] = name_op_dict[op_name]
        else:
            self.op_name_list.extend(op_name_list)
            self.name_op_dict = name_op_dict

    def update(self, name_op_dict):
        self._findGraphInputsOutputs()
        self._summaryLatency(name_op_dict)


    def buildWithOpList(self, op_name_list, name_op_dict, pattern=None):
        # Filter op with pattern, else directly set list
        self._set_op_list_with_filter(op_name_list, name_op_dict, pattern)
        self._findGraphInputsOutputs()
        self._summaryLatency(name_op_dict)

    # Set the subgraph computing latency as the sum of all the internal operators latency
    # Set the subgraph data transformation latency as the sum of all the input operators' data transformation latency
    def _summaryLatency(self, global_name_op_dict):
        self.op_def.operator_latency.CPU_latency = 0
        self.op_def.operator_latency.GPU_latency = 0
        for op_name in self.op_name_list:
            op = self.name_op_dict[op_name]
            self.op_def.operator_latency.CPU_latency += op.op_def.operator_latency.CPU_latency
            self.op_def.operator_latency.GPU_latency += op.op_def.operator_latency.GPU_latency
        # The data transformation latency is the sum of all the operators that are directly connect with
        # the subgraph's parents operator
        input_tensor_set = set()
        output_tensor_set = set()
        parent_output_tensor_set = set()
        children_input_tensor_set = set()
        for op_name in self.op_name_list:
            op = global_name_op_dict[op_name]
            input_tensor_set = input_tensor_set.union(op.input_tensors)
            output_tensor_set = output_tensor_set.union(op.output_tensors)
        for op_name in self.parents:
            op = global_name_op_dict[op_name]
            parent_output_tensor_set = parent_output_tensor_set.union(op.output_tensors)
        for op_name in self.children:
            op = global_name_op_dict[op_name]
            children_input_tensor_set = children_input_tensor_set.union(op.input_tensors)
        
        self.input_tensors = input_tensor_set.intersection(parent_output_tensor_set)
        self.output_tensors = output_tensor_set.intersection(children_input_tensor_set)
        for op_name in self.op_name_list:
            op = global_name_op_dict[op_name]
            for input_tensor in op.input_tensors:
                if input_tensor in self.input_tensors:
                    self.input_nodes.append(op_name)
            for output_tensor in op.output_tensors:
                if output_tensor in self.output_tensors:
                    self.output_nodes.append(op_name)
        
        self.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW = 0.
        self.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC = 0.
        # Remove dulplicate elements
        self.input_node = [i_node for i_node in set(self.input_nodes)]
        for op_name in self.input_nodes:
            op_input = global_name_op_dict[op_name]
            # self.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW += op_input.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW
            # self.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC += op_input.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC
            for (addr, _) in self.input_tensors:
                if addr in op_input.op_def.operator_latency.input_data_trans_latency.keys():
                    self.op_def.operator_latency.input_data_trans_latency[addr] = op_input.op_def.operator_latency.input_data_trans_latency[addr]
        for (addr, latency) in self.op_def.operator_latency.input_data_trans_latency.items():
            self.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW += float(latency[0])
            self.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC += float(latency[1])
        # Debug info
        # print(self.name + "input nodes and output nodes:")
        # print(self.parents)
        # print(self.children)
        # print(self.input_nodes)
        # print(self.input_tensors)
        # print(self.output_nodes)
        # print(self.output_tensors)
        # print(self.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC)
        # print(self.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW)
        # print(self.op_def.operator_latency.input_data_trans_latency)
        # print()


    def buildMultiSubgraph(self, op_name_list, name_op_dict, sub_prefix_list, pattern=None):
        subgraph_list = []
        # Filter op_names with the most outer pattern
        op_name_list = filter_op_name_list(op_name_list, pattern)
        untreated_op_name_list = op_name_list
        # Build subgraph with prefix
        print("In buildMultiSubgraph print pattern %s" % (pattern))
        for subprefix in sub_prefix_list:
            prefix = pattern + subprefix
            untreated_op_name_list = filter_op_name_not_in_pattern(untreated_op_name_list, prefix)
            subgraph = Subgraph(prefix)
            subgraph.buildWithOpList(op_name_list, name_op_dict, pattern=prefix)
            subgraph_list.append(subgraph)
            # print(subgraph)
        # Find op_names that are not been include by the patterns
        # print("Un include op names")
        # print(untreated_op_name_list)
        subgraph_idx = 0
        # print("untreated subgraphs:")
        # For op_names that are not match certain pattern
        # We group ops that have direct relationships into a subgraph

        # Step 1. This passDeal with nasnet structure
        # Let one concat op be a subgraph
        concat_reg = 'cell_(stem_)?[0-9]+/cell_output/concat'
        tmp_untreated_op_name_list = untreated_op_name_list
        for op_name in tmp_untreated_op_name_list:
            matched_str = re.finditer(concat_reg, op_name)
            is_concat =False
            for ms in matched_str:
                if ms.group() == op_name:
                    is_concat = True
                    break
            if is_concat:
                subgraph = Subgraph(pattern+"subgraph_%d" % (subgraph_idx))
                subgraph.buildWithOpList([op_name], name_op_dict)
                subgraph_list.append(subgraph)
                untreated_op_name_list.remove(op_name)
                subgraph_idx += 1
        # Step 2.
        # Build subgraph with operators that does not belong to prefix
        while(len(untreated_op_name_list)>0):
            tmp_list = []
            op_name = untreated_op_name_list[0]
            tmp_list.append(op_name)
            untreated_op_name_list.remove(op_name)
            
            op = name_op_dict[op_name]
            op_queue = queue.Queue()
            put_op_parents_and_children(op_queue, op, untreated_op_name_list)
            while(not op_queue.empty()):
                tmp_op_name = op_queue.get()
                if tmp_op_name in untreated_op_name_list:
                    tmp_list.append(tmp_op_name)
                    untreated_op_name_list.remove(tmp_op_name)
                tmp_op = name_op_dict[tmp_op_name]
                put_op_parents_and_children(op_queue, tmp_op, untreated_op_name_list)
            subgraph = Subgraph(pattern+"subgraph_%d"%(subgraph_idx))
            subgraph.buildWithOpList(tmp_list, name_op_dict)
            subgraph_list.append(subgraph)
            subgraph_idx += 1
        # Add subgraph in to self op_name and name_op_dict
        for subgraph in subgraph_list:
            self.op_name_list.append(subgraph.name)
            self.name_op_dict[subgraph.name] = subgraph
            name_op_dict[subgraph.name] = subgraph
        
        # Setup relationship for subgraphs
        # Add subgraph name into subgraph's parents and children set
        build_subgraph_relationship(self.op_name_list, self.name_op_dict)
        # for parent_subgraph_name in self.op_name_list:
        #     for child_subgraph_name in self.op_name_list:
        #         if parent_subgraph_name == child_subgraph_name:
        #             continue
        #         parent_subgraph = self.name_op_dict[parent_subgraph_name]
        #         child_subgraph = self.name_op_dict[child_subgraph_name]
        #         for p in child_subgraph.parents:
        #             if p in parent_subgraph.op_name_list:
        #                 child_subgraph.parents.add(parent_subgraph.name)
        #                 parent_subgraph.children.add(child_subgraph.name)
        #                 break
        #         for c in parent_subgraph.children:
        #             if c in child_subgraph.op_name_list:
        #                 parent_subgraph.children.add(child_subgraph.name)
        #                 child_subgraph.parents.add(parent_subgraph.name)

        # print("All subgraph:")
        # for subgraph_name in self.op_name_list:
        #     print(self.name_op_dict[subgraph_name])
    

    def out_op_device_type(self):
        lines = []
        for op_name in self.op_name_list:
            line = "%s %d\n" % (op_name, self.op_def.device_type)
            lines.append(line)
        return lines

    
    # Used to compare the alone result with parallel result
    def summary_new_latency(self, new_op_name_dict):
        cpu_latency = 0.0
        gpu_latency = 0.0
        for op_name in self.op_name_list:
            op = new_op_name_dict[op_name]
            cpu_latency += op.op_def.operator_latency.CPU_latency
            gpu_latency += op.op_def.operator_latency.GPU_latency
        return cpu_latency, gpu_latency


    def __str__(self):
        operator_latency = self.op_def.operator_latency
        str_latency = "(%f,%f,%f,%f)" % \
            (operator_latency.CPU_latency, \
                operator_latency.GPU_latency, \
                    operator_latency.Transpose_latency_NCHW_to_NHWC, \
                        operator_latency.Transpose_latency_NHWC_to_NCHW)
        return ("name: %s\nlatency: %s\nnodes:%s\nparents:%s\nchildren:%s\ninput_tensors_map%s\ninput_tensors%s\noutput_tensors%s\n" \
            %(self.name, str_latency, self.op_name_list, self.parents, self.children,\
                self.op_def.operator_latency.input_data_trans_latency, self.input_tensors, self.output_tensors))

    
# Get 'cpu_name_list' and gpu_name_list from GLPK result file
def write_subgraph_device_placement_result(cpu_name_list=None, gpu_name_list=None, \
    name_op_dict=None, op_execution_order_list=None):
    print("CPU subgraphs:")
    print(cpu_name_list)
    print("GPU subgraphs:")
    print(gpu_name_list)
    lines = []
    # if has op execution order, append op name 
    if op_execution_order_list != None:
        cpu_name_list = []
        gpu_name_list = []
        for (op_name, device, start_time) in op_execution_order_list:
            subgraph = name_op_dict[op_name]
            # compatibility for Greedy and LP
            if device == 1 or device == 0:
                subgraph.op_def.device_type = net_struct.DeviceType.CPU
            elif device == 2 or device == 3:
                subgraph.op_def.device_type = net_struct.DeviceType.GPU
            if isinstance(subgraph, Subgraph):
                lines.extend(subgraph.out_op_device_type())
            else:
                lines.append("%s %d\n" % (op_name, subgraph.op_def.device_type))
    elif cpu_name_list != None and gpu_name_list != None:
        for op_name in cpu_name_list:
            subgraph = name_op_dict[op_name]
            assert(isinstance(subgraph, Subgraph))
            subgraph.op_def.device_type = net_struct.DeviceType.CPU
            lines.extend(subgraph.out_op_device_type())
        for op_name in gpu_name_list:
            subgraph = name_op_dict[op_name]
            assert(isinstance(subgraph, Subgraph))
            subgraph.op_def.device_type = net_struct.DeviceType.GPU
            lines.extend(subgraph.out_op_device_type())

    return lines


def write_op_device_placement_result(cpu_name_list, gpu_name_list, result_file_path):
    print("CPU subgraphs:")
    print(cpu_name_list)
    print("GPU subgraphs:")
    print(gpu_name_list)
    f = open(result_file_path, 'w')
    lines = []
    for op_name in cpu_name_list:
        lines.append("%s %d\n" % (op_name, 0))
    for op_name in gpu_name_list:
        lines.append("%s %d\n" % (op_name, 3))
    f.writelines(lines)
    f.flush()
    f.close()
    print("Write result done.")


def get_model_module_name_list(model):
    # Using module prefix to form the subgraph
    module_name_list = []
    if model.find('nasnet') >= 0:
        module_name_list.extend(['cell_stem_0/', 'cell_stem_1/'])
    if model == 'pnasnet-large':
        module_name_list.extend(['cell_'+str(i)+'/' for i in range(12)])
    elif model == 'pnasnet-mobile':
        module_name_list.extend(['cell_'+str(i)+'/' for i in range(9)])
    elif model == 'nasnet-large':
        module_name_list.extend(['cell_'+str(i)+'/' for i in range(18)])
        module_name_list.insert(8, 'reduction_cell_0/')
        module_name_list.insert(15, 'reduction_cell_1/')
    elif model == 'nasnet-mobile':
        module_name_list.extend(['cell_'+str(i)+'/' for i in range(12)])
        module_name_list.insert(6, 'reduction_cell_0/')
        module_name_list.insert(11, 'reduction_cell_1/')
    elif model == "inception-v3":
        inception_prefix = "InceptionV3/InceptionV3/"
        inception_module_list = ["Mixed_5b/", "Mixed_5c/", "Mixed_5d/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/", "Mixed_7a/", "Mixed_7b/", "Mixed_7c/"]
        inception_module_list = [inception_prefix + module for module in inception_module_list]
        module_name_list.extend(inception_module_list)
    elif model == "inception-v4":
        inception_prefix = "InceptionV4/InceptionV4/"
        inception_module_list = ["Mixed_4a/", "Mixed_5b/", "Mixed_5c/", "Mixed_5d/", "Mixed_5e/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/","Mixed_6f/","Mixed_6g/","Mixed_6h/",\
            "Mixed_7a/","Mixed_7b/","Mixed_7c/","Mixed_7d/",]
        inception_module_list = [inception_prefix + module for module in inception_module_list]
        module_name_list.extend(inception_module_list)
    else:
        print("Model %s does not suport yet." % (model))
        return None
    return module_name_list


def get_inception_one_module_name(op_name_prefix, op_name_list):
    module_op_names = []
    branches = set()
    for op_name in op_name_list:
        if op_name.find(op_name_prefix) == 0:
            module_op_names.append(op_name)
            if op_name.find("Branch") > 0:
                branches.add(op_name.split("/")[3])
    
    return module_op_names, list(branches)


def build_multi_subgraphs(model, mobile, thread):
    # Read profile data
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread)
    
    module_name_list = get_model_module_name_list(model)
    subgraph_name_list = []
    for module_name in module_name_list:
        # For one module with multiple subgraphs, we need build subgraph and update the op_dict
        parent_subgraph = Subgraph(module_name)
        if model != None and model.find("inception") >=0:
            one_module_name, branches = get_inception_one_module_name(module_name, op_name_list)
            parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, branches, pattern=module_name)
            subgraph_name_list.extend(parent_subgraph.op_name_list)
        elif model != None and (model.find("pnasnet") >=0 or model.find("nasnet") >=0):
            parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, pnasnet_mobile_subgraph_subprefix(), pattern=module_name)
            subgraph_name_list.extend(parent_subgraph.op_name_list)
    # Build the relationship for subgraphs
    for subgraph_a_name in subgraph_name_list:
        subgraph_a = name_op_dict[subgraph_a_name]
        
        for subgraph_b_name in subgraph_name_list:
            subgraph_b = name_op_dict[subgraph_b_name]
            if subgraph_a_name == subgraph_b_name:
                continue
            
            for parent_name in subgraph_a.parents:
                if parent_name in subgraph_b.op_name_list:
                    subgraph_b.children.add(subgraph_a_name)
                    subgraph_a.parents.add(subgraph_b_name)
                    name_op_dict[subgraph_b_name] = subgraph_b
                    name_op_dict[subgraph_a_name] = subgraph_a
                    break
    # for subgraph_name in subgraph_name_list:
    #     print(name_op_dict[subgraph_name])
    return subgraph_name_list, name_op_dict
    

def build_graphs_with_cluster_lists(clusters, op_name_list, name_op_dict):
    subgraph_name_list = []
    for cluster_id, op_names in clusters.items():
        subgraph_name = "subgraph_{}".format(cluster_id)
        subgraph = Subgraph(subgraph_name)
        
        subgraph.buildWithOpList(op_names, name_op_dict)
        subgraph_name_list.append(subgraph_name)
        name_op_dict[subgraph_name] = subgraph
    build_subgraph_relationship(subgraph_name_list, name_op_dict)
    return subgraph_name_list, name_op_dict


def find_input_nodes(op_name_list, name_op_dict):
    children_set = set()
    # [print(op_name) for op_name in op_name_list]
    for op_name in op_name_list:
        op = name_op_dict[op_name]
        children_set = children_set.union(op.children)
    input_nodes = list(set(op_name_list).difference(children_set))
    return input_nodes


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
                            cpu_latency = op.op_def.operator_latency.CPU_latency
                            gpu_latency = op.op_def.operator_latency.GPU_latency
                            if op_name not in unsupported_op_names and gpu_latency < cpu_latency:
                                lines.insert(index, "%s %d\n" % (op_name, 3))
                                untreated_op_latency += gpu_latency
                            else:
                                lines.insert(index, "%s %d\n" % (op_name, 0))
                                untreated_op_latency += cpu_latency
                            print(index, "%s %d\n" % (op_name, 0))
                            break
    return lines, untreated_op_latency



if __name__ == "__main__":
    model, mobile, thread = utils.parse_model_mobile()
    subgraph_name_list, name_op_dict = build_multi_subgraphs(model, mobile, thread)
    print(find_input_nodes(subgraph_name_list, name_op_dict))
