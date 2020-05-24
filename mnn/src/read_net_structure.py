
from net_struct import *
from read_profile_data import *
import queue
import re
        
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


class Subgraph(Operator):
    def __init__(self, name):
        super().__init__(name)
        self.op_name_list = []
        # self.subgraph_list = []
        self.name_op_dict = {}
        self.op_def = OperatorDef()
        self.op_def.type = "OpType_Subgraph"

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
        

    def buildWithOpList(self, op_name_list, name_op_dict, pattern=None):
        # Filter op with pattern, else directly set list
        self._set_op_list_with_filter(op_name_list, name_op_dict, pattern)
        self._findGraphInputsOutputs()
        self._summaryLatency(name_op_dict, name_op_dict)

    # Set the subgraph computing latency as the sum of all the internal operators latency
    # Set the subgraph data transformation latency as the sum of all the input operators' data transformation latency
    def _summaryLatency(self, name_op_latency, global_name_op_dict):
        for op_name in self.op_name_list:
            op = self.name_op_dict[op_name]
            self.op_def.operatorLatency.CPU_latency += op.op_def.operatorLatency.CPU_latency
            self.op_def.operatorLatency.GPU_latency += op.op_def.operatorLatency.GPU_latency
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
        
        for op_name in self.input_nodes:
            op_input = global_name_op_dict[op_name]
            self.op_def.operatorLatency.Transpose_latency_NHWC_to_NCHW += op_input.op_def.operatorLatency.Transpose_latency_NHWC_to_NCHW
            self.op_def.operatorLatency.Transpose_latency_NCHW_to_NHWC += op_input.op_def.operatorLatency.Transpose_latency_NCHW_to_NHWC
            for (addr, _) in self.input_tensors:
                if addr in op_input.op_def.operatorLatency.input_data_trans_latency.keys():
                    self.op_def.operatorLatency.input_data_trans_latency[addr] = op_input.op_def.operatorLatency.input_data_trans_latency[addr]

        
        # Debug info
        print(self.name + "input nodes and output nodes:")
        print(self.parents)
        print(self.children)
        print(self.input_nodes)
        print(self.input_tensors)
        print(self.output_nodes)
        print(self.output_tensors)
        print(self.op_def.operatorLatency.Transpose_latency_NCHW_to_NHWC)
        print(self.op_def.operatorLatency.Transpose_latency_NHWC_to_NCHW)
        print(self.op_def.operatorLatency.input_data_trans_latency)
        print()


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
        for parent_subgraph_name in self.op_name_list:
            for child_subgraph_name in self.op_name_list:
                if parent_subgraph_name == child_subgraph_name:
                    continue
                parent_subgraph = self.name_op_dict[parent_subgraph_name]
                child_subgraph = self.name_op_dict[child_subgraph_name]
                for p in child_subgraph.parents:
                    if p in parent_subgraph.op_name_list:
                        child_subgraph.parents.add(parent_subgraph.name)
                        parent_subgraph.children.add(child_subgraph.name)
                        break
                for c in parent_subgraph.children:
                    if c in child_subgraph.op_name_list:
                        parent_subgraph.children.add(child_subgraph.name)
                        child_subgraph.parents.add(parent_subgraph.name)

        print("All subgraph:")
        for subgraph_name in self.op_name_list:
            print(self.name_op_dict[subgraph_name])
    

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
            cpu_latency += op.op_def.operatorLatency.CPU_latency
            gpu_latency += op.op_def.operatorLatency.GPU_latency
        return cpu_latency, gpu_latency


    def __str__(self):
        operator_latency = self.op_def.operatorLatency
        str_latency = "(%f,%f,%f,%f)" % \
            (operator_latency.CPU_latency, \
                operator_latency.GPU_latency, \
                    operator_latency.Transpose_latency_NCHW_to_NHWC, \
                        operator_latency.Transpose_latency_NHWC_to_NCHW)
        return ("name: %s\nlatency: %s\nnodes:%s\nparents:%s\nchildren:%s\ninput_tensors%s\noutput_tensors%s\n" \
            %(self.name, str_latency, self.op_name_list, self.parents, self.children,\
                self.op_def.operatorLatency.input_data_trans_latency, self.output_tensors))

    

def write_subgraph_device_placement_result(cpu_name_list, gpu_name_list, \
    name_op_dict, result_file_path, op_execution_order_list=None):
    print("CPU subgraphs:")
    print(cpu_name_list)
    print("GPU subgraphs:")
    print(gpu_name_list)
    f = open(result_file_path, 'w')
    lines = []
    # if has op execution order, append op name 
    if op_execution_order_list != None:
        cpu_name_list = []
        gpu_name_list = []
        for (op_name, device, start_time) in op_execution_order_list:
            subgraph = name_op_dict[op_name]
            assert(isinstance(subgraph, Subgraph))
            if device == 1:    
                subgraph.op_def.device_type = DeviceType.CPU
            elif device == 2:
                subgraph.op_def.device_type = DeviceType.GPU
            lines.extend(subgraph.out_op_device_type())
    else:
        for op_name in cpu_name_list:
            subgraph = name_op_dict[op_name]
            assert(isinstance(subgraph, Subgraph))
            subgraph.op_def.device_type = DeviceType.CPU
            lines.extend(subgraph.out_op_device_type())
        for op_name in gpu_name_list:
            subgraph = name_op_dict[op_name]
            assert(isinstance(subgraph, Subgraph))
            subgraph.op_def.device_type = DeviceType.GPU
            lines.extend(subgraph.out_op_device_type())
    # f.writelines(lines)
    # f.flush()
    # f.close()
    # print("Write result done.")
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



if __name__ == "__main__":
    # op_name_list, name_op_dict = read_net_info("pnasnet-mobile/pnasnet-info.txt")
    op_name_list, name_op_dict, _  = gather_model_profile(
        "/mnt/d/home/Projects/DAG-scheduler/mnn/pnasnet-mobile/pnasnet-info.txt",
        "/mnt/d/home/Projects/DAG-scheduler/mnn/redmi_data_trans.txt",
        "/mnt/d/home/Projects/DAG-scheduler/mnn/experimental_result_mnn/redmi-pnasnet-mobile-latency.csv", 1)
    module_name = 'cell_0/'
    parent_subgraph = Subgraph(module_name)
    parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, pnasnet_mobile_subgraph_subprefix(), pattern=module_name)
    write_device_placement_result(['cell_0/subgraph_0', 'cell_0/comb_iter_0/', 'cell_0/comb_iter_1/', 'cell_0/subgraph_2'],\
        ['cell_0/subgraph_1', 'cell_0/comb_iter_2/', 'cell_0/comb_iter_3/', 'cell_0/comb_iter_4/'], \
         name_op_dict, \
        '/mnt/d/home/Projects/DAG-scheduler/mnn/pnasnet-mobile/mDevice_map_pnasnet-mobile-cell_0.txt')
    
    

# cell_stem_1/comb_iter_0/ 1 Operator latency: 2.531502 4.512119 1.600000 1.600000
# cell_stem_1/comb_iter_1/ 2 Operator latency: 3.601987 4.949267 1.600000 1.600000
# cell_stem_1/comb_iter_2/ 3 Operator latency: 3.249724 6.747899 1.600000 1.600000
# cell_stem_1/comb_iter_3/ 4 Operator latency: 1.272897 3.625851 3.200000 3.200000
# cell_stem_1/comb_iter_4/ 5 Operator latency: 2.046891 3.655301 3.200000 3.200000
# cell_stem_1/subgraph_0 6 Operator latency: 12.171825 25.147106 1.600000 1.600000
# cell_stem_1/subgraph_1 7 Operator latency: 2.874615 2.618410 1.600000 1.600000
# cell_stem_1/subgraph_2 8 Operator latency: 0.294615 4.516769 8.000000 8.000000
