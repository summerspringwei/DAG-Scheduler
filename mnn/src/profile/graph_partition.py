
import queue
import pysnooper
import os

from profile import subgraph
from utils import utils
from profile import find_critical_node
from profile import read_profile_data

def auto_build_multi_subgraph(op_name_list, name_op_dict):
    def add_new_subgraph(subgraph_op_name_list, name_op_dict):
        pass
    subgraph_list = []
    input_nodes = subgraph.find_input_nodes(op_name_list, name_op_dict)
    visited = set()
    subgraph_op_name_list = []
    to_be_sche_ops = input_nodes
    subgraph_idx = 1
    while len(to_be_sche_ops)>0:
        # Remove op that h
        for tmp_op_name in visited:
            if tmp_op_name in to_be_sche_ops:
                to_be_sche_ops.remove(tmp_op_name)
        if len(to_be_sche_ops) == 0:
            break
        op_name = to_be_sche_ops[0]
        
        to_be_sche_ops.remove(op_name)
        visited.add(op_name)
        op = name_op_dict[op_name]
        subgraph_op_name_list.append(op_name)
        # group one path into a subgraph
        while len(op.children) == 1 and len(op.parents) <= 1:
            for op_child_name in op.children:
                op_name = op_child_name
            if op_name in visited:
                break
            op = name_op_dict[op_name]
            subgraph_op_name_list.append(op_name)
            visited.add(op_name)
        subgraph = subgraph.Subgraph(str(subgraph_idx))
        subgraph_idx += 1
        subgraph.buildWithOpList(subgraph_op_name_list, name_op_dict)
        subgraph_op_name_list.clear()
        subgraph_list.append(subgraph)
        # Move node with multi inputs or outputs 
        if op_name not in visited and (len(op.children) > 1 or len(op.parents) > 1):
            subgraph_op_name_list = [op_name]
            visited.add(op_name)
            subgraph = subgraph.Subgraph(str(subgraph_idx))
            subgraph_idx += 1
            subgraph.buildWithOpList(subgraph_op_name_list, name_op_dict)
            subgraph_list.append(subgraph)
        for child_name in op.children:
            if child_name in op_name_list:
                to_be_sche_ops.append(child_name)
        subgraph_op_name_list.clear()
    
    return subgraph_list


def auto_build_multi_subgrap_with_weight(op_name_list, name_op_dict, name_weight_dict, weight_threshold):

    def add_new_subgraph(subgraph_op_name_list, subgraph_idx, name_op_dict):
        subgraph = subgraph.Subgraph(str(subgraph_idx))
        subgraph.buildWithOpList(subgraph_op_name_list, name_op_dict)
        subgraph_list.append(subgraph)
        name_op_dict[str(subgraph_idx)] = subgraph
        # print("Add new subgraph")
        return subgraph_idx + 1

    subgraph_list = []
    input_nodes = subgraph.find_input_nodes(op_name_list, name_op_dict)
    visited = set()
    subgraph_op_name_list = []
    to_be_sche_ops = input_nodes
    subgraph_idx = 1
    while len(to_be_sche_ops)>0:
        # Remove op that h
        for tmp_op_name in visited:
            if tmp_op_name in to_be_sche_ops:
                to_be_sche_ops.remove(tmp_op_name)
        if len(to_be_sche_ops) == 0:
            break
        op_name = to_be_sche_ops[0]
        to_be_sche_ops.remove(op_name)
        op = name_op_dict[op_name]
        
        op_weight = name_weight_dict[op_name]
        # group one path into a subgraph
        # critical node
        if op_name not in visited and (op_weight >= weight_threshold or len(op.children)==0):
            print(op_name, name_weight_dict[op_name])
            subgraph_op_name_list = [op_name]
            visited.add(op_name)
            subgraph_idx = add_new_subgraph(subgraph_op_name_list, subgraph_idx, name_op_dict)
            for child_name in op.children:
                if child_name not in visited and child_name in op_name_list:
                    to_be_sche_ops.append(child_name)
        elif op_name not in visited and op_weight < weight_threshold:
            # broud-first search
            subgraph_op_name_list = [op_name]
            tmp_queue =queue.Queue()
            tmp_queue.put(op_name)
            visited.add(op_name)
            while tmp_queue.qsize() > 0:
                op_name = tmp_queue.get()
                op = name_op_dict[op_name]
                for child_name in (list(op.children) + list(op.parents)):
                    if child_name not in visited:
                        if name_weight_dict[child_name] < weight_threshold:
                            tmp_queue.put(child_name)
                            visited.add(child_name)
                            subgraph_op_name_list.append(child_name)
                        elif child_name in op_name_list:
                            to_be_sche_ops.append(child_name)
                
            subgraph_idx = add_new_subgraph(subgraph_op_name_list, subgraph_idx, name_op_dict)
            subgraph_op_name_list.clear()
    
    return subgraph_list


def auto_merge_subgraph(subgraph_list, name_op_dict, num_ops_threshold=3, num_subgraph_threshold=8):
    subgraph_name_op_dict = {}
    for subgraph in subgraph_list:
        subgraph_name_op_dict[subgraph.name] = subgraph
    class Relationship:
        parents = 1
        children = 2
    
    def get_subgraph_relationship(subgraph, subgraph_name_op_dict, relationship):
        subgraph_relation_names = []
        if subgraph.name in subgraph_name_op_dict.keys():
            relations = None
            if relationship == Relationship.parents:
                relations = subgraph.parents
            elif relationship == Relationship.children:
                relations = subgraph.children
            for relation_name in relations:
                if relation_name in subgraph_name_op_dict.keys() \
                    and isinstance(subgraph_name_op_dict[relation_name], Subgraph):
                    subgraph_relation_names.append(relation_name)
        return subgraph_relation_names

    def merge_parent_and_child_subgraph(parent_subgraph, child_subgraph, subgraph_list, subgraph_name_op_dict):
        parent_subgraph.op_name_list.extend(child_subgraph.op_name_list)
        # @pysnooper.snoop()
        #with pysnooper.snoop():
        print("child child {}".format(child_subgraph.children))
        print("parent child {}".format(parent_subgraph.children))
        parent_subgraph.children.update(child_subgraph.children)
        print("parent child {}".format(parent_subgraph.children))
        print("merge {} into {}".format(child_subgraph.name, parent_subgraph.name))
        # Child's children's parent are need to update
        for child_child_name in child_subgraph.children:
            if child_child_name in subgraph_name_op_dict.keys():
                sg = subgraph_name_op_dict[child_child_name]
                if child_subgraph.name in sg.parents:
                    sg.parents.remove(child_subgraph.name)
                    print("{} removed {}".format(sg.name, child_subgraph.name))
                sg.parents.add(parent_subgraph.name)
        subgraph_list.remove(child_subgraph)
        subgraph_name_op_dict.pop(child_subgraph.name)
        parent_subgraph.update(name_op_dict)
    
    # while len(subgraph_list) > num_subgraph_threshold:
    try_count = len(subgraph_list) - num_subgraph_threshold + 1
    for i in range(try_count):
        parent_subgraph = None
        child_subgraph = None
        find = False
        for subgraph in subgraph_list:
            if len(subgraph.op_name_list) < num_ops_threshold:
                subgraph_parent_names = get_subgraph_relationship(subgraph, subgraph_name_op_dict, Relationship.parents)
                if len(subgraph_parent_names) == 1:
                    parent_subgraph = subgraph_name_op_dict[subgraph_parent_names[0]]
                    child_subgraph = subgraph
                    find = True
                    break
        # Merge child subgraph into parent subgraph
        if find:
            merge_parent_and_child_subgraph(parent_subgraph, child_subgraph, subgraph_list, subgraph_name_op_dict)
            print("out parent child {}".format(parent_subgraph.children))
            # pass
        else:
            break
    # For subgraph that has no parents, just merge it with it's children
    try_count = len(subgraph_list) - num_subgraph_threshold + 1
    for i in range(try_count):
        tmp_subgraph = None
        for subgraph in subgraph_list:
            subgraph_parent_names = get_subgraph_relationship(subgraph, subgraph_name_op_dict, Relationship.parents)
            subgraph_children_names = get_subgraph_relationship(subgraph, subgraph_name_op_dict, Relationship.children)
            if len(subgraph_parent_names) == 0 and len(subgraph_children_names) == 1:
                child_name = subgraph_children_names[0]
                child_subgraph = subgraph_name_op_dict[child_name]
                child_subgraph.op_name_list.extend(subgraph.op_name_list)
                child_subgraph.update(name_op_dict)
                tmp_subgraph = subgraph
                print("Merged {}".format(subgraph.name))
                break
        if tmp_subgraph != None:
            subgraph_list.remove(tmp_subgraph)
            if tmp_subgraph.name in name_op_dict:
                name_op_dict.pop(tmp_subgraph.name)
        else:
            break

    return subgraph_list


if __name__ == "__main__":
    model, mobile, thread = utils.parse_model_mobile()
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread)
    name_weight_dict, result = get_node_weight_wrapper(model)
    pnasnet_module_list = ['cell_stem_0/', 'cell_stem_1/']
    if model == 'pnasnet-large':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
    for module_name in pnasnet_module_list:
        if module_name != 'cell_0/':
            continue
        module_op_name_list = []
        for op_name in op_name_list:
            if op_name.find(module_name)==0:
                module_op_name_list.append(op_name)
        # subgraph_list = auto_build_multi_subgrap_with_weight(module_op_name_list, name_op_dict, name_weight_dict, 513000.0)
        subgraph_list = auto_build_multi_subgrap_with_weight(module_op_name_list, name_op_dict, name_weight_dict, 648000)
        
        # subgraph_list = auto_build_multi_subgrap_with_weight(op_name_list, name_op_dict, name_weight_dict, 3240000)
        # subgraph_list = auto_build_multi_subgraph(module_op_name_list, name_op_dict)
        nodes = []
        for subgraph in subgraph_list:
            name_op_dict[subgraph.name] = subgraph
        subgraph.build_subgraph_relationship([subgraph.name for subgraph in subgraph_list], name_op_dict)
        print(len(subgraph_list))
        for subgraph in subgraph_list:
            print(subgraph)
        auto_merge_subgraph(subgraph_list, name_op_dict, num_ops_threshold=5)
        subgraph.build_subgraph_relationship([subgraph.name for subgraph in subgraph_list], name_op_dict)
        print("After merge " + str(len(subgraph_list)))
        
        for subgraph in subgraph_list:
            print(subgraph)
            nodes.extend(subgraph.op_name_list)
        print(len(subgraph_list))
        print(len(nodes))
        print(len(module_op_name_list))
        