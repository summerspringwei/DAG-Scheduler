
from utils import *
from profile.read_profile_data import *
import os
import queue

def get_node_weight(model, input_name):
    model_dir = os.path.join("../models/", model)
    raw_info_file_path = os.path.join(model_dir, model + "-info.txt")
    name_op_list, name_op_dict = read_net_info(raw_info_file_path)
    visited = set()
    name_weight_map = {}
    for name in name_op_list:
        name_weight_map[name] = 0
        op = name_op_dict[name]
        # Special Case, some node in pnasnet-* does not have parents
        if len(op.parents) == 0:
            print("%s does not have parents" % name)
            visited.add(name)
    initial_weight = 360 * 360 * 5 * 5
    name_weight_map[input_name] = initial_weight
    visited.remove(input_name)
    # print(visited)
    op_queue = queue.Queue()
    op_queue.put(input_name)
    while not op_queue.empty():
        op_name = op_queue.get()
        if op_name in visited:
            continue
        op = name_op_dict[op_name]
        # Check whether all his parents has give their weight to op
        prepared = True
        for parent in op.parents:
            if parent not in visited:
                # print("%s not prepared %s" % (op_name, parent))
                op_queue.put(parent)
                prepared = False
                break
        if not prepared:
            op_queue.put(op_name)
            continue
        
        visited.add(op_name)
        if len(op.children) == 0:
            continue
        # print("%s %f %d => %s" %(op_name, name_weight_map[op_name], len(op.children), op.children))
        # if name_weight_map[op_name] % len(op.children) != 0:
        #     print("%d %d" % (name_weight_map[op_name], len(op.children)))
        weight = (name_weight_map[op_name] / len(op.children))
        for child_name in op.children:
            name_weight_map[child_name] = name_weight_map[child_name] + weight
            # print("<%s %s %f>" % (op_name, child_name, weight))
            op_queue.put(child_name)
    
    weight_count = {}
    for name in name_op_list:
        # print("%s %d" %(name, name_weight_map[name]))
        if name_weight_map[name] not in weight_count.keys():
            weight_count[name_weight_map[name]] = 1
        else:
            weight_count[name_weight_map[name]] += 1
    result = sorted(weight_count.items())
    print("Results")
    for r in result:
        print(r)
    return name_weight_map, result


def get_node_weight_wrapper(model):
    input_node_names = ['InceptionV3/InceptionV3/Conv2d_1a_3x3/Conv2D', \
        'InceptionV4/InceptionV4/Conv2d_1a_3x3/Conv2D', \
        'conv0/Conv2D']
    if model.find('nasnet') >= 0:
        return get_node_weight(model, input_node_names[2])
    elif model == 'inception-v3':
        return get_node_weight(model, input_node_names[0])
    elif model == 'inception-v4':
        return get_node_weight(model, input_node_names[1])
    else:
        return (None, None)

if __name__ == "__main__":
    model, mobile, thread = parse_model_mobile()
    get_node_weight_wrapper(model)