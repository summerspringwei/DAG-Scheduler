import os

from utils import *
from read_profile_data import *
from read_net_structure import *


def read_device_placement(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    result = {}
    index = 0
    for line in lines:
        index += 1
        com = line.strip().split(' ')
        assert(len(com)>=2)
        if len(com) != 2:
            print(com)
        result[com[0]] = (com[1], index)
    return result


def reset_op_name(op_name):
    com = op_name.split("/")[1:]
    short_op_name = ""
    for c in com:
        short_op_name += (c+"/")
    return op_name


def generate_graphviz_file(device_placement_result, op_name_list, name_op_dict, title=""):
    lines = []
    lines.append("digraph G {\n")
    lines.append("""label     = "{}"
    labelloc  =  t // t: Place the graph's title on top.
    fontsize  = 40 // Make title stand out by giving a large font size
    fontcolor = black""".format(title))
    for op_name in op_name_list:
        device, index = device_placement_result[op_name]
        op = name_op_dict[op_name]
        short_op_name = reset_op_name(op_name)
        for child in op.children:
            _, child_index = device_placement_result[child]
            lines.append('\"{}: {}\"->\"{}: {}";\n'.format(index, short_op_name, child_index, reset_op_name(child)))
        node_attr = ""
        if device == '0':
            node_attr = '\"{}: {}\" [shape=box,style=filled,color={}];\n'.format(index, short_op_name, "red")
        else:
            node_attr = '\"{}: {}\" [shape=box,style=filled,color={}];\n'.format(index, short_op_name, "green")
        lines.append(node_attr)
    lines.append("}")
    return lines


def generate_graphviz_diff_file(ilp_device_placement, greedy_device_placement, op_name_list, name_op_dict, title=""):
    lines = []
    lines.append("digraph G {\n")
    lines.append("""label     = "{}"
    labelloc  =  t // t: Place the graph's title on top.
    fontsize  = 40 // Make title stand out by giving a large font size
    fontcolor = black""".format(title))
    for op_name in op_name_list:
        ilp_device, ilp_index = ilp_device_placement[op_name]
        greedy_device, greedy_index = greedy_device_placement[op_name]
        op = name_op_dict[op_name]
        short_op_name = reset_op_name(op_name)
        index = "-"
        node_attr = ""
        for child in op.children:
            _, child_index = ilp_device_placement[child]
            lines.append('\"{}: {}\"->\"{}: {}";\n'.format(index, short_op_name, index, reset_op_name(child)))
        if ilp_device == greedy_device:
            node_attr = '\"{}: {}\" [shape=box,style=filled,color={}];\n'.format(index, short_op_name, "white")
        else:
            if ilp_device == '0':
                node_attr = '\"{}: {}\" [shape=box,style=filled,color={}];\n'.format(index, short_op_name, "red")
            else:
                node_attr = '\"{}: {}\" [shape=box,style=filled,color={}];\n'.format(index, short_op_name, "green")
        lines.append(node_attr)
    lines.append("}")
    return lines


def dot2png(graphviz_file_path, graph_dot):
    write_lines(graphviz_file_path, graph_dot)
    png_file_path = graphviz_file_path.split('.')[0]+'.png'
    sh_cmd  = "dot -Tpng {} -o {}".format(graphviz_file_path, png_file_path)
    print(sh_cmd)
    os.system(sh_cmd)


if __name__ == "__main__":
    model, mobile, thread = parse_model_mobile()
    model_dir = os.path.join("/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/", model)
    folder_path = os.path.join(model_dir, mobile)
    op_name_list, name_op_dict, net_def = gather_model_profile(
        os.path.join(model_dir, model + "-info.txt"),
        os.path.join(model_dir, mobile, model+'-'+mobile+'-data-trans.csv'),
        os.path.join(model_dir, mobile, mobile+"-"+model+"-layerwise-latency.csv"),
        thread)
    # greedy-placement-inception-v3-cpu-1.txt # mDeviceMap-inception-v3-cpu-1.txt

    greedy_device_placement_file_path = os.path.join(model_dir, mobile, "greedy-placement-{}-cpu-{}.txt".format(model, thread))
    ilp_device_placement_file_path = os.path.join(model_dir, mobile, "mDeviceMap-{}-cpu-{}.txt".format(model, thread))
    greedy_placement = read_device_placement(greedy_device_placement_file_path)
    ilp_placement = read_device_placement(ilp_device_placement_file_path)
    greedy_graphviz_file_path = os.path.join(model_dir, mobile, "{}-graphviz-{}-cpu-{}.gv".format("greedy", model, thread))
    ilp_graphviz_file_path = os.path.join(model_dir, mobile, "{}-graphviz-{}-cpu-{}.gv".format("ilp", model, thread))
    diff_graphviz_file_path = os.path.join(model_dir, mobile, "{}-graphviz-{}-cpu-{}.gv".format("ilp-greedy-diff", model, thread))
    greedy_graph_dot = generate_graphviz_file(greedy_placement, op_name_list, name_op_dict, "greedy {} {} {} thread(s)".format(model, mobile, thread))
    ilp_graph_dot = generate_graphviz_file(ilp_placement, op_name_list, name_op_dict, "ILP {} {} {} thread(s)".format(model, mobile, thread))
    diff_graph_dot = generate_graphviz_diff_file(ilp_placement, greedy_placement, op_name_list, name_op_dict, \
        "ILP & greedy compare {} {} {} thread(s)".format(model, mobile, thread))
    
    dot2png(greedy_graphviz_file_path, greedy_graph_dot)
    dot2png(ilp_graphviz_file_path, ilp_graph_dot)
    dot2png(diff_graphviz_file_path, diff_graph_dot)
    

    
    