import os

from utils import utils
from profile import read_profile_data


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


def generate_graphviz_file_for_partitioning(op_name_list, name_op_dict, set_weight=False):
    """Generate graphviz content for a DAG
    and will be partitioned by dagP
    """
    lines = []
    lines.append("digraph G {\n")
    for op_name in op_name_list:
        op = name_op_dict[op_name]
        name = op.name
        for child in op.children:
            lines.append('\"{}\"->\"{}\";\n'.format(name, child))
        node_attr = ""
        if set_weight:
            # node_attr = '\"{}\" [weight={},label={}];\n'.format(name, op.op_def.operator_latency.CPU_latency, name.split("/")[-1])
            node_attr = '\"{}\" [weight={}];\n'.format(name, op.op_def.operator_latency.CPU_latency)
        else:
            # node_attr = '\"{} [label={}]\";\n'.format(name, name.split("/")[-1])
            node_attr = '\"{}\";\n'.format(name)
        lines.append(node_attr)
    lines.append("}")
    return lines


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



def generate_graphviz_latency_file():
    pass


# def draw_graphviz_dot(model, mobile, thread, device, solver):
#     solver_str = ""
#     if solver == "greedy":
#         solver_str = "greedy-"
#     device_placement_file_path = os.path.join(model_dir, mobile, "greedy-placement-{}-cpu-{}.txt".format(model, thread))


def dot2png(graphviz_file_path, graph_dot):
    utils.write_lines(graphviz_file_path, graph_dot)
    extend_str = graphviz_file_path.split('.')[-1]
    # png_file_path = graphviz_file_path.split('.')[0]+'.png'
    png_file_path = graphviz_file_path.replace(extend_str, 'png')
    sh_cmd  = "dot -Tpng {} -o {}".format(graphviz_file_path, png_file_path)
    print(sh_cmd)
    os.system(sh_cmd)


def generate_device_placement_figures(model, mobile, thread):
    model_dir = os.path.join(utils.get_project_path(), "mnn/models/", model)
    read_profile_data.load_model_profile(model, mobile, thread)
    op_name_list, name_op_dict, net_def = read_profile_data.gather_model_profile(
        os.path.join(model_dir, model + "-info.txt"),
        os.path.join(model_dir, mobile, model+'-'+mobile+'-data-trans.csv'),
        os.path.join(model_dir, mobile, mobile+"-"+model+"-layerwise-latency.csv"),
        thread)
    
    greedy_device_placement_file_path = os.path.join(model_dir, mobile, "greedy-placement-{}-cpu-{}.txt".format(model, thread))
    ilp_device_placement_file_path = os.path.join(model_dir, mobile, "mDeviceMap-{}-cpu-{}.txt".format(model, thread))

    greedy_placement, ilp_placement = None, None
    if os.path.exists(greedy_device_placement_file_path):
        greedy_placement = read_device_placement(greedy_device_placement_file_path)
        greedy_graphviz_file_path = os.path.join(model_dir, mobile, "{}-graphviz-{}-cpu-{}.gv".format("greedy", model, thread))
        greedy_graph_dot = generate_graphviz_file(greedy_placement, op_name_list, name_op_dict, "greedy {} {} {} thread(s)".format(model, mobile, thread))
        dot2png(greedy_graphviz_file_path, greedy_graph_dot)
    
    if os.path.exists(ilp_device_placement_file_path):
        ilp_placement = read_device_placement(ilp_device_placement_file_path)
        ilp_graphviz_file_path = os.path.join(model_dir, mobile, "{}-graphviz-{}-cpu-{}.gv".format("ilp", model, thread))
        ilp_graph_dot = generate_graphviz_file(ilp_placement, op_name_list, name_op_dict, "ILP {} {} {} thread(s)".format(model, mobile, thread))
        dot2png(ilp_graphviz_file_path, ilp_graph_dot)
    if os.path.exists(greedy_device_placement_file_path) and os.path.exists(ilp_device_placement_file_path):
        diff_graphviz_file_path = os.path.join(model_dir, mobile, "{}-graphviz-{}-cpu-{}.gv".format("ilp-greedy-diff", model, thread))
        diff_graph_dot = generate_graphviz_diff_file(ilp_placement, greedy_placement, op_name_list, name_op_dict, \
            "ILP & greedy compare {} {} {} thread(s)".format(model, mobile, thread))
        dot2png(diff_graphviz_file_path, diff_graph_dot)
    
    # Draw ILP op device placement
    ilp_op_device_placement_file_path = os.path.join(model_dir, mobile, "mDeviceMap-{}-cpu-{}.txt.op".format(model, thread))
    # os.system("diff {} {}".format(ilp_device_placement_file_path, ilp_op_device_placement_file_path))
    if os.path.exists(ilp_op_device_placement_file_path):
        ilp_op_placement = read_device_placement(ilp_op_device_placement_file_path)
        ilp_op_graphviz_file_path = os.path.join(model_dir, mobile, "{}-graphviz-{}-cpu-{}.op.gv".format("ilp", model, thread))
        ilp_op_graph_dot = generate_graphviz_file(ilp_op_placement, op_name_list, name_op_dict, "ILP {} {} {} thread(s)".format(model, mobile, thread))
        dot2png(ilp_op_graphviz_file_path, ilp_op_graph_dot)


def generate_model_figures(model, mobile, thread, set_weight=True):
    """Generate the dot file of model
    @param: set_weight: whether using the latency on CPU as the nodes' weight
    @returns: return the file path of the dot file
    """
    model_dir = os.path.join(utils.get_project_path(), "mnn/models/", model)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread)
    
    lines = generate_graphviz_file_for_partitioning(op_name_list, name_op_dict, set_weight=set_weight)
    graphviz_file_path = os.path.join(model_dir, model+".dot")
    dot2png(graphviz_file_path, lines)
    return graphviz_file_path



if __name__ == "__main__":
    model, mobile, thread, _ = utils.parse_model_mobile()
    # model, mobile, thread = "inception-v4", "redmi", 2
    generate_model_figures(model, mobile, thread)
    generate_device_placement_figures(model, mobile, thread)
