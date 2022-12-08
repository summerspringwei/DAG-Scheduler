import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from profile import subgraph

def draw_single_bars(labels, data, fig_y_label, fig_title, color=(0.9, 0.0, 0.0, 0.5)):

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, data, width, color=color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(fig_y_label)
    ax.set_title(fig_title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # autolabel(rects1)
    fig.tight_layout()
    plt.show()


def test_plt_bar():
    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    men_means = [20, 34, 30, 35, 27]
    draw_single_bars(labels, men_means, "Score", "Scores by group and gender")


def read_compare_file(file_path):
    f = open(file_path)
    lines = f.readlines()
    cpu_name_ratio_list = []
    gpu_name_ratio_list = []
    for line in lines:
        com = line.split(" ")
        if (len(com) != 4):
            print(com)
            continue
        name = com[0]
        device = com[1]
        alone = float(com[2].strip())
        parallel = float(com[3].strip())
        if device == 'CPU':
            cpu_name_ratio_list.append((name, (parallel) / (alone), parallel))
        elif device == 'OpenCL':
            gpu_name_ratio_list.append((name, (parallel) / (alone), parallel))
    f.close()
    return cpu_name_ratio_list, gpu_name_ratio_list


def compare_nasnet_latency(model, mobile, thread):
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(
        model, mobile, thread)
    
    # Read parallel profile
    compare_file_path = os.path.join(model_dir, mobile, '{}-{}-cpu-{}-compare.csv'.format(mobile, model, thread))
    cpu_op_name_ratio_list, gpu_op_name_ratio_list = read_compare_file(compare_file_path)
    new_name_op_dict = {}
    for (name, ratio, parallel) in cpu_op_name_ratio_list:
        op = Operator(name)
        op.op_def.operator_latency.CPU_latency = parallel / 1000.0
        new_name_op_dict[name] = op
    for (name, ratio, parallel) in gpu_op_name_ratio_list:
        op = Operator(name)
        op.op_def.operator_latency.GPU_latency = parallel / 1000.0
        new_name_op_dict[name] = op

    pnasnet_module_list = ['cell_stem_0/', 'cell_stem_1/']
    if model == 'pnasnet-large':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
    elif model == 'pnasnet-mobile':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(9)])
    elif model == 'nasnet-large':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(18)])
        pnasnet_module_list.extend(['reduction_cell_0/', 'reduction_cell_1/'])
    elif model == 'nasnet-mobile':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
        pnasnet_module_list.extend(['reduction_cell_0/', 'reduction_cell_1/'])
    else:
        print("Model %s does not suport yet." % (model))
        return
    
    cpu_subgraph_ratio_list = []
    gpu_subgraph_ratio_list = []
    for module_name in pnasnet_module_list:
        # For one module with multiple subgraphs, we need build subgraph and update the op_dict
        parent_subgraph = Subgraph(module_name)
        parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, pnasnet_mobile_subgraph_subprefix(), pattern=module_name)
        for subgraph_name in parent_subgraph.op_name_list:
            subgraph = name_op_dict[subgraph_name]
            # print(subgraph)
            parallel_cpu_latency, parallel_gpu_latency = subgraph.summary_new_latency(new_name_op_dict)
            if parallel_cpu_latency * parallel_gpu_latency != 0:
                print((parallel_cpu_latency, parallel_gpu_latency))
            # assert(parallel_cpu_latency * parallel_gpu_latency == 0)
            if parallel_cpu_latency != 0:
                cpu_subgraph_ratio_list.append((subgraph.name, parallel_cpu_latency / subgraph.op_def.operator_latency.CPU_latency))
            elif parallel_gpu_latency != 0:
                gpu_subgraph_ratio_list.append((subgraph.name, parallel_gpu_latency / subgraph.op_def.operator_latency.GPU_latency))
    
    return cpu_subgraph_ratio_list, gpu_subgraph_ratio_list


def draw_inception(model, mobile, thread):
    model_dir = os.path.join("../models/", model)
    
    compare_file_path = os.path.join(model_dir, mobile, '{}-{}-cpu-{}-compare.csv'.format(mobile, model, thread))
    cpu_name_ratio_list, gpu_name_ratio_list = read_compare_file(compare_file_path)
    cpu_ratio_list = [ratio for (name, ratio, _) in cpu_name_ratio_list]
    gpu_ratio_list = [ratio for (name, ratio, _) in gpu_name_ratio_list]
    cpu_labels = range(len(cpu_ratio_list))
    gpu_labels = range(len(gpu_ratio_list))
    draw_single_bars(cpu_labels, cpu_ratio_list, "parallel / alone ", "Snapdragon 625 inception-v3 cpu-2 parallel")
    draw_single_bars(gpu_labels, gpu_ratio_list, "parallel / alone ", "Snapdragon 625 inception-v3 cpu-2 parallel")


def draw_nasnet(model, mobile, thread):
    cpu_subgraph_ratio_list, gpu_subgraph_ratio_list = compare_nasnet_latency(model, mobile, thread)
    print(len(cpu_subgraph_ratio_list))
    print(len(gpu_subgraph_ratio_list))
    cpu_labels = [name for (name, ratio) in cpu_subgraph_ratio_list]
    gpu_labels = [name for (name, ratio) in gpu_subgraph_ratio_list]
    print(cpu_labels)
    print(gpu_labels)
    cpu_ratios = [ratio for (name, ratio) in cpu_subgraph_ratio_list]
    gpu_ratios = [ratio for (name, ratio) in gpu_subgraph_ratio_list]
    draw_single_bars(cpu_labels, cpu_ratios, "parallel / alone ", "Snapdragon 625 CPU {} cpu-{} parallel".format(model, thread), color=(1, 0, 0, 0.5))
    draw_single_bars(gpu_labels, gpu_ratios, "parallel / alone ", "Snapdragon 625 GPU {} gpu-{} parallel".format(model, thread), color=(0, 1, 0, 0.5))
  


if __name__ == "__main__":
    model, mobile, thread = parse_model_mobile()
    if model in ['pnasnet-mobile', 'pnasnet-large', 'nasnet-large', 'nasnet-mobile']:
        draw_nasnet(model, mobile, thread)
    elif model in ['inception-v3', 'inception-v4']:
        draw_inception(model, mobile, thread)
    
