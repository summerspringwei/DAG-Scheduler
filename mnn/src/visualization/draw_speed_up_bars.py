import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

mobile_soc_dict = {"lenovo_k5": "Snapdragon 450", "redmi": "Snapdragon 625", \
    "vivo_z3": "Snapdragon 710", "oneplus5t": "Snapdragon 835"}

# data format: [[LP theory], [LP real], [alone]]
def draw_normalized_bars(data, policy_labels, module_name, mobile_name, fig, k):

    x_labels = []
    y = []
    idx_2_thread_dict = {0:1, 1:2, 2:4}
    for i in range(len(data[0])):
        idx = 0
        for latency_list in data:
            x_labels.append("{}-{}".format(policy_labels[idx], idx_2_thread_dict[i]))
            y.append(latency_list[i])
            idx+=1

    # GPU latency at the end
    gpu_latency = data[-1][-1]
    x_labels.append('gpu')
    y.append(gpu_latency)
    
    x = np.arange(len(x_labels))  # the label locations
    width = 0.9  # the width of the bars
    
    # fig, ax = plt.subplots()
    ax = fig.add_subplot(3, 2, k)
    rects1 = ax.bar(x_labels, y, width)
    colors = ['#ce4a50', '#377375', '#4c525f', '#ab9c81', '#517c63', '#f28665', '#ffca0f', \
        '#f1e8cd', '#1263a1', '#e2ab7f', '#505668', 'burlywood', 'chartreuse']
    colors = ['#ab9c81', '#e2ab7f', '#1263a1', '#ce4a50', '#f28665', '#bf418c', \
    '#f28665', '#377375', '#f1e8cd', 'chartreuse']
    for i in range(len(rects1)):
        rects1[i].set_color(colors[i%len(colors)])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized latency')
    ax.set_title('{} {}'.format(mobile_soc_dict[mobile_name], module_name))
    ax.set_xticks(x)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.{}f}'.format(height, 2),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    fig.tight_layout()
    # plt.show()


def read_and_normalize_data(mobile, policy_labels):
    file_path = "result_data/greedy-result-latency-{}.txt".format(mobile)
    f = open(file_path, 'r')
    lines = f.readlines()
    model_data_dict = {}
    
    for line in lines:
        com = line.split('\t')
        model_name = ""
        data = []
        for i in range(len(com)):
            if i == 0:
                model_name = com[i]
            else:
                data.append(float(com[i]))
        # Normalize with gpu latency
        data = [d/data[-1] for d in data]
        tmp_data = []
        num_of_threads = int((len(data)-1) / len(policy_labels))
        for i in range(len(policy_labels)):
            
            tmp_data.append(data[i*num_of_threads: (i+1)*num_of_threads])
        tmp_data[-1].append(data[-1])
        model_data_dict[model_name] = tmp_data
        print(model_name, tmp_data)
    return model_data_dict


def main():
    mobile = 'vivo_z3'
    policy_labels=[ "Theory", "Real", "Alone"]
    model_data_dict = read_and_normalize_data(mobile, policy_labels)
    fig = plt.figure()
    count = 0
    for model, data in model_data_dict.items():
        draw_normalized_bars(data, policy_labels, model, mobile, fig, count+1)
        count += 1
    plt.show()

if __name__ == "__main__":
    main()
