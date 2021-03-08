
from utils import utils

app_file_path = "/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/dfmodel1/npu/appendix-layerwise-latency.csv"
app_f = open(app_file_path, 'r')
app_lines = app_f.readlines()
app_name_latency_dict = {}
for line in app_lines:
    com = line.strip().split(" ")
    app_name_latency_dict[com[0]] = com[1]
app_f.close()

latency_file_path = "/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/dfmodel1/npu/npu-dfmodel1-layerwise-latency.csv"
new_latency_file_path = "/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/dfmodel1/npu/npu-dfmodel1-layerwise-latency-new.csv"
f = open(latency_file_path)
lines = f.readlines()
new_lines = []
for line in lines:
    com = line.strip().split(" ")
    new_line = None
    if com[0] in app_name_latency_dict.keys():
        npu_latency = app_name_latency_dict[com[0]]
        if com[4] == '100000':
            com[4] = npu_latency
            new_line = "{} {} {} {} {}\n".format(com[0], com[1], com[2], com[3], com[4])
        if abs(float(npu_latency)-float(com[4]))/float(com[4]) > 0.1:
            print("{} {} {}".format(com[0], com[4], npu_latency))
    if new_line != None:
        new_lines.append(new_line)
    else:
        new_lines.append(line)

f.close()

new_file = open(new_latency_file_path, 'x')
new_file.writelines(new_lines)
new_file.flush()
new_file.close()
