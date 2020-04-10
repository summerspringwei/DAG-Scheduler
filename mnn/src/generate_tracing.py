

# from read_data_trans import *
import json


def read_bench_result(file_path):
    f = open(file_path, 'r')
    op_list = []
    # line format: name, device, latency, start, end
    for line in f.readlines():
        com = line.strip().split(" ")
        if len(com) < 5:
            continue
        name = com[0].strip().split('/')[-1]
        
        device = ''
        
        # print(com[2])
        if com[1].strip() == "CPU":
            device = 'CPU'
        elif com[1].strip() in ["OpenCL"]:
            device = 'GPU'
        else:
            device = 'CONVERT'
        # latency = com[2]
        print(device)
        start = com[3]
        end = com[4]
        op_start = {"name": name, "ph": "B", "pid": device, "ts": start}
        op_end = {"name": name, "ph": "E", "pid": device, "ts": end}
        op_list.append(op_start)
        op_list.append(op_end)
    # print(op_list)
    f.close()
    f_json = open(file_path+'.json', 'w')
    f_json.writelines(json.dumps(op_list))
    f_json.flush()
    f_json.close()


if __name__ == "__main__":
    # read_bench_result("/mnt/d/home/Projects/DAG-scheduler/mnn/inception-v3/tmp.csv")
    # read_bench_result("inception-v3/tmp.csv")
    read_bench_result("../models/pnasnet-large/vivo_z3/tmp.csv")
