

# from read_data_trans import *
import json
import os

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


def prepare_once_bench():
    profile_file_name = "/tmp/profile.txt"
    sh_cmd = "adb pull /data/local/tmp/profile.txt {}".format(profile_file_name)
    f = open(profile_file_name)
    num_of_lines = len(f.readlines())
    loop_count = 10
    num_of_lines = num_of_lines / 10
    print(sh_cmd)
    os.system(sh_cmd)
    csv_file_path = "/tmp/profile.csv"
    sh_cmd = "tail -n {} {}  | grep Iter | awk '{print $3, $5, $6, $7, $8}' > {}".format(num_of_lines, profile_file_name, csv_file_path)
    print(sh_cmd)
    os.system(sh_cmd)
    return csv_file_path
    

if __name__ == "__main__":
    # read_bench_result("/mnt/d/home/Projects/DAG-scheduler/mnn/inception-v3/tmp.csv")
    # read_bench_result("inception-v3/tmp.csv")
    csv_file_path = prepare_once_bench()
    read_bench_result(csv_file_path)
