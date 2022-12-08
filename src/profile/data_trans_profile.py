import os
import subprocess


def count_duplicate_data_trans():
    sh_cmd = "adb shell cat /data/local/tmp/tmp.txt | grep pointer"
    print(sh_cmd)
    result = str(subprocess.check_output(sh_cmd, shell=True))
    lines = result.split("\\n")
    tensor_set = set()
    tensor_count = 0
    for line in lines:
        line = line.strip()
        tensor_count += 1
        com = line.split(" ")
        if(len(com) != 3):
            print(com)
            continue
        tensor_set.add(com[2])
    print("{} {}".format(len(tensor_set), tensor_count))
    print(tensor_set)


def data_trans_overhead():
    sh_cmd = 'adb shell cat /data/local/tmp/profile.txt | grep \"/Convert\" | awk \'{{print $6}}\''
    print(sh_cmd)
    result = str(subprocess.check_output(sh_cmd, shell=True))
    result = result[2:len(result)-1]
    lines = result.split("\\n")
    sum_of_latency = 0.0
    for line in lines:
        line = line.strip()
        if line != "":
            latency = float(line)
            sum_of_latency += latency
    loop = 10
    print("{} {}".format(sum_of_latency/loop/1000, int(len(lines)/loop)))

if __name__ == "__main__":
    count_duplicate_data_trans()
    data_trans_overhead()