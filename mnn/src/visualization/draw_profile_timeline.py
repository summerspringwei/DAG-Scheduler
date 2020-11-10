
import os

def read_profile():
    profile_file_name = "/tmp/profile.txt"
    sh_cmd = "adb pull /data/local/tmp/profile.txt {}".format(profile_file_name)
    print(sh_cmd)
    os.system(sh_cmd)
    sh_cmd = "cat {}  | grep Iter | awk '{print $3, $5, $6, $7, $8}' > /tmp/profile.csv"
    f = open(profile_file_name)
    lines = f.readlines()
    sh_cmd = "cat {}  | grep Iter | awk '{print $3, $5, $6, $7, $8}' > /tmp/profile.csv"
    for line in lines:
        com = line.split(" ")
        assert(len(com)>7)
        op_name = com[2]
        device = com[4]
        latency = float(com[5])
        start = float(com[6])