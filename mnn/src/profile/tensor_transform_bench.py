
import os

def read_tensors(file_path):
    cmd_get_tensors = "cat %s | awk '{print $3}'" % (file_path)
    lines = os.popen(cmd_get_tensors)
    tensors = set()
    for line in lines:
        # remove the 
        line = line[0:len(line)-2]
        # print(line)
        com = line.split(',')
        # check whether the tensor dimensions have zero values
        for c in com:
            if c != '' and c != 'none' and int(c) == 0:
                continue
        line = line.replace(',', ' ')
        print(line)
        tensors.add(line)
    return tensors


if __name__ == "__main__":
    read_tensors("../models/inception-v4/inception-v4-info.txt")
