import os
import numpy as np

def get_tensors_shape(file_path):
    f = open(file_path)
    lines = f.readlines()
    tensor_shape_list = []
    for line in lines:
        com = line.split(" ")
        com[0] = com[0].replace(',', ' ')
        tensor_shape_list.append(com[0])
    f.close()
    return tensor_shape_list


def auto_generate_tflite_tensors_shape(file_path):
    f = open(file_path)
    lines = f.readlines()
    tensor_shape_list = []
    for line in lines:
        com = line.split(" ")
        shapes = com[0].split(',')
        if len(shapes) != 4:
            continue
        result = "BHWC({},{},{},{})".format(shapes[0], shapes[2], shapes[3], shapes[1])
        tensor_shape_list.append(result)
    f.close()
    return tensor_shape_list


def bench_tensor_transform_on_device(tensor_shape_list):
    android_path = "/data/local/tmp/"
    executable = "./benchmarkTensorTransform"
    tensor_shape_trans_copy_latency = []
    tensor_shape_trans_all_latency = []
    for tensor_shape in tensor_shape_list:
        tensor_shape_trans_copy_latency.clear()
        tensor_shape_trans_all_latency.clear()
        sh_cmd = 'adb shell "cd {} && source set_env.sh && {} {} | grep TensorConvert > tmp.txt"'.format(android_path, executable, tensor_shape)
        print(sh_cmd)
        os.system(sh_cmd)
        sh_cmd = "adb pull /data/local/tmp/tmp.txt /tmp/"
        os.system(sh_cmd)
        f = open("/tmp/tmp.txt")
        lines = f.readlines()
        first = True
        for line in lines:
            if first:
                first = False
                continue
            com = line.split(" ")
            copy_latency = int(com[1])
            overall_latency = int(com[4])
            tensor_shape_trans_copy_latency.append(copy_latency)
            tensor_shape_trans_all_latency.append(overall_latency)
        avg_copy_latency = np.average(tensor_shape_trans_copy_latency)
        avg_all_latency = np.average(tensor_shape_trans_all_latency)
        tensor_size = 1
        for c in tensor_shape.split(' '):
            tensor_size = tensor_size * int(c)
        tensor_shape = tensor_shape.replace(" ", ",")
        print("result: {} {} {} {}".format(tensor_shape, tensor_size, avg_copy_latency, avg_all_latency-avg_copy_latency))
    

# Convert tensor layout to NCHW
def preprocess_tensor_shape(tensor_shape_list):
    new_shape_list = []
    for shape_str in tensor_shape_list:
        if len(shape_str.split(" ")) < 4:
            continue
        tensor_shape = [int(s) for s in shape_str.split(" ")]
        if tensor_shape[1] == tensor_shape[2]:#NHWC -> NCHW
            tensor_shape = [tensor_shape[0], tensor_shape[3], tensor_shape[1], tensor_shape[2]]
        new_shape_list.append(tensor_shape)
        print(tensor_shape)
    
    new_shape_list = list(set(["{} {} {} {}".format(s[0], s[1], s[2], s[3]) for s in new_shape_list]))
    new_shape_list.sort(key=lambda s: int(s.split(" ")[1]))
    print('-'*10)
    for s in new_shape_list:
        print(s)
    return new_shape_list


def bench_tensor_transform_on_CPU(tensor_shape_list):
    android_path = "/data/local/tmp/"
    # executable = "test_c4hw4_to_image"
    executable = "benchmarkTensorTransform"
    sh_cmd = 'adb shell "echo \"\" > /data/local/tmp/tmp.txt"'
    print(sh_cmd)
    os.system(sh_cmd)
    for tensor_shape in tensor_shape_list:
        sh_cmd = 'adb shell "source /data/local/tmp/set_env.sh && {} {} >> /data/local/tmp/tmp.txt"'.format(os.path.join(android_path, executable), tensor_shape)
        print(sh_cmd)
        os.system(sh_cmd)
    print("Done")


def get_tflite_tensor_data_trans(file_path):
    f = open(file_path)
    lines = f.readlines()
    old_tensor_shape = None
    tensor_shape_trans_copy_latency = []
    tensor_shape_trans_all_latency = []
    for line in lines:
        com = line.strip().split(" ")
        if len(com) < 5:
            continue
        tensor_shape = com[2]
        copy_latency = int(com[1])
        overall_latency = int(com[4])
        if old_tensor_shape == None or old_tensor_shape == tensor_shape:
            tensor_shape_trans_copy_latency.append(copy_latency)
            tensor_shape_trans_all_latency.append(overall_latency)
            old_tensor_shape = tensor_shape
        else:
            avg_copy_latency = np.average(tensor_shape_trans_copy_latency)
            avg_all_latency = np.average(tensor_shape_trans_all_latency)
            tensor_size = 1
            for c in tensor_shape.split(','):
                tensor_size = tensor_size * int(c)
            print("{} {} {} {}".format(old_tensor_shape, tensor_size, avg_copy_latency, avg_all_latency-avg_copy_latency))
            tensor_shape_trans_copy_latency.clear()
            tensor_shape_trans_all_latency.clear()
            tensor_shape_trans_copy_latency.append(copy_latency)
            tensor_shape_trans_all_latency.append(overall_latency)
            old_tensor_shape = tensor_shape
    

if __name__ == "__main__":
    # tensor_shape_list = get_tensors_shape("/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/inception-v3/redmi/inception-v3-redmi-data-trans.csv")
    tensor_shape_list = get_tensors_shape("/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/pnasnet-large/redmi/pnasnet-large-redmi-data-trans.csv")
    new_tensor_shape_list = preprocess_tensor_shape(tensor_shape_list)
    print(new_tensor_shape_list)
    # bench_tensor_transform_on_device(tensor_shape_list)
    # exit(0)
    bench_tensor_transform_on_CPU(new_tensor_shape_list)
    # a = str(auto_generate_tflite_tensors_shape("/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/inception-v3/redmi/inception-v3-redmi-data-trans.csv"))
    # a = a.replace('\'', "")
    # print(a)
    # get_tflite_tensor_data_trans("/tmp/tmp.txt")