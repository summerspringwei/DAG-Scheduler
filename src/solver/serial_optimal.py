from read_profile_data import *

# GPU_SCALE_FACTOR = 417.0 / 655
GPU_SCALE_FACTOR = 1
GPU_SCALE_FACTOR = 1
# For the GPU latency, we need to force the OpenCL command queue finish `OpenCLBackend->onWaitFinish()`
# The GPU latency will increase when comparing with the version without `OpenCLBackend->onWaitFinish()`
def serial_optimal(file_path):
    f = open(file_path, 'r')
    operator_latency_dict = {}
    # Correspond to # thread 1,2,4
    serial_optimal = [0, 0, 0]
    for line in f.readlines():
        com = line.strip().split(" ")
        if len(com) < 5:
            continue
        op_name = com[0].split('/')[-1]
        GPU_latency = float(com[4]) * GPU_SCALE_FACTOR
        for i in range(3):
            CPU_latency = float(com[i+1])
            if CPU_latency > GPU_latency:
                print("thread %d CPU (%f %f)" % (i, CPU_latency, GPU_latency))
                serial_optimal[i] += GPU_latency
            else:
                print("thread %d GPU (%f %f)" % (i, CPU_latency, GPU_latency))
                serial_optimal[i] += CPU_latency
    print(serial_optimal)


if __name__ == "__main__":
    # serial_optimal("pnasnet-mobile/redmi-pnasnet-mobile-latency-onwait.csv")
    # serial_optimal("inception-v3/redmi-inception-v3-layerwise-latency.csv")
    # serial_optimal("lanenet/oneplus-3-lanenet-layerwise-latency.csv")
    # serial_optimal("inception-v4/redmi-inception-v4-layerwise-latency.csv")
    serial_optimal("pnasnet-large/oneplus3-pnasnet-large-latency-onwait.csv")