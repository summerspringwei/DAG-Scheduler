#!python3
import subprocess
import os
import time
import argparse

def get_avaliable_cpu_freq(core_id):
    cmd = "adb shell cat /sys/devices/system/cpu/cpu{}/cpufreq/scaling_available_frequencies".format(core_id)
    process = subprocess.Popen(cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result_f = process.stdout.read()
    return str(result_f, encoding = "utf-8").strip().split(" ")

# Lighter: lightweight heterogenous inference 
# cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
def set_cpu_freq(min_cpu_freq, max_cpu_freq, cores):
    cmd = "adb devices"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result_f = str(process.stdout.read(), encoding='utf-8')
    is_mi9 = False
    if result_f.find("e130cb65") >= 0:
        is_mi9 = True
    
    for c in cores:
        if is_mi9:
            cmd = 'adb shell "su -c \'echo {} >  /sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq\'"'.format(max_cpu_freq, c)
        else:
            cmd = 'adb shell "echo {} >  /sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq"'.format(max_cpu_freq, c)
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result_f = str(process.stdout.read(), encoding='utf-8')
        print(result_f)
        if is_mi9:
            cmd = 'adb shell "su -c \'echo {} >  /sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq\'"'.format(min_cpu_freq, c)
        else:
            cmd = 'adb shell "echo {} >  /sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq"'.format(min_cpu_freq, c)
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result_f = str(process.stdout.read(), encoding='utf-8')
        print(result_f)



class PerformanceMode:
    PerformanceHigh = 0
    PerformanceMedium = 1
    PerformanceLow = 2
    PerformanceDefault = 3


def reset_freq():
    for core_id in range(8):
        cpu_freq_list = get_avaliable_cpu_freq(core_id)
        set_cpu_freq(cpu_freq_list[0], cpu_freq_list[-1], [core_id])


# Default 8 cores
def set_performance(mode):
    """Set all CPU cores frequency based on the mode.
    """
    if mode == PerformanceMode.PerformanceDefault:
        reset_freq()
        return
    for core_id in range(8):
        cpu_freq_list = get_avaliable_cpu_freq(core_id)
        freq = cpu_freq_list[-1]
        if mode == PerformanceMode.PerformanceHigh:
            freq = cpu_freq_list[-1]
        elif mode == PerformanceMode.PerformanceMedium:
            freq = cpu_freq_list[len(cpu_freq_list)//2]
        else:
            freq = cpu_freq_list[0]
        set_cpu_freq(freq, freq, [core_id])



def set_cpu_governors(gov, cores):
    for c in cores:
        cmd = 'adb shell "echo 1 > /sys/devices/system/cpu/cpu{}/online"'.format(c)
        os.system(cmd)
        cmd = 'adb shell "echo {} >  /sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor"'.format(gov, c)
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result_f = process.stdout.read()
        print(result_f)


def bench_tflite_resnet_v1_50_cmd():
    sh_cmd = 'adb shell /data/local/tmp/tflite_benchmark_model --graph=/sdcard/dnntune_models/resnet-v1-50.tflite \
        --input_layer=input --input_layer_shape=1,224,224,3 --num_threads=4'
    print(sh_cmd)
    os.system(sh_cmd)

def bench_tflite_deepspeech_cmd():
    sh_cmd = 'adb shell /data/local/tmp/tflite_benchmark_model --graph=/sdcard/dnntune_models/deepspeech.tflite \
        --input_layer=input_node,previous_state_c,previous_state_h --input_layer_shape=1,16,19,26:1,2048:1,2048 --num_threads=4 --num_runs=10'
    print(sh_cmd)
    os.system(sh_cmd)


def parse_freq():
    parser = argparse.ArgumentParser(description='Set CPU frequency.')
    parser.add_argument('mode', type=str, help='Enter the performance mode')
    args = parser.parse_args()
    mode = args.mode
    if mode == "l" or mode == "low":
        return PerformanceMode.PerformanceLow
    elif mode == "m" or mode == "medium":
        return PerformanceMode.PerformanceMedium
    elif mode == "h" or mode == "high":
        return PerformanceMode.PerformanceHigh
    elif mode== 'r' or mode == "reset":
        return PerformanceMode.PerformanceDefault

if __name__=="__main__":
    mode = parse_freq()
    for i in range(8):
        print(get_avaliable_cpu_freq(i))
    set_performance(mode)
    set_cpu_governors('performance', range(8))