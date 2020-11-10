import subprocess
import os
import time
# redmi 652800 1036800 1401600 1689600 1804800 1958400 2016000
# avaliable governors: interactive conservative ondemand userspace powersave performance
# 855 超大核心： 825600 940800 1056000 1171200 1286400 1401600 1497600 1612800 1708800 1804800 
# 1920000 2016000 2131200 2227200 2323200 2419200 2534400 2649600 2745600 2841600
mobiles_cpu_freq_map = {
    "redmi": [652800, 1036800, 1401600, 1689600, 1804800, 1958400, 2016000],
    "snapdragon_855": [710400, 825600, 940800, 1056000, 1171200, 1286400, 1401600, \
        1497600, 1612800, 1708800, 1804800, 1920000, 2016000, 2131200, 2227200, 2323200, 2419200], # Big cores
}


def get_avaliable_cpu_freq():
    cmd = "adb shell cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies"
    process = subprocess.Popen(cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result_f = process.stdout.read()
    print(str(result_f).strip().split(" "))

# Lighter: lightweight heterogenous inference 
# cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
def set_cpu_freq(min_cpu_freq, max_cpu_freq, cores):
    for c in cores:
        cmd = 'adb shell "echo {} >  /sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq"'.format(max_cpu_freq, c)
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result_f = process.stdout.read()
        print(result_f)
        cmd = 'adb shell "echo {} >  /sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq"'.format(min_cpu_freq, c)
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result_f = process.stdout.read()
        print(result_f)


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



if __name__=="__main__":
    # redmi_cpu_freqs = [652800, 1036800, 1401600, 1689600, 1804800, 1958400, 2016000]
    # for freq in cpu_freqs:
    #     set_cpu_freq(freq, freq, [0,1,2,3,4,5,6,7])
    #     bench_tflite_deepspeech_cmd()
    #     time.sleep(5)
    # freq = 2016000
    # set_cpu_freq(652800, 2016000, [0,1,2,3,4,5,6,7])
    # set_cpu_governors("performance", [0,1,2,3,4,5,6,7])
    # mobile_name = "snapdragon_855"
    # set_cpu_freq(mobiles_cpu_freq_map[mobile_name][-1], mobiles_cpu_freq_map[mobile_name][-1], [4,5,6,7])
    mobile_name = "redmi"
    set_cpu_freq(mobiles_cpu_freq_map[mobile_name][-1], mobiles_cpu_freq_map[mobile_name][-1], [0,1,2,3,4,5,6,7])
