
set -xe
open /Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/pnasnet-large/pnasnet-large.png
read -n 1
adb shell "cd /data/local/tmp && source set_env.sh && ./cpu-run-pnasnet.sh"
read -n 1
adb shell "cd /data/local/tmp && source set_env.sh && ./gpu-run-pnasnet.sh"
read -n 1
python solver/generate_LP.py pnasnet-large huawei_p40 2
read -n 1
open /Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/pnasnet-large/huawei_p40/ilp-graphviz-pnasnet-large-cpu-2.png
read -n 1
adb shell "cd /data/local/tmp && source set_env.sh && ./run_mnn.sh pnasnet-large 2"

