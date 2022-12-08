set -x
ANDROID_DIR=/data/local/tmp
MODEL=$1
MOBILE=$2
OPT=$3
DEVICE=cpu
THREAD=1

CPU_TYPE=big
AFFINITY=c0

python3 scripts/set_cpu_freq.py h

ACL_MODEL=acl-$MODEL


gatherProfle(){
    adb $OPT pull $ANDROID_DIR/profile.txt ../models/$ACL_MODEL/$MOBILE/
    cat ../models/$ACL_MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$ACL_MODEL/$MOBILE/$MOBILE-$ACL_MODEL-$DEVICE-${CPU_TYPE}-$THREAD.csv
    echo "Write to ../models/$ACL_MODEL/$MOBILE/$MOBILE-$ACL_MODEL-$DEVICE-$THREAD.csv"
    echo "Bench $ACL_MODEL on cpu thread $THREAD done"
    sleep 10
}
# THREAD=1
# DEVICE=gpu
# gatherProfle
# exit

adb $OPT shell "LD_LIBRARY_PATH=$ANDROID_DIR taskset $AFFINITY $ANDROID_DIR/graph_${MODEL} --target=Neon --threads=${THREAD} --layout=NCHW --execution-type=default --num_runs=10"
gatherProfle

THREAD=2
adb $OPT shell "LD_LIBRARY_PATH=$ANDROID_DIR  taskset $AFFINITY $ANDROID_DIR/graph_${MODEL} --target=Neon --threads=${THREAD} --layout=NCHW --execution-type=default --num_runs=10"
gatherProfle

THREAD=4
adb $OPT shell "LD_LIBRARY_PATH=$ANDROID_DIR  taskset $AFFINITY $ANDROID_DIR/graph_${MODEL} --target=Neon --threads=${THREAD} --layout=NCHW --execution-type=default --num_runs=10"
gatherProfle

DEVICE=gpu
THREAD=1
adb $OPT shell "LD_LIBRARY_PATH=$ANDROID_DIR  taskset f0 $ANDROID_DIR/graph_${MODEL} --target=CL --threads=${THREAD} --layout=NCHW --execution-type=default --num_runs=10"
gatherProfle

python3 analyze/measure_interference.py acl-$MODEL $MOBILE 2
