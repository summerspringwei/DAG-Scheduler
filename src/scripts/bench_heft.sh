ANDROID_DIR=/data/local/tmp
MODEL=$1
MOBILE=$2
THREAD=$3
OPT=$4
DEVICE_MAP_FILE=../models/$MODEL/$MOBILE/heft-placement-$MODEL-cpu-$THREAD.txt

python3 solver/heft_dag_scheduler.py $MODEL $MOBILE $THREAD
adb $OPT push $DEVICE_MAP_FILE $ANDROID_DIR
