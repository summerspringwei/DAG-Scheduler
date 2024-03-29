
ANDROID_DIR=/data/local/tmp
MODEL=$1
MOBILE=$2
THREAD=$3
OPT=$4
DEVICE_MAP_FILE=../models/$MODEL/$MOBILE/mDeviceMap-$MODEL-cpu-$THREAD.txt

python3 solver/generate_LP.py $MODEL $MOBILE $THREAD
adb $OPT push $DEVICE_MAP_FILE $ANDROID_DIR

# cat ../models/$MODEL/$MOBILE/mDeviceMap-subgraphs-* > $DEVICE_MAP_FILE
# cat ../models/$MODEL/$MOBILE/mDeviceMap-serial-$MODEL-cpu-$THREAD.txt >> $DEVICE_MAP_FILE
# adb push $DEVICE_MAP_FILE $ANDROID_DIR
