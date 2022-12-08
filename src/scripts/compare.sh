
set -xe
ANDROID_DIR=/data/local/tmp
MODEL=$1
MOBILE=$2
OPT=$3
DEVICE=cpu
THREAD=2


adb $OPT shell "MNN_USE_CACHED=true MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 3 $THREAD 2 1 $ANDROID_DIR/pnasnet-large-final-layer.txt"
echo "Bench $MODEL on GPU thread $THREAD done"

adb $OPT shell "MNN_USE_CACHED=true MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 0 $THREAD 2 0"
echo "Bench $MODEL on CPU thread $THREAD done"

# adb $OPT shell "MNN_USE_CACHED=true MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 3 $THREAD 2 2 mDeviceMap-$MODEL-cpu-$THREAD.txt"
adb $OPT shell "cd /data/local/tmp/ && source set_env.sh && ./run_mnn.sh $MODEL $THREAD"
echo "Bench $MODEL on CPU+GPU thread $THREAD done"
