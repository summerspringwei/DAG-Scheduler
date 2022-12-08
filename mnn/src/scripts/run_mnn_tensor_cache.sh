MODEL=$1
THREAD=$2
set -xe
export MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=cpu
./benchmark.out mnn_models/mnn_$MODEL 7 3 $THREAD 2 1 mDeviceMap-$MODEL-cpu-$THREAD.txt 
sleep 5

export MNN_USE_CACHED=true MNN_LAYOUT_CONVERTER=cpu
./benchmark.out mnn_models/mnn_$MODEL 7 3 $THREAD 2 1 mDeviceMap-$MODEL-cpu-$THREAD.txt 

