MODEL=$1
THREAD=$2
if [ ! -n "$3" ] ;then
    FREQ=$THREAD
else
    FREQ=$3
fi
echo $FREQ

MNN_USE_CACHED=true MNN_LAYOUT_CONVERTER=cpu LD_LIBRARY_PATH="/data/local/tmp/":$LD_LIBRARY_PATH ./benchmark.out mnn_models/mnn_$MODEL 7 3 $THREAD 2 2 mDeviceMap-$MODEL-cpu-$FREQ.txt
