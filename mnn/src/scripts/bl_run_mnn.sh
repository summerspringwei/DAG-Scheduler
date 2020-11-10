MODEL=$1
THREAD=$2
./benchmark.out mnn_models/mnn_$MODEL 7 0 $THREAD 2 3 mDeviceMap-$MODEL-cpu-big-$THREAD-little-4.txt 4