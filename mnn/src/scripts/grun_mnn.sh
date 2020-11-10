
MODEL=$1
THREAD=$2
MNN_USE_CACHED=true MNN_LAYOUT_CONVERTER=CPU ./benchmark.out mnn_models/mnn_$MODEL 7 3 $THREAD 2 2 greedy-placement-$MODEL-cpu-$THREAD.txt
