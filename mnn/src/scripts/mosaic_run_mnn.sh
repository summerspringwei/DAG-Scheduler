MODEL=$1
THREAD=$2
./benchmark.out mnn_models/mnn_$MODEL 7 3 $THREAD 2 1 mosaic-placement-$MODEL-cpu-$THREAD.txt
