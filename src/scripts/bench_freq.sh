set -xe
ANDROID_DIR=/data/local/tmp
MODEL=$1
MOBILE=$2
THREAD=$3
OPT=$4
DEVICE=cpu

adb $OPT push ../models/pnasnet-large/pnasnet-large-final-layer.txt /data/local/tmp/

python3 scripts/set_cpu_freq.py l
adb $OPT shell "MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 0 $THREAD 2 0"
adb $OPT pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-big-$THREAD.csv
echo "Write to ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-$THREAD.csv"
echo "Bench $MODEL on CPU thread $THREAD  freq low done"
sleep 10

python3 scripts/set_cpu_freq.py m
adb $OPT shell "MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 0 $THREAD 2 0"
adb $OPT pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-big-$THREAD.csv
echo "Bench $MODEL on CPU thread $THREAD freq medium done"
sleep 10

python3 scripts/set_cpu_freq.py h
adb $OPT shell "MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 0 $THREAD 2 0"
adb $OPT pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-big-$THREAD.csv
echo "Bench $MODEL on CPU thread $THREAD  freq high done"
sleep 10

DEVICE=gpu
THREAD=1
adb $OPT shell "MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 3 $THREAD 2 1 $ANDROID_DIR/pnasnet-large-final-layer.txt"
adb $OPT pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-$THREAD.csv
echo "Bench $MODEL on GPU done"

python3 analyze/measure_interference.py $MODEL $MOBILE 2
