ANDROID_DIR=/data/local/tmp
MODEL=$1
MOBILE=$2
DEVICE=cpu
THREAD=1



adb shell "MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 0 $THREAD 2 0"
adb pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-big-$THREAD.csv
echo "Write to ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-$THREAD.csv"
echo "Bench $MODEL thread $THREAD done"
sleep 10

THREAD=2
adb shell "MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 0 $THREAD 2 0"
adb pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-big-$THREAD.csv
echo "Bench $MODEL thread $THREAD done"
sleep 10

THREAD=4
adb shell "MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 0 $THREAD 2 0"
adb pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-big-$THREAD.csv
echo "Bench $MODEL thread $THREAD done"
sleep 10

DEVICE=gpu
THREAD=1
adb shell "MNN_USE_CACHED=false MNN_LAYOUT_CONVERTER=CPU LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 3 $THREAD 2 1 $ANDROID_DIR/pnasnet-large-final-layer.txt"
adb pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-$THREAD.csv
echo "Bench $MODEL thread $THREAD done"
