ANDROID_DIR=/data/local/tmp
MODEL=inception-v3
MOBILE=vivo_z3
DEVICE=cpu
THREAD=4


adb shell "LD_LIBRARY_PATH=$ANDROID_DIR taskset f0 $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL/ 7 0 $THREAD 2 0 $ANDROID_DIR/mDevice_map_pnasnet-large_alone.txt"
adb pull $ANDROID_DIR/profile.txt ../models/$MODEL/$MOBILE/
cat ../models/$MODEL/$MOBILE/profile.txt | grep Iter | awk '{print $3, $5, $6, $7, $8}' > ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-$DEVICE-$THREAD.csv
