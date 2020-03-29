
ANDROID_DIR=/data/local/tmp
MODEL=pnasnet-large
MOBILE=lenovo_k5
THREAD=4

python generate_Lp.py $MODEL $MOBILE $THREAD
cat ../models/$MODEL/$MOBILE/lp-result-subgraphs-* | grep Objective | awk "{print \$4}"
cat ../models/$MODEL/$MOBILE/mDeviceMap-subgraphs-* > ../models/$MODEL/$MOBILE/mDeviceMap-$MODEL-cpu-$THREAD.txt
# adb push ../model/$MODEL/$MOBILE/mDeviceMap-$MODEL-cpu-$THREAD.txt $ANDROID_DIR