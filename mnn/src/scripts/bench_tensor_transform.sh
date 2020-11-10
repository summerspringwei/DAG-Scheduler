ANDROID_DIR=/data/local/tmp
MODEL=$1
MOBILE=$2
THREAD=2
DEVICE_MAP_FILE1=mDeviceMap-$MODEL-cpu-to-gpu-tensor-trans.txt
DEVICE_MAP_FILE2=mDeviceMap-$MODEL-gpu-to-cpu-tensor-trans.txt

# Bench cpu-2-gpu
python profile/profile_tensor_transform.py $MODEL $MOBILE 2
adb push ../models/$MODEL/$DEVICE_MAP_FILE1 $ANDROID_DIR
# echo "LD_LIBRARY_PATH=$ANDROID_DIR:$LD_LIBRARY_PATH $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL 7 3 1 2 1 $ANDROID_DIR/$DEVICE_MAP_FILE1 > $ANDROID_DIR/tmp.txt"
# exit 0
adb shell "MNN_LAYOUT_CONVERTER=cpu MNN_USE_CACHED=false LD_LIBRARY_PATH=$ANDROID_DIR:$LD_LIBRARY_PATH $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL 7 3 1 2 1 $ANDROID_DIR/$DEVICE_MAP_FILE1 > $ANDROID_DIR/tmp.txt"
adb shell "cat $ANDROID_DIR/tmp.txt | grep TensorConvert | grep copyToDevice > $ANDROID_DIR/$MODEL-$MOBILE-c2g-multi-data-trans.txt"
adb shell "cat $ANDROID_DIR/tmp.txt | grep TensorConvert | grep copyFromDevice > $ANDROID_DIR/$MODEL-$MOBILE-g2c-multi-data-trans.txt"

adb push ../models/$MODEL/$DEVICE_MAP_FILE2 $ANDROID_DIR
adb shell "MNN_LAYOUT_CONVERTER=cpu MNN_USE_CACHED=false LD_LIBRARY_PATH=$ANDROID_DIR:$LD_LIBRARY_PATH $ANDROID_DIR/benchmark.out $ANDROID_DIR/mnn_models/mnn_$MODEL 7 3 1 2 1 $ANDROID_DIR/$DEVICE_MAP_FILE2 > $ANDROID_DIR/tmp.txt"
adb shell "cat $ANDROID_DIR/tmp.txt | grep TensorConvert | grep copyToDevice >> $ANDROID_DIR/$MODEL-$MOBILE-c2g-multi-data-trans.txt"
adb shell "cat $ANDROID_DIR/tmp.txt | grep TensorConvert | grep copyFromDevice >> $ANDROID_DIR/$MODEL-$MOBILE-g2c-multi-data-trans.txt"

adb pull $ANDROID_DIR/$MODEL-$MOBILE-c2g-multi-data-trans.txt ../models/$MODEL/$MOBILE/
adb pull $ANDROID_DIR/$MODEL-$MOBILE-g2c-multi-data-trans.txt ../models/$MODEL/$MOBILE/
python profile/data_trans_generate.py $MODEL $MOBILE $THREAD
