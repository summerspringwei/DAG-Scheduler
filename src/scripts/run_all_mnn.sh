set -xe
ANDROID_DIR=/data/local/tmp
SOLVRE=$1
OPT=$2
adb shell 'export ANDROID_DIR=/data/local/tmp && cd $ANDROID_DIR && echo "" > $ANDROID_DIR/tmp.txt'

FREQ=4
python ./scripts/set_cpu_freq.py h

for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-large
    do
    for THREAD in 4
    do
        if [ $SOLVRE == 'i' ]
        then
        adb shell "cd $ANDROID_DIR && ./irun_mnn.sh $MODEL $THREAD $FREQ >> $ANDROID_DIR/tmp.txt 2>&1"
        elif [ $SOLVRE == 'g' ]
        then
        adb shell "cd $ANDROID_DIR && ./grun_mnn.sh $MODEL $THREAD $FREQ >> $ANDROID_DIR/tmp.txt 2>&1"
        elif [ $SOLVRE == 'h' ]
        then
        adb shell "cd $ANDROID_DIR && ./hrun_mnn.sh $MODEL $THREAD $FREQ >> $ANDROID_DIR/tmp.txt 2>&1"
        elif [ $SOLVRE == 'm' ]
        then
        adb shell "cd $ANDROID_DIR && ./mosaic_run_mnn.sh $MODEL $THREAD $FREQ >> $ANDROID_DIR/tmp.txt 2>&1"
        else echo "Argument error"
        fi
        sleep 10
    done
done
adb shell "cat $ANDROID_DIR/tmp.txt | grep avg"

