set -xe
MOBILE=huawei_p40
for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-large
do
    python solver/greedy_device_placement.py $MODEL huawei_p40 2 4 > tmp.txt 2>&1
    cat tmp.txt | grep "Greedy"
    # python analyze/measure_interference.py $MODEL huawei_p40 2
    # ls -lth ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-layerwise-latency.csv
    # ls -lth ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-cpu-little-1.csv
    # cp ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-layerwise-latency.csv ../models/$MODEL/$MOBILE/$MOBILE-$MODEL-layerwise-latency.csv.thread.backup
    # cp ../models/$MODEL/$MOBILE/$MODEL-$MOBILE-data-trans.csv ../models/$MODEL/$MOBILE/$MODEL-$MOBILE-data-trans.csv.thread.backup
done
