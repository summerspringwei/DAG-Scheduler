MOBILE=$1
echo "" > tmp.txt
for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-large
    do
    python3 scripts/set_cpu_freq.py l
    ./scripts/bench_tensor_transform.sh $MODEL $MOBILE > tmp2.txt
    sleep 10
    ./scripts/bench_freq.sh $MODEL $MOBILE 4 >> tmp.txt
    sleep 10
    done
cat tmp.txt | grep avg | awk '{print $15}'