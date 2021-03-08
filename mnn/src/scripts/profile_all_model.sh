set -xe
MOBILE=$1
echo "" > tmp.txt
for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-large
    do
    # ./scripts/bench_tensor_transform.sh $MODEL $MOBILE > tmp2.txt
    # sleep 10
    # ./scripts/bench_alone.sh $MODEL $MOBILE >> tmp.txt
    # sleep 10
    ./scripts/bench_alone-little.sh $MODEL $MOBILE >> tmp.txt
    sleep 10
    done
cat tmp.txt | grep avg | awk '{print $15}'