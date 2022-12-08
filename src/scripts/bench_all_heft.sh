MOBILE=$1
echo "" > tmp.txt
for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-large
do
  for THREAD in 1 2 4
  do
    ./scripts/bench_heft.sh $MODEL $MOBILE $THREAD >> tmp.txt 2>&1
  done
done
cat tmp.txt | grep "result"  | awk '{print $7}'
