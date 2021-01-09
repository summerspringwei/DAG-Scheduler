MOBILE=$1
for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-mobile nasnet-large
do
  for THREAD in 1 2 4
  do
    python solver/greedy_device_placement.py $MODEL $MOBILE $THREAD > tmp.txt
    cat tmp.txt | grep "Greedy Result"
  done
done
