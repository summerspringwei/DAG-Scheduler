MOBILE=$1
for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-mobile nasnet-large
do
  for THREAD in 1 2 4
  do
    python solver/mosiac_dp.py $MODEL $MOBILE $THREAD > tmp.txt
    cat tmp.txt | grep "MOSAIC result"
  done
done
